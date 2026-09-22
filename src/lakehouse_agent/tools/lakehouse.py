"""The Lakehouse tools the model calls: find a series, fetch it, derive from it.

The brief's "Lakehouse" tool (plan 3.3), as three functions over the lake
surface in docs/contracts.md. `ctx.lake` is a `LakeAdapter` in production and
anything with the same methods -- `FakeLake` -- in a test.

Rules all three follow, and why:

* **The model sees a preview, never the table** (contracts.py `ToolResult`).
  First and last rows, plus a summary that already states what the model
  would otherwise be tempted to compute.
* **Tables are monthly.** `get_series` aligns everything to months, by each
  series' kind (a balance takes the month's last value, a rate its average,
  a flow its sum), because the scenario's questions are monthly and a mixed
  weekly/monthly table has almost no row where both are filled. Going the
  other way would invent data (CLAUDE.md #4), so a quarterly or yearly series
  is refused rather than spread over months. Every conversion is written into
  the column's lineage and into `warnings`, which reach the answer.
* **Existing columns are never touched.** Both "add to the table" paths add
  columns and refuse a name that is already taken: `merge` would rename the
  old column to `_x`, and the loop's "existing columns came back unchanged"
  check cannot see a column that is no longer there.
* **Nothing is filled** (CLAUDE.md #2). A month with no value stays empty and
  the summary says how many there are.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from lakehouse_agent.agent.context import TableNotFound, ToolContext
from lakehouse_agent.agent.formatting import preview_rows
from lakehouse_agent.analysis import transforms
from lakehouse_agent.contracts import (
    ColumnInfo,
    Kind,
    SeriesNotFound,
    Source,
    TableRef,
    ToolResult,
)
from lakehouse_agent.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

#: How many candidates search hands the model. The pool is small, and every
#: extra candidate is text the model has to read to rule out.
SEARCH_CANDIDATES = 5

#: TÜFE for deflation: TP.TUKFIY2025.GENEL, 2025=100, no chaining
#: (docs/decisions.md, "TÜFE serisi").
CPI_SERIES = "evds:TP.TUKFIY2025.GENEL"

#: How many flagged periods `mark` lists by name before summarising the rest.
MARKED_LISTED = 24

YEAR_MONTH = r"^\d{4}-(0[1-9]|1[0-2])$"

_FREQ_TR = {
    "D": "gunluk",
    "W": "haftalik",
    "M": "aylik",
    "Q": "uc aylik",
    "A": "yillik",
}
_FREQ_RANK = {"D": 0, "W": 1, "M": 2, "Q": 3, "A": 4}
_PERIODS_PER_YEAR = {"M": 12, "W": 52, "Q": 4, "A": 1}

#: What a period is called, and whose value a conversion takes.
_PERIOD_WORD = {"D": "gun", "W": "hafta", "M": "ay", "Q": "ceyrek", "A": "yil"}
_PERIOD_OF = {"D": "gunun", "W": "haftanin", "M": "ayin", "Q": "ceyregin", "A": "yilin"}

#: How a finer series becomes coarser, per the catalog's `agg_method`: the
#: rule `analysis.transforms.to_freq` applies inside `wide()`, in words.
_AGG_WORDS = {
    "last": "son degeri",
    "first": "ilk degeri",
    "sum": "degerlerinin toplami",
    "avg": "degerlerinin ortalamasi",
    "min": "en dusuk degeri",
    "max": "en yuksek degeri",
}
_KIND_TO_AGG = {"stock": "last", "flow": "sum", "rate": "avg", "index": "avg"}

#: A weekly value belongs to the period its observation day falls in: the
#: week of Monday 31 March 2025, observed Friday 4 April, is April's. (Until
#: 22 Sep it was the Monday's month, and 6 of 2025's 12 "March-end" style
#: values of TP.HPBITABLO6.3 were the next month's first Friday.)
_WEEK_RULE = "hafta, gozlendigi gunun (ornegin Cuma) donemine sayilir"


def _conversion_rule(
    source: str | None, target: str, agg_method: str | None, kind: str | None
) -> str | None:
    """How a `source`-frequency series became `target`, in words; None when
    it needed no conversion."""
    if not source or _FREQ_RANK.get(source, 99) >= _FREQ_RANK[target]:
        return None
    method = agg_method or _KIND_TO_AGG.get(kind or "")
    if method not in _AGG_WORDS:
        return None
    rule = f"{_PERIOD_OF[target]} {_AGG_WORDS[method]}"
    return f"{rule} ({_WEEK_RULE})" if source == "W" else rule


def _label(value: Any, freq: str) -> str:
    """A period as a person reads it at the table's frequency."""
    stamp = pd.Timestamp(value)
    if freq in ("D", "W"):
        return stamp.strftime("%Y-%m-%d")
    if freq == "Q":
        return f"{stamp.year}-Q{stamp.quarter}"
    if freq == "A":
        return str(stamp.year)
    return stamp.strftime("%Y-%m")


def column_name(series_id: str) -> str:
    """`evds:TP.KTF12` -> `TP_KTF12`, `bddk:aylik:...:sektor` -> `aylik_..._sektor`.

    Only the source prefix is stripped. BDDK ids use ':' as a hierarchy
    separator, so taking the last segment would collapse every consumer-credit
    series onto the name `sektor`. `.` and `:` go because the model copies a
    column name into the next call verbatim, and a name that looks like an
    id invites it to pass the id instead. The fakes use this same function,
    so a test written against them sees the names the real tools produce.
    """
    _, _, rest = series_id.partition(":")
    return (rest or series_id).replace(":", "_").replace(".", "_")


def _no_lake() -> ToolResult:
    return ToolResult(ok=False, summary="Veri havuzu bu oturuma bagli degil.")


def _clean(value: Any) -> Any:
    """Catalog cells as plain values: NaN/NA/'' become None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return None if value == "" else value


def _fetched_at(long: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Latest fetch time per series, from the long observations."""
    if long.empty or "fetched_at" not in long.columns:
        return {}
    stamps = pd.to_datetime(long["fetched_at"], utc=True, errors="coerce")
    latest = stamps.groupby(long["series_id"]).max()
    return {sid: stamp for sid, stamp in latest.items() if not pd.isna(stamp)}


def _month(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def _preview(frame: pd.DataFrame) -> list[dict]:
    """First and last rows for the model, floats rounded to four decimals.

    A monthly difference of 0.45 arrives from float arithmetic as
    0.4499999999999993, and a model shown that either copies the noise into
    the answer or rounds it itself -- the one thing planner rule 1 forbids.
    Only the preview is rounded; the table keeps every digit.
    """
    floats = frame.select_dtypes("float").columns
    return preview_rows(frame.round({column: 4 for column in floats}))


# --------------------------------------------------------------------------
# search_catalog
# --------------------------------------------------------------------------


class SearchCatalogArgs(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        description=(
            "Dogal dilde arama metni ya da seri kodu, ornek 'konut kredisi faiz "
            "orani' veya 'TP.KTF12'. Bir seferde tek bir kavram ara."
        ),
        examples=["konut kredisi faiz orani"],
    )


#: How a candidate was found, as the model reads it: "kod" means the query
#: named the series, and the series-choice gate leaves such hits alone.
_MATCH_TR = {"code": "kod", "semantic": "anlam", "keyword": "kelime"}


def make_search_catalog() -> ToolSpec:
    """Candidates from the pool, with what the model needs to choose among
    them: kind, unit, frequency and the months the pool actually holds."""

    def run(args: SearchCatalogArgs, ctx: ToolContext) -> ToolResult:
        lake = ctx.lake
        if lake is None:
            return _no_lake()
        detailed = getattr(lake, "search_detailed", None)
        if callable(detailed):
            hits, notes = detailed(args.query, SEARCH_CANDIDATES)
        else:
            hits, notes = lake.search(args.query, SEARCH_CANDIDATES), []

        if not hits and notes:
            # A degraded search (keyword fallback) finding nothing proves
            # nothing: "enflasyon" shares no word with the CPI's catalog text.
            # Saying the pool lacks it would put a false "we do not have
            # that" in the answer (planner rule 5).
            return ToolResult(
                ok=True,
                summary=(
                    f"'{args.query}' icin seri bulunamadi, ancak arama su an "
                    "yalnizca kelime eslesmesiyle calisiyor; bu, verinin havuzda "
                    "olmadigini gostermez. Baska kelimelerle ya da seri koduyla "
                    "tekrar ara. Yine bulunamazsa kullaniciya aramanin su an "
                    "sinirli oldugunu soyle."
                ),
                warnings=notes,
            )
        if not hits:
            return ToolResult(
                ok=True,
                summary=(
                    f"Havuzda '{args.query}' ile eslesen seri yok. Bu veri havuzda "
                    "degilse bunu kullaniciya acikca soyle; baska bir seriyle "
                    "yerine koyma."
                ),
                warnings=notes,
            )

        preview = [
            {
                "series_id": hit.series_id,
                "name_tr": hit.name_tr,
                "unit": hit.unit,
                "freq": hit.freq,
                "kind": hit.kind,
                "source": hit.source,
                "kapsam": (
                    f"{hit.start_date} - {hit.end_date}"
                    if hit.start_date and hit.end_date
                    else None
                ),
                "score": hit.score,
                "eslesme": _MATCH_TR.get(hit.match or ""),
            }
            for hit in hits
        ]
        return ToolResult(
            ok=True,
            summary=(
                f"{len(hits)} aday seri bulundu, benzerlige gore sirali. Soruya "
                "uyani ada, ture ve kapsama bakarak sec; en ustteki her zaman dogru "
                "seri degildir. Hicbiri soruya uymuyorsa havuzda bu veri yoktur."
            ),
            preview=preview,
            warnings=notes,
        )

    return ToolSpec(
        name="search_catalog",
        description_tr=(
            "Veri havuzundaki serileri dogal dille arar; aday seri kimliklerini, "
            "birimlerini, turlerini ve havuzdaki kapsamlarini dondurur. Yalnizca "
            "havuzda olan serileri bulur."
        ),
        args_model=SearchCatalogArgs,
        fn=run,
        brief_name="Lakehouse",
        returns_candidates=True,
    )


# --------------------------------------------------------------------------
# get_series
# --------------------------------------------------------------------------


class GetSeriesArgs(BaseModel):
    series_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "search_catalog'un dondurdugu series_id listesi, ornek "
            "['evds:TP.KTF12', 'bddk:aylik:tuketici:konut:TL:sektor']. SQL veya "
            "seri adi degil."
        ),
        examples=[["evds:TP.KTF12"]],
    )
    start: str = Field(
        ...,
        pattern=YEAR_MONTH,
        description="Baslangic ayi YYYY-AA.",
        examples=["2021-01"],
    )
    end: str = Field(
        ...,
        pattern=YEAR_MONTH,
        description="Bitis ayi YYYY-AA, dahil.",
        examples=["2025-12"],
    )
    into: Literal["new", "active"] = Field(
        "new",
        description=(
            "'new': yeni bir tablo kurar ve onu aktif tablo yapar. 'active': "
            "serileri aktif tabloya yeni sutun olarak ekler, mevcut sutunlara "
            "dokunmaz ('tabloyu bozmadan' istekleri)."
        ),
    )
    freq: Literal["D", "W", "M", "Q", "A"] = Field(
        "M",
        description=(
            "Tablonun frekansi: D gunluk, W haftalik, M aylik (varsayilan), Q uc "
            "aylik, A yillik. Kullanici gunluk ya da haftalik istemediyse M birak. "
            "Seri kendi frekansindan daha sik bir tabloya konamaz."
        ),
    )


def make_get_series() -> ToolSpec:
    """Fetch pool series as one table at the asked frequency (monthly by
    default), new or added to the active one."""

    def run(args: GetSeriesArgs, ctx: ToolContext) -> ToolResult:
        lake = ctx.lake
        if lake is None:
            return _no_lake()
        if args.start > args.end:
            return ToolResult(
                ok=False,
                summary=f"Baslangic ({args.start}) bitisten ({args.end}) sonra olamaz.",
            )
        ids = list(dict.fromkeys(args.series_ids))
        try:
            catalog = lake.catalog_lookup(ids)
        except SeriesNotFound as exc:
            hint = (
                "" if "search_catalog" in str(exc) else " Once search_catalog ile ara."
            )
            return ToolResult(ok=False, summary=f"{exc}{hint}")
        info = {row["series_id"]: row for _, row in catalog.iterrows()}
        target = args.freq

        coarse = {
            sid: native
            for sid in ids
            if _FREQ_RANK.get(native := _clean(info[sid].get("freq")), 0)
            > _FREQ_RANK[target]
        }
        if coarse:
            listing = ", ".join(
                f"{sid} ({_FREQ_TR.get(native, native)})"
                for sid, native in coarse.items()
            )
            return ToolResult(
                ok=False,
                summary=(
                    f"{listing} {_FREQ_TR[target]} tablodan daha seyrek yayimlaniyor. "
                    f"{_FREQ_TR[target].capitalize()} tabloya koymak her donemi ayri "
                    "bir deger gibi gostermek, yani veri uydurmak olur. freq ile "
                    "serinin kendi frekansini ya da daha seyrek bir frekansi iste."
                ),
            )

        if args.into == "active" and ctx.has_table():
            table_freq = _table_freq(ctx)
            if table_freq and table_freq != target:
                return ToolResult(
                    ok=False,
                    summary=(
                        f"Aktif tablo {_FREQ_TR.get(table_freq, table_freq)}; yeni "
                        f"sutunlar ayni frekansta olmali. freq='{table_freq}' ile "
                        "iste ya da into='new' ile ayri tablo kur."
                    ),
                )

        names = {sid: column_name(sid) for sid in ids}
        frame = lake.wide(ids, args.start, args.end, target).rename(columns=names)
        frame["period"] = pd.to_datetime(frame["period"])
        frame = frame.sort_values("period").reset_index(drop=True)
        new_columns = [names[sid] for sid in ids]
        if frame.empty or frame[new_columns].isna().all().all():
            return ToolResult(
                ok=False,
                summary=(
                    f"{args.start} - {args.end} araliginda bu seriler icin havuzda "
                    "veri yok."
                ),
            )

        long = lake.get(ids, args.start, args.end)
        warnings = _conversion_warnings(ids, names, info, target)
        warnings += _partial_period_warnings(long, ids, names, info, target)
        columns = [
            _series_column(sid, names[sid], info[sid], ctx, target) for sid in ids
        ]

        if args.into == "active" and ctx.has_table():
            existing = ctx.table()
            clash = [name for name in new_columns if name in existing.columns]
            if clash:
                return ToolResult(
                    ok=False,
                    summary=(
                        f"{', '.join(clash)} sutunu tabloda zaten var. "
                        "into='new' ile ayri bir tablo olustur."
                    ),
                )
            periods = pd.to_datetime(existing["period"])
            outside = int((~frame["period"].isin(periods)).sum())
            if outside:
                warnings.append(
                    f"Yeni serilerin {outside} {_PERIOD_WORD[target]}i aktif "
                    "tablonun donemleri disinda kaldigi icin eklenmedi."
                )
            added = frame.set_index("period")[new_columns]
            merged = existing.copy()
            for name in new_columns:
                merged[name] = periods.map(added[name]).to_numpy()
            # Gaps are counted in the table the columns landed in: a series
            # fetched for 2024-2025 onto a 2021-2025 table is empty for three
            # years there, although the fetch itself had no gap.
            warnings += _coverage_warnings(
                merged.assign(period=periods), new_columns, window=None
            )
            ref = ctx.active_table or ctx.new_table_ref("get_series")
            ctx.put_table(ref, merged)
            shown = merged.assign(period=periods)
            where = "Aktif tabloya yeni sutun olarak eklendi"
        else:
            warnings += _coverage_warnings(
                frame, new_columns, window=(args.start, args.end)
            )
            ref = ctx.new_table_ref("get_series")
            ctx.put_table(ref, frame)
            ctx.set_active(ref)
            shown = frame
            where = "Yeni tablo kuruldu ve aktif tablo yapildi"

        warnings += _money_scale_warning(shown, ref, ctx, columns)
        first = _label(shown["period"].min(), target)
        last = _label(shown["period"].max(), target)
        return ToolResult(
            ok=True,
            summary=(
                f"{len(ids)} seri, {len(shown)} {_PERIOD_WORD[target]}, {first} - "
                f"{last}, {_FREQ_TR[target]}. {where}: {', '.join(new_columns)}."
            ),
            table_ref=ref,
            preview=_preview(shown[["period", *new_columns]]),
            columns=columns,
            sources=_sources(long, ids, info, target),
            warnings=warnings,
        )

    return ToolSpec(
        name="get_series",
        description_tr=(
            "Havuzdaki serileri verilen aralikta tek tabloya getirir; varsayilan "
            "AYLIK, freq ile gunluk/haftalik/uc aylik/yillik. Daha sik yayimlanan "
            "seriler kaynagin onerdigi yontemle (son deger, ortalama ya da toplam) "
            "cevrilir ve bu uyarida yazar. into='active' ile aktif tabloya yeni "
            "sutun ekler."
        ),
        args_model=GetSeriesArgs,
        fn=run,
        brief_name="Lakehouse",
    )


def _conversion_warnings(
    ids: list[str], names: dict[str, str], info: dict[str, pd.Series], target: str
) -> list[str]:
    warnings = []
    for sid in ids:
        freq = _clean(info[sid].get("freq"))
        rule = _conversion_rule(
            freq,
            target,
            _clean(info[sid].get("agg_method")),
            _clean(info[sid].get("kind")),
        )
        if rule:
            warnings.append(
                f"{names[sid]} ({sid}) {_FREQ_TR.get(freq, freq)} yayimlaniyor; "
                f"{_FREQ_TR[target]} degere {rule} alinarak cevrildi."
            )
    return warnings


def _table_freq(ctx: ToolContext) -> str | None:
    """The frequency of the active table, from its columns' lineage."""
    ref = ctx.active_table or ""
    freqs = {
        entry.freq
        for name in ctx.table().columns
        if (entry := ctx.lineage.get(f"{ref}.{name}")) is not None and entry.freq
    }
    return freqs.pop() if len(freqs) == 1 else None


_BUCKET_ALIAS = {"M": "M", "Q": "Q", "A": "Y"}


def _partial_period_warnings(
    long: pd.DataFrame,
    ids: list[str],
    names: dict[str, str],
    info: dict[str, pd.Series],
    target: str,
) -> list[str]:
    """Periods a weekly series fills with fewer weeks than the period has.

    A week belongs to the period its observation day is in, so a month has
    as many weeks as it has of that weekday -- four or five Fridays. Fewer
    observed weeks means the month's average or total is built from part of
    it, and the table cannot show that on its own.
    """
    warnings: list[str] = []
    if long.empty or target not in _BUCKET_ALIAS:
        return warnings
    for sid in ids:
        if _clean(info[sid].get("freq")) != "W":
            continue
        rows = long[(long["series_id"] == sid) & long["value"].notna()]
        if rows.empty:
            continue
        observed = pd.to_datetime(rows["obs_date"])
        weekday = int(observed.dt.weekday.mode().iloc[0])
        periods = observed.dt.to_period(_BUCKET_ALIAS[target]).value_counts()
        short = []
        for period, count in sorted(periods.items()):
            days = pd.date_range(
                period.start_time, period.end_time.normalize(), freq="D"
            )
            expected = int((days.weekday == weekday).sum())
            if count < expected:
                short.append(
                    f"{_label(period.start_time, target)} ({count}/{expected} hafta)"
                )
        if short:
            warnings.append(
                f"{names[sid]}: su donemler eksik haftayla hesaplandi, deger "
                f"donemin tamamini yansitmiyor: {', '.join(short)}."
            )
    return warnings


def _coverage_warnings(
    frame: pd.DataFrame, columns: list[str], *, window: tuple[str, str] | None
) -> list[str]:
    """Months with no value, and a requested `window` (start, end) the pool
    only partly covers. No window for columns added to an existing table,
    whose months are the table's, not the request's.

    Where the first value sits is only news about the pool when the whole
    window was asked for. On `into='active'` the column was fetched for its
    own range and laid onto a wider table, so the empty months in front of it
    are the request's doing: measured, a series the pool holds from 2021 was
    reported as "seri havuzda 2024-01 itibariyla basliyor" because the model
    had asked for 2024 onwards. That sentence reaches the answer, and
    planner rule 5 forbids telling anyone the pool lacks data it has.
    """
    warnings = []
    for name in columns:
        missing = int(frame[name].isna().sum())
        if not missing:
            continue
        filled = frame.loc[frame[name].notna(), "period"]
        note = f"{name}: {len(frame)} ayin {missing}'inda deger yok (doldurulmadi)"
        if (
            window is not None
            and not filled.empty
            and filled.min() > frame["period"].min()
        ):
            note += f"; seri havuzda {_month(filled.min())} itibariyla basliyor"
        elif window is None and not filled.empty:
            note += (
                f"; bu sutun yalnizca {_month(filled.min())} - "
                f"{_month(filled.max())} icin getirildi"
            )
        warnings.append(note + ".")
    if window is None:
        return warnings
    start, end = window
    last = _month(frame["period"].max())
    if last < end:
        warnings.append(f"Istenen bitis {end}; havuzdaki veri {last} ayinda bitiyor.")
    first = _month(frame["period"].min())
    if first > start:
        warnings.append(f"Istenen baslangic {start}; tablodaki ilk ay {first}.")
    return warnings


def _money_scale_warning(
    table: pd.DataFrame, ref: str, ctx: ToolContext, added: list[ColumnInfo]
) -> list[str]:
    """Warn when one table holds money in different scales, e.g. bin TL next
    to milyon TL (plan 5.3). Units are never converted (CLAUDE.md #3), so a
    comparison across such columns is off by a factor of a thousand unless
    whoever reads it knows."""
    units = {column.name: column.unit for column in added}
    for name in table.columns:
        if name not in units:
            entry = ctx.lineage.get(f"{ref}.{name}")
            units[name] = entry.unit if entry is not None else None
    scales: dict[str, list[str]] = {}
    for name, unit in units.items():
        match = re.match(r"\s*(bin|milyon|milyar)\s+TL", unit or "", re.I)
        if match:
            scales.setdefault(match.group(1).lower(), []).append(name)
    if len(scales) < 2:
        return []
    listing = "; ".join(
        f"{scale} TL: {', '.join(names)}" for scale, names in scales.items()
    )
    return [
        f"Tabloda farkli olcekte parasal birimler var ({listing}). Birim cevrilmedi; "
        "karsilastirirken olcek farkini belirt."
    ]


def _series_column(
    series_id: str, name: str, row: pd.Series, ctx: ToolContext, target: str = "M"
) -> ColumnInfo:
    kind = _clean(row.get("kind"))
    rule = _conversion_rule(
        _clean(row.get("freq")), target, _clean(row.get("agg_method")), kind
    )
    return ColumnInfo(
        name=name,
        label_tr=_clean(row.get("name_tr")),
        unit=_clean(row.get("unit")),
        kind=kind,
        freq=target,
        source_series=[series_id],
        op="to_freq" if rule else None,
        formula=f"{_FREQ_TR[target]} = {rule}" if rule else None,
        tool="get_series",
        step_no=ctx.step_no,
    )


def _sources(
    long: pd.DataFrame, ids: list[str], info: dict[str, pd.Series], target: str = "M"
) -> list[Source]:
    """One source per series, stamped with when its data was actually fetched.

    A series whose fetch time cannot be read gets no source line rather than
    an invented one; the lineage still names it.
    """
    stamps = _fetched_at(long)
    sources = []
    for sid in ids:
        if sid not in stamps:
            logger.warning("get_series: %s icin fetched_at okunamadi", sid)
            continue
        row = info[sid]
        freq = _clean(row.get("freq"))
        kind = _clean(row.get("kind"))
        agg = _clean(row.get("agg_method"))
        sources.append(
            Source(
                series_id=sid,
                name=_clean(row.get("name_tr")),
                fetched_at=stamps[sid].to_pydatetime(),
                metadata_url=_clean(row.get("metadata_url")),
                note=(
                    f"{_FREQ_TR.get(freq, freq)} -> {_FREQ_TR[target]}: {rule}"
                    if (rule := _conversion_rule(freq, target, agg, kind))
                    else None
                ),
            )
        )
    return sources


# --------------------------------------------------------------------------
# transform_table
# --------------------------------------------------------------------------


class Condition(BaseModel):
    column: str = Field(
        ...,
        description="Kosulun uygulanacagi sutun; tablodan birebir kopyala.",
        examples=["TP_KTF12_diff"],
    )
    op: Literal["<", "<=", ">", ">=", "==", "!="] = Field(
        ..., description="Karsilastirma."
    )
    value: float = Field(..., description="Esik degeri.", examples=[0])


Op = Literal[
    "diff",
    "pct_change",
    "yoy",
    "change_since",
    "rebase",
    "ratio",
    "deflate",
    "ytd",
    "mark",
]


class TransformArgs(BaseModel):
    op: Op = Field(
        ...,
        description=(
            "diff: bir onceki aya gore fark. pct_change: bir onceki aya gore % "
            "degisim. yoy: yillik % degisim. change_since: base_period'a gore % "
            "degisim. rebase: base_period = 100 olacak sekilde endeks. ratio: "
            "column / denominator x 100 (yuzde pay). deflate: parasal tutari "
            "TUFE ile enflasyondan arindirir (yalniz bakiye/akim tutarlari; oran "
            "ve endekslere uygulanmaz). ytd: aylik akimin yil basindan bu yana "
            "toplami (ornegin BDDK kar-zarar kalemleri; havuzda aylik akim olarak "
            "durur). mark: conditions'in hepsinin saglandigi donemleri isaretler "
            "ve listeler."
        ),
    )
    column: str | None = Field(
        None,
        description=(
            "Islenecek sutun; mark disindaki her islemde zorunlu. Onceki arac "
            "ciktisindan birebir kopyala."
        ),
        examples=["TP_KTF12"],
    )
    base_period: str | None = Field(
        None,
        pattern=YEAR_MONTH,
        description=(
            "rebase ve change_since icin zorunlu baz ay, ornek '2021-01'. deflate "
            "icin istege bagli: verilirse tutarlar o ayin fiyatlarina cekilir, "
            "verilmezse 2025 yili ortalama fiyatlari (TUFE 2025=100) kullanilir."
        ),
    )
    denominator: str | None = Field(
        None,
        description="ratio icin payda sutunu; tablodan birebir kopyala.",
    )
    conditions: list[Condition] | None = Field(
        None,
        description=(
            "mark icin kosullar, hepsi ayni ayda saglanmali. Ornek: faizin dustugu "
            "ve kredinin artmadigi aylar icin [{'column': 'TP_KTF12_diff', 'op': "
            "'<', 'value': 0}, {'column': '..._diff', 'op': '<=', 'value': 0}]."
        ),
    )
    name: str | None = Field(
        None,
        pattern=r"^[a-z][a-z0-9_]{0,39}$",
        description=(
            "mark icin yeni sutunun adi (istege bagli), ornek "
            "'faiz_dustu_kredi_artmadi'. Verilmezse 'isaret' kullanilir."
        ),
    )
    table_ref: TableRef | None = Field(
        None, description="Hedef tablo; bos birakilirsa aktif tablo kullanilir."
    )

    @model_validator(mode="after")
    def _arguments_for_op(self) -> TransformArgs:
        """Say which argument an operation is missing, or does not take,
        before anything runs.

        Raised as a validation error, so the loop feeds it back to the model
        once with the field named, instead of the tool failing halfway. An
        argument the operation ignores is refused too: a stray denominator on
        a diff went into the column's lineage as an input it never had, and a
        base month on a pct_change reads to the model as a "change since"
        that was never computed.
        """
        if self.op != "mark" and not self.column:
            raise ValueError(f"'{self.op}' icin column zorunlu")
        if self.op in ("rebase", "change_since") and not self.base_period:
            raise ValueError(f"'{self.op}' icin base_period zorunlu (YYYY-AA)")
        if self.op == "ratio" and not self.denominator:
            raise ValueError("'ratio' icin denominator zorunlu")
        if self.op == "mark" and not self.conditions:
            raise ValueError("'mark' icin en az bir kosul (conditions) zorunlu")
        unused = [
            name
            for name, given, allowed in (
                ("column", self.column, self.op != "mark"),
                ("base_period", self.base_period, self.op in _TAKES_BASE),
                ("denominator", self.denominator, self.op == "ratio"),
                ("conditions", self.conditions, self.op == "mark"),
                ("name", self.name, self.op == "mark"),
            )
            if given and not allowed
        ]
        if unused:
            raise ValueError(
                f"'{self.op}' su argumanlari almaz: {', '.join(unused)}; bunlari "
                "cikar. (mark sutunlari conditions icinde verir.)"
            )
        return self


#: The operations a base month changes.
_TAKES_BASE = ("rebase", "change_since", "deflate")


def make_transform_table() -> ToolSpec:
    """Adds one derived column. Never touches the ones already there."""

    def run(args: TransformArgs, ctx: ToolContext) -> ToolResult:
        try:
            frame = ctx.table(args.table_ref)
        except TableNotFound as exc:
            return ToolResult(ok=False, summary=str(exc).strip("'\""))
        ref = args.table_ref or ctx.active_table or ""
        # Computed on a copy with a datetime period; `frame` itself only ever
        # gains the new column, so no existing column can change underneath
        # the loop's "existing columns came back unchanged" check.
        work = frame.assign(period=pd.to_datetime(frame["period"]))

        if args.op == "mark":
            inputs = [condition.column for condition in args.conditions or []]
        elif args.op == "ratio":
            inputs = [args.column, args.denominator]
        else:
            inputs = [args.column]
        inputs = list(dict.fromkeys(inputs))
        missing = [name for name in inputs if name not in frame.columns]
        if missing:
            return ToolResult(
                ok=False,
                summary=(
                    f"{', '.join(repr(name) for name in missing)} sutunu tabloda yok. "
                    f"Mevcut sutunlar: {', '.join(map(str, frame.columns))}."
                ),
            )
        arithmetic = args.op != "mark"
        not_numbers = [
            name
            for name in inputs
            if not _is_numeric(frame, name, arithmetic=arithmetic)
        ]
        if not_numbers:
            usable = [
                str(c)
                for c in frame.columns
                if _is_numeric(frame, c, arithmetic=arithmetic)
            ]
            what = (
                "sayisal bir deger sutunu degil" if arithmetic else "karsilastirilamaz"
            )
            return ToolResult(
                ok=False,
                summary=(
                    f"{', '.join(repr(name) for name in not_numbers)} {what} "
                    "('period' donem sutunudur; isaret sutunlari yalnizca mark "
                    "kosullarinda kullanilir). Kullanilabilir sutunlar: "
                    f"{', '.join(usable) or 'yok'}."
                ),
            )

        lineage = {name: ctx.lineage.get(f"{ref}.{name}") for name in inputs}
        try:
            outcome = _OPS[args.op](args, work, lineage, ctx)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            # Not just ValueError: a column of numeric-looking text reaches
            # pandas as a string dtype and raises TypeError on subtraction,
            # and a zero divisor raises ZeroDivisionError. Both used to leave
            # the tool as an opaque crash the model could not act on.
            return ToolResult(ok=False, summary=f"{args.op} uygulanamadi: {exc}")
        if isinstance(outcome, ToolResult):
            return outcome

        existing = ctx.lineage.get(f"{ref}.{outcome.name}")
        renamed = ""
        if outcome.name in frame.columns:
            if existing is not None and (
                existing.op,
                existing.inputs,
                existing.formula,
            ) == (args.op, inputs, outcome.formula):
                return ToolResult(
                    ok=False,
                    summary=(
                        f"'{outcome.name}' sutunu ayni islemle zaten hesaplanmis; "
                        "mevcut sutunu kullan."
                    ),
                )
            # Same name, different operation: a ratio over another
            # denominator, a mark with other conditions. Refusing it as "the
            # same operation" pointed the model at a column that answers a
            # different question.
            taken = outcome.name
            outcome.name = _free_name(taken, frame.columns)
            renamed = f" ('{taken}' adi baska bir hesapla dolu oldugundan.)"
        # A Series on the table's own index, so a flag column stays a nullable
        # boolean instead of collapsing into `object`.
        frame[outcome.name] = work["period"].map(outcome.values)
        ctx.put_table(ref, frame)

        upstream = [
            series
            for entry in lineage.values()
            if entry is not None
            for series in entry.source_series
        ]
        column = ColumnInfo(
            name=outcome.name,
            unit=outcome.unit,
            kind=outcome.kind,
            freq=next((e.freq for e in lineage.values() if e and e.freq), None),
            source_series=list(dict.fromkeys([*upstream, *outcome.extra_series])),
            op=args.op,
            formula=outcome.formula,
            inputs=inputs,
            tool="transform_table",
            step_no=ctx.step_no,
        )
        shown = frame[["period", *inputs, outcome.name]]
        return ToolResult(
            ok=True,
            summary=(
                f"'{outcome.name}' sutunu eklendi"
                + (f" (birim: {outcome.unit})" if outcome.unit else "")
                + f".{renamed} {outcome.summary}"
            ).strip(),
            table_ref=ref,
            preview=_preview(shown),
            columns=[column],
            sources=outcome.sources,
            warnings=outcome.warnings,
        )

    return ToolSpec(
        name="transform_table",
        description_tr=(
            "Tabloya turetilmis YENI bir sutun ekler; mevcut sutunlara dokunmaz. "
            "Fark, yuzde degisim, yillik degisim, baz aya gore degisim, endeks, "
            "oran, enflasyondan arindirma ve kosula gore donem isaretleme yapar. "
            "Kullanicinin istemedigi bir donusumu yapma."
        ),
        args_model=TransformArgs,
        fn=run,
        brief_name="Lakehouse",
    )


@dataclass
class _Outcome:
    """What one operation produced, before it is written into the table."""

    name: str
    values: pd.Series
    formula: str
    unit: str | None = None
    kind: Kind | None = None
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    extra_series: list[str] = field(default_factory=list)


OpFn = Callable[
    [TransformArgs, pd.DataFrame, dict[str, ColumnInfo | None], ToolContext],
    "_Outcome | ToolResult",
]


def _is_numeric(frame: pd.DataFrame, column: str, *, arithmetic: bool) -> bool:
    """A column an operation can compute on, never the period column.

    Checked up front, because `period` passes the "column exists" check and
    then broke `_single` with a bare KeyError.

    `arithmetic` separates the two kinds of operation. `mark` compares a
    column with a threshold and is happy with a flag column; every other
    operation divides or subtracts, and a flag column there is meaningless
    at best and fatal at worst -- measured, `pct_change` on a `mark` column
    raises `ZeroDivisionError` (its first period is False, and the change
    from False is a division by zero), while `diff` quietly adds a column of
    XOR-ish numbers with no unit. Neither is something to hand a model.
    """
    if column == "period":
        return False
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return not arithmetic
    if pd.api.types.is_numeric_dtype(values):
        return True
    present = values.dropna()
    return bool(pd.to_numeric(present, errors="coerce").notna().all())


def _free_name(name: str, taken: Any) -> str:
    """`name` with the first numeric suffix not already a column."""
    used = set(map(str, taken))
    number = 2
    while f"{name}_{number}" in used:
        number += 1
    return f"{name}_{number}"


def _single(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[["period", column]].rename(columns={column: "value"})


def _by_period(result: pd.DataFrame) -> pd.Series:
    """A transform's output keyed by period, ready to map onto the table."""
    return result.assign(period=pd.to_datetime(result["period"])).set_index("period")[
        "value"
    ]


def _base_stamp(base_period: str) -> pd.Timestamp:
    return pd.Timestamp(f"{base_period}-01")


def _unit_of(entry: ColumnInfo | None) -> str | None:
    return entry.unit if entry is not None else None


def _op_diff(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    result, formula = transforms.diff(_single(frame, args.column))
    entry = lineage[args.column]
    unit = _unit_of(entry)
    summary = "Ilk ay icin onceki ay olmadigindan fark bos."
    # Keyed on the kind, not the unit text: the pool's rate series says
    # "Ağırlıklı ortalama", not "%", so a unit check never fired on real
    # data -- the very setup of the 22 Sep "%27,85 puan" answer (-0.2785).
    if unit == "%" or (entry is not None and entry.kind == "rate"):
        unit = "yuzde puan"
        summary += " Degerler yuzde puandir: -0.25, 0,25 puanlik dusus demektir."
    return _Outcome(
        name=f"{args.column}_diff",
        values=_by_period(result),
        formula=formula,
        unit=unit,
        summary=summary,
    )


def _op_pct_change(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    result, formula = transforms.pct_change(_single(frame, args.column))
    return _Outcome(
        name=f"{args.column}_pct",
        values=_by_period(result),
        formula=formula,
        unit="%",
        kind="rate",
    )


def _op_yoy(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    entry = lineage[args.column]
    freq = entry.freq if entry is not None else None
    lag = _PERIODS_PER_YEAR.get(freq)
    if lag is None:
        raise ValueError(
            f"'{args.column}' sutununun frekansi bilinmiyor; yillik gecikme "
            "belirlenemiyor. pct_change ya da change_since kullan."
        )
    result, formula = transforms.yoy(_single(frame, args.column), lag)
    return _Outcome(
        name=f"{args.column}_yoy",
        values=_by_period(result),
        formula=formula,
        unit="%",
        kind="rate",
        summary=f"Ilk {lag} donem icin bir yil oncesi olmadigindan deger bos.",
    )


def _op_change_since(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    result, formula = transforms.change_since(
        _single(frame, args.column), _base_stamp(args.base_period)
    )
    return _Outcome(
        name=f"{args.column}_degisim_{args.base_period.replace('-', '')}",
        values=_by_period(result),
        formula=formula,
        unit="%",
        kind="rate",
        summary=f"Her ay {args.base_period} ayina gore yuzde degisim.",
    )


def _op_rebase(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    result, formula = transforms.rebase(
        _single(frame, args.column), _base_stamp(args.base_period)
    )
    return _Outcome(
        name=f"{args.column}_endeks_{args.base_period.replace('-', '')}",
        values=_by_period(result),
        formula=formula,
        unit="endeks",
        kind="index",
        summary=f"{args.base_period} = 100.",
    )


def _op_ratio(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    result, formula = transforms.ratio(
        _single(frame, args.column), _single(frame, args.denominator)
    )
    warnings = []
    top, bottom = _unit_of(lineage[args.column]), _unit_of(lineage[args.denominator])
    if top and bottom and top != bottom:
        warnings.append(
            f"Pay ve payda farkli birimde ({top} / {bottom}); oran birim cevrilmeden "
            "hesaplandi."
        )
    return _Outcome(
        name=f"{args.column}_pay",
        values=_by_period(result),
        formula=formula,
        unit="%",
        kind="rate",
        summary=f"{args.column} / {args.denominator} x 100.",
        warnings=warnings,
    )


_NOT_DEFLATABLE = {
    "rate": (
        "bir oran serisi (%). Enflasyondan arindirma parasal tutarlara (bakiye ya "
        "da akim) uygulanir; bir orani TUFE'ye bolmek anlamsiz bir sayi uretir. "
        "Reel faiz bu aracta yok."
    ),
    "index": (
        "bir endeks. Endeksler enflasyondan arindirilmaz; TUFE'ye gore goreli "
        "degisim isteniyorsa ratio kullan."
    ),
}

#: A money unit: TL at any scale. Counts ("Adet") are stocks and flows too.
_MONEY_UNIT = re.compile(r"\bTL\b|\blira\b", re.IGNORECASE)


def deflate_refusal(column: str, entry: ColumnInfo | None) -> str | None:
    """Why `column` cannot be deflated, or None if it can.

    Only a monthly money amount that is still nominal: a stock or flow in
    TL. A rate or an index divided by CPI is a number with no meaning (the
    22 Sep live run did it to the interest rate); so is a count of houses
    "at 2025 prices", and a column already deflated would be deflated twice.
    Shared with the fake tools, so `--live` and the tests refuse what the
    real tool refuses.
    """
    kind = entry.kind if entry is not None else None
    if kind not in ("stock", "flow"):
        reason = _NOT_DEFLATABLE.get(
            kind,
            "turu bilinmiyor (turetilmis bir sutun olabilir). Enflasyondan "
            "arindirma yalnizca get_series ile gelen parasal tutarlara uygulanir.",
        )
        return f"'{column}' {reason}"
    if entry.op == "deflate":
        return (
            f"'{column}' zaten enflasyondan arindirilmis (reel) bir sutun; ikinci "
            "kez arindirilmaz. Baska bir baz ay isteniyorsa nominal sutunu kullan."
        )
    if not _MONEY_UNIT.search(entry.unit or ""):
        return (
            f"'{column}' parasal bir tutar degil (birim: {entry.unit or 'bilinmiyor'})."
            " Enflasyondan arindirma yalnizca TL cinsinden tutarlara uygulanir."
        )
    if entry.freq != "M":
        return (
            f"'{column}' aylik degil; TUFE aylik oldugundan once aylik tablo gerekir."
        )
    return None


def _op_deflate(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome | ToolResult:
    """Real values of a money amount, at 2025 average prices unless a base
    month is given. What may be deflated is `deflate_refusal`'s rule."""
    entry = lineage[args.column]
    refusal = deflate_refusal(args.column, entry)
    if refusal:
        return ToolResult(ok=False, summary=refusal)

    lake = ctx.lake
    if lake is None:
        return _no_lake()
    start, end = _month(frame["period"].min()), _month(frame["period"].max())
    cpi = _cpi(lake, start, end)

    if args.base_period:
        # Read on its own: the base month need not lie inside the table's
        # window ("at December 2025 prices" for a 2021-2023 table), and
        # looking for it only there called a month the pool holds missing.
        base_rows = _cpi(lake, args.base_period, args.base_period)["value"]
        if base_rows.empty or pd.isna(base_rows.iloc[0]):
            return ToolResult(
                ok=False,
                summary=f"TUFE {args.base_period} ayi icin havuzda yok; baska bir baz ay sec.",
            )
        base, prices = float(base_rows.iloc[0]), f"{args.base_period} fiyatlariyla"
    else:
        base, prices = 100.0, "2025 yili ortalama fiyatlariyla (TUFE 2025=100)"

    nominal = _single(frame, args.column)
    result, _ = transforms.deflate(nominal, cpi[["period", "value"]], base=base)
    if result["value"].isna().all() and nominal["value"].notna().any():
        # Every month lost its value to a missing CPI: the table lies outside
        # what the pool holds the index for. Adding an entirely empty column
        # and calling it a success invites the model to describe a real
        # series that does not exist; saying which months are covered lets it
        # narrow the table instead.
        return ToolResult(
            ok=False,
            summary=(
                f"{start} - {end} araliginda TUFE havuzda olmadigi icin "
                f"'{args.column}' enflasyondan arindirilamadi. Tabloyu TUFE'nin "
                "kapsadigi aylara daraltip tekrar dene."
            ),
        )
    unmatched = int(result["value"].isna().sum() - nominal["value"].isna().sum())
    warnings = [f"Reel {args.column} {prices}; nominal degil, reel tutar."]
    if unmatched > 0:
        warnings.append(f"{unmatched} ayda TUFE olmadigindan reel deger bos.")

    stamps = _fetched_at(lake.get([CPI_SERIES], start, end))
    sources = []
    if CPI_SERIES in stamps:
        sources.append(
            Source(
                series_id=CPI_SERIES,
                name="TUFE Genel Endeks (2025=100)",
                fetched_at=stamps[CPI_SERIES].to_pydatetime(),
                note="Enflasyondan arindirmada fiyat endeksi olarak kullanildi.",
            )
        )
    unit = entry.unit
    suffix = f"_{args.base_period.replace('-', '')}" if args.base_period else ""
    return _Outcome(
        name=f"reel_{args.column}{suffix}",
        values=_by_period(result),
        formula=(
            f"reel = nominal x TUFE_baz / TUFE_t; TUFE = {CPI_SERIES}, "
            f"baz = {args.base_period or '2025 ortalamasi = 100'}"
        ),
        unit=f"{unit}, {prices}" if unit else prices,
        kind=entry.kind,
        summary=f"Tutarlar {prices}.",
        warnings=warnings,
        sources=sources,
        extra_series=[CPI_SERIES],
    )


def _cpi(lake: Any, start: str, end: str) -> pd.DataFrame:
    """The deflator, monthly, as `period`/`value`."""
    cpi = lake.wide([CPI_SERIES], start, end, "M").rename(columns={CPI_SERIES: "value"})
    cpi["period"] = pd.to_datetime(cpi["period"])
    return cpi


def _op_ytd(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome | ToolResult:
    """The running total since January of a monthly flow. The inverse of
    what silver does to BDDK's year-to-date profit-and-loss lines, for the
    "yilbasindan bu yana" question."""
    entry = lineage[args.column]
    if entry is None or entry.kind != "flow" or entry.freq != "M":
        return ToolResult(
            ok=False,
            summary=(
                f"'{args.column}' aylik bir akim degil; yil basindan toplam yalnizca "
                "aylik akimlara (kar-zarar kalemleri, satislar) uygulanir. Bakiye "
                "zaten donem sonu degeridir."
            ),
        )
    result, formula = transforms.year_to_date(_single(frame, args.column))
    return _Outcome(
        name=f"{args.column}_yilbasi",
        values=_by_period(result),
        formula=formula,
        unit=entry.unit,
        kind="stock",
        summary="Her ay, o yilin Ocak ayindan o aya kadarki toplam.",
    )


def _op_mark(
    args: TransformArgs,
    frame: pd.DataFrame,
    lineage: dict[str, ColumnInfo | None],
    ctx: ToolContext,
) -> _Outcome:
    """Flag the months where every condition holds, and name them.

    Naming them in the summary is the point: "which months?" is answered by
    the tool, not by the model reading two columns and counting (planner
    rule 1).
    """
    series = {c.column: _single(frame, c.column) for c in args.conditions}
    conditions = [(c.column, c.op, c.value) for c in args.conditions]
    result, formula = transforms.mark_periods(series, conditions)
    values = _by_period(result)

    # By value, not `is True`: the cells are numpy booleans, and numpy 2 even
    # names that type "bool", which is how an identity check found no month.
    flagged = [_month(p) for p in values.index[values.eq(True).fillna(False)]]
    unknown = int(values.isna().sum())
    if flagged:
        listed = ", ".join(flagged[:MARKED_LISTED])
        more = len(flagged) - MARKED_LISTED
        summary = f"{len(flagged)} ay kosullarin hepsini sagliyor: {listed}"
        summary += f" ve {more} ay daha." if more > 0 else "."
    else:
        summary = "Kosullarin hepsini saglayan ay yok."
    if unknown:
        summary += f" {unknown} ay eksik veri yuzunden degerlendirilemedi."
    return _Outcome(
        name=args.name or "isaret",
        values=values,
        formula=formula,
        summary=summary,
    )


_OPS: dict[str, OpFn] = {
    "diff": _op_diff,
    "pct_change": _op_pct_change,
    "yoy": _op_yoy,
    "change_since": _op_change_since,
    "rebase": _op_rebase,
    "ratio": _op_ratio,
    "deflate": _op_deflate,
    "ytd": _op_ytd,
    "mark": _op_mark,
}


#: Built once, like `tools.analysis.TOOLS`; `tools.toolset` wires them in.
TOOLS = [make_search_catalog(), make_get_series(), make_transform_table()]
