"""Agent-facing wrappers around analysis/anomaly.py and analysis/change.py
(plan 4.5).

A tool never sees a DataFrame in its arguments and never returns one: the
model passes a `table_ref` + a column name, this module pulls the frame out
of the session (`ToolContext.table`), calls the pure analysis function, and
puts the result table back into the session (`ToolContext.put_table`). Only
`table_ref`, `preview` (first/last rows) and `summary` travel back to the
model -- never the full table (CLAUDE.md #7, contracts.py's ToolResult
docstring).

Per docs/contracts.md section 5 ("Aktif tablo kuralı"): analysis-result
tables never change the session's active table (only get_series(into="new"),
run_sql, ingest, read_url do) -- so these tools deliberately do not call
`ctx.set_active()`.

ToolResult / ToolContext / ToolSpec are defined once, in track B's
contracts.py / agent/context.py / tools/registry.py -- imported here, never
redefined (CLAUDE.md, contracts.py's own "SINGLE DEFINITION RULE").
tools/registry.py itself is track B's file: not edited here, and the actual
tool registration (wiring `TOOLS` below into the live registry) is theirs.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from lakehouse_agent.analysis.anomaly import (
    _MIN_PERIODS_FOR_STL,
    IQR_FALLBACK_NOTE,
    detect_anomalies,
)
from lakehouse_agent.analysis.causality import NotEnoughData, lead_lag
from lakehouse_agent.analysis.change import detect_changes

try:
    from lakehouse_agent.agent.context import ToolContext
    from lakehouse_agent.contracts import ColumnInfo, TableRef, ToolResult
    from lakehouse_agent.tools.registry import ToolSpec
except ImportError as exc:  # pragma: no cover - real absence, not exercised by tests
    raise ImportError(
        "tools/analysis.py, track B'nin contracts.py / agent/context.py / "
        "tools/registry.py dosyalarina ihtiyac duyar. Bu dalda henuz yoklar "
        "(B'nin dali kasitli olarak merge edilmedi -- config.py ve "
        "pyproject.toml degisiklikleri cakisiyor). Gercek registry icinde "
        "kullanmadan once B'nin dalini merge edin. Testlerde bu modulu import "
        "etmeden once ToolResult/ToolContext/ToolSpec/ColumnInfo icin "
        "sys.modules stub'lari kurun (bkz. tests/test_tools_analysis.py)."
    ) from exc

PREVIEW_ROWS = 5

_STL_FORMULAS = {
    "beklenen": "STL trend + mevsimsellik",
    "z": "z = (kalıntı - medyan) / (1.4826 × MAD)",
}
_IQR_FORMULAS = {
    "beklenen": "Medyan (Tukey 1.5×IQR yolunda)",
    "z": "Çit mesafesi: max(0, deger-üst_çit, alt_çit-deger), 1.5×IQR çiti",
}


def _preview_rows(frame: pd.DataFrame, n: int = PREVIEW_ROWS) -> list[dict]:
    """First and last `n` rows -- enough for the model to see the shape of
    the result without ever holding the whole table (contracts.py)."""
    if len(frame) <= 2 * n:
        return frame.to_dict("records")
    return pd.concat([frame.head(n), frame.tail(n)]).to_dict("records")


def _missing_warning(series: pd.DataFrame) -> list[str]:
    missing = int(series["value"].isna().sum())
    if not missing:
        return []
    return [f"{missing} eksik değerli dönem atlandı (doldurulmadı, CLAUDE.md #2)."]


def _series_for_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[["period", column]].rename(columns={column: "value"})


def _lineage_entry(ctx: ToolContext, table_ref: TableRef | None, column: str):
    """The session's recorded lineage for `column`
    (`ctx.lineage["{table_ref}.{column}"]`), or None if it has none (e.g. it
    was hand-built in a test fixture)."""
    ref = table_ref or ctx.active_table or ""
    return ctx.lineage.get(f"{ref}.{column}")


def _upstream_source_series(
    ctx: ToolContext, table_ref: TableRef | None, column: str
) -> list[str]:
    """The catalog series_id(s) behind `column`."""
    upstream = _lineage_entry(ctx, table_ref, column)
    return list(upstream.source_series) if upstream else []


# EVDS/BDDK frequency (contracts.py's Freq literal, D|W|M|Q|A) -> the STL
# seasonal period detect_anomalies needs. Weekly TP.KTF12 needs period=52,
# not the function's monthly-shaped default of 12 -- this tool layer never
# passed a period at all before, so every non-monthly column got the wrong
# seasonal cycle length.
_FREQ_TO_PERIOD = {"D": 7, "W": 52, "M": 12, "Q": 4, "A": 1}


def _upstream_period(
    ctx: ToolContext, table_ref: TableRef | None, column: str
) -> int | None:
    """STL seasonal period implied by `column`'s own frequency (via
    lineage), e.g. weekly -> 52, monthly -> 12. None when lineage/freq is
    unknown -- detect_anomalies then falls back to its own default."""
    upstream = _lineage_entry(ctx, table_ref, column)
    freq = getattr(upstream, "freq", None) if upstream else None
    return _FREQ_TO_PERIOD.get(freq)


class DetectAnomaliesArgs(BaseModel):
    table_ref: TableRef | None = Field(
        None,
        description="Analiz edilecek tablo; boş bırakılırsa aktif tablo kullanılır.",
        examples=["t01_get_series"],
    )
    column: str = Field(
        ...,
        description=(
            "Tabloda anomali aranacak sütun adı. Önceki araç çıktısından "
            "birebir kopyalayın, uydurmayın."
        ),
        examples=["TP_KTF12"],
    )


def make_detect_anomalies() -> ToolSpec:
    """STL+MAD point-anomaly detection (analysis/anomaly.py), exposed as a tool."""

    def run(args: DetectAnomaliesArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table(args.table_ref)
        if args.column not in frame.columns:
            return ToolResult(
                ok=False,
                summary=(
                    f"'{args.column}' sütunu tabloda yok. "
                    f"Mevcut sütunlar: {', '.join(map(str, frame.columns))}."
                ),
            )

        series = _series_for_column(frame, args.column)
        warnings = _missing_warning(series)
        if len(series.dropna(subset=["value"])) < _MIN_PERIODS_FOR_STL:
            warnings.append(
                "Seri 24 dönemden kısa; STL yerine IQR (Tukey) kuralı kullanıldı."
            )

        period = _upstream_period(ctx, args.table_ref, args.column)
        try:
            if period is not None:
                result, summary = detect_anomalies(series, period=period)
            else:
                result, summary = detect_anomalies(series)
        except Exception as exc:
            return ToolResult(
                ok=False, summary=f"detect_anomalies çalıştırılamadı: {exc}"
            )

        ref = ctx.new_table_ref("detect_anomalies")
        ctx.put_table(ref, result)

        source_series = _upstream_source_series(ctx, args.table_ref, args.column)
        used_iqr = (
            len(series.dropna(subset=["value"])) < _MIN_PERIODS_FOR_STL
            or IQR_FALLBACK_NOTE in summary
        )
        formulas = _IQR_FORMULAS if used_iqr else _STL_FORMULAS
        columns = [
            ColumnInfo(
                name=name,
                op="detect_anomalies",
                formula=formulas.get(name),
                inputs=[args.column],
                source_series=source_series,
                tool="detect_anomalies",
                step_no=ctx.step_no,
            )
            for name in result.columns
        ]

        return ToolResult(
            ok=True,
            summary=summary,
            table_ref=ref,
            preview=_preview_rows(result),
            columns=columns,
            warnings=warnings,
        )

    return ToolSpec(
        name="detect_anomalies",
        description_tr=(
            "Bir tablodaki sütunda nokta anomalilerini tespit eder "
            "(STL + MAD; 24 dönemden kısa seride IQR)."
        ),
        args_model=DetectAnomaliesArgs,
        fn=run,
        brief_name="Analysis",
    )


class DetectChangesArgs(BaseModel):
    table_ref: TableRef | None = Field(
        None,
        description="Analiz edilecek tablo; boş bırakılırsa aktif tablo kullanılır.",
        examples=["t01_get_series"],
    )
    column: str = Field(
        ...,
        description=(
            "Tabloda kırılma (yapısal değişim) aranacak sütun adı. Önceki "
            "araç çıktısından birebir kopyalayın, uydurmayın."
        ),
        examples=["TP_KTF12"],
    )


def make_detect_changes() -> ToolSpec:
    """PELT structural-break detection (analysis/change.py), exposed as a tool."""

    def run(args: DetectChangesArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table(args.table_ref)
        if args.column not in frame.columns:
            return ToolResult(
                ok=False,
                summary=(
                    f"'{args.column}' sütunu tabloda yok. "
                    f"Mevcut sütunlar: {', '.join(map(str, frame.columns))}."
                ),
            )

        series = _series_for_column(frame, args.column)
        warnings = _missing_warning(series)

        try:
            result, summary = detect_changes(series)
        except Exception as exc:
            return ToolResult(
                ok=False, summary=f"detect_changes çalıştırılamadı: {exc}"
            )

        ref = ctx.new_table_ref("detect_changes")
        ctx.put_table(ref, result)

        source_series = _upstream_source_series(ctx, args.table_ref, args.column)
        formulas = {
            "ortalama_öncesi": "kırılma öncesi bölümün ortalaması",
            "ortalama_sonrası": "kırılma sonrası bölümün ortalaması",
            "eğim_öncesi": "kırılma öncesi bölümün doğrusal regresyon eğimi",
            "eğim_sonrası": "kırılma sonrası bölümün doğrusal regresyon eğimi",
            "seviye_degisim_yuzde": (
                "(ortalama_sonrası - ortalama_öncesi) / |ortalama_öncesi| × 100"
            ),
            "sicrama_yuzde": (
                "kırılma anındaki değer ile bir önceki dönemin değeri arasındaki "
                "yüzde değişim (anlık sıçrama, ortalama farkı değil)"
            ),
        }
        columns = [
            ColumnInfo(
                name=name,
                op="detect_changes",
                formula=formulas.get(name),
                inputs=[args.column],
                source_series=source_series,
                tool="detect_changes",
                step_no=ctx.step_no,
            )
            for name in result.columns
        ]

        return ToolResult(
            ok=True,
            summary=summary,
            table_ref=ref,
            preview=_preview_rows(result),
            columns=columns,
            warnings=warnings,
        )

    return ToolSpec(
        name="detect_changes",
        description_tr=(
            "Bir tablodaki sütunda yapısal kırılma (seviye/eğim değişimi) "
            "tespit eder (PELT)."
        ),
        args_model=DetectChangesArgs,
        fn=run,
        brief_name="Analysis",
    )


# --- causality --------------------------------------------------------------

#: Lags a pair is tested over by default: half a year of monthly data, which
#: is where a credit or price reaction is looked for. Twelve is the most the
#: tool accepts -- every extra lag costs observations and widens the
#: correction that keeps the result honest.
DEFAULT_MAX_LAG = 6
MAX_MAX_LAG = 12

#: Quantities whose change is read as a growth rate, so differencing happens
#: in logs. A rate or an index is differenced as it is: a percentage point is
#: already a change.
_LOG_KINDS = frozenset({"stock", "flow"})


class TestCausalityArgs(BaseModel):
    table_ref: TableRef | None = Field(
        None,
        description="Analiz edilecek tablo; boş bırakılırsa aktif tablo kullanılır.",
        examples=["t01_get_series"],
    )
    column: str = Field(
        ...,
        description=(
            "Öncül olup olmadığı sınanan sütun (X). Önceki araç çıktısından "
            "birebir kopyalayın."
        ),
        examples=["TP_KFE_TR"],
    )
    against: str = Field(
        ...,
        description=(
            "Diğer sütun (Y). Araç iki yönü de sınar: X, Y'nin önünde mi, "
            "yoksa Y, X'in önünde mi."
        ),
        examples=["aylik_tuketici_konut_TL_sektor"],
    )
    max_lag: int = Field(
        DEFAULT_MAX_LAG,
        ge=1,
        le=MAX_MAX_LAG,
        description=(
            f"Kaç döneme kadar gecikme sınanacak (varsayılan {DEFAULT_MAX_LAG}). "
            "Seri kısaysa araç bunu kendiliğinden düşürür ve söyler."
        ),
    )


def _is_level(ctx: ToolContext, table_ref: TableRef | None, column: str) -> bool:
    """Whether this column is a level (a balance or a flow), read from the
    lineage the tools recorded. Unknown counts as not a level: differencing
    the log of something that turns out to be a rate would be wrong, while
    differencing a level without logs is merely coarser."""
    entry = _lineage_entry(ctx, table_ref, column)
    return bool(entry and entry.kind in _LOG_KINDS)


def make_test_causality() -> ToolSpec:
    """Granger lead-lag test plus lagged correlation (analysis/causality.py)."""

    def run(args: TestCausalityArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table(args.table_ref)
        missing = [
            name for name in (args.column, args.against) if name not in frame.columns
        ]
        if missing:
            return ToolResult(
                ok=False,
                summary=(
                    f"{', '.join(repr(name) for name in missing)} sütunu tabloda yok. "
                    f"Mevcut sütunlar: {', '.join(map(str, frame.columns))}."
                ),
            )
        if args.column == args.against:
            return ToolResult(
                ok=False,
                summary="Aynı sütun iki kez verildi; iki farklı sütun gerekiyor.",
            )

        try:
            result, correlation, summary = lead_lag(
                frame,
                args.column,
                args.against,
                max_lag=args.max_lag,
                log_x=_is_level(ctx, args.table_ref, args.column),
                log_y=_is_level(ctx, args.table_ref, args.against),
            )
        except NotEnoughData as exc:
            return ToolResult(ok=False, summary=f"Öncüllük sınanamadı: {exc}")
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the loop
            return ToolResult(
                ok=False, summary=f"test_causality çalıştırılamadı: {exc}"
            )

        ref = ctx.new_table_ref("causality")
        ctx.put_table(ref, result)
        lag_ref = ctx.new_table_ref("lag_corr")
        ctx.put_table(lag_ref, correlation)

        sources = sorted(
            {
                series
                for column in (args.column, args.against)
                for series in _upstream_source_series(ctx, args.table_ref, column)
            }
        )
        formulas = {
            "yon": "sınanan yön: X -> Y, 'X'in geçmişi Y'yi öngörüyor mu'",
            "en_iyi_gecikme": "p değeri en küçük olan gecikme (dönem)",
            "p_degeri": "Granger F testinin p değeri (ssr_ftest)",
            "anlamli": "p < 0,05 / denenen gecikme sayısı (Bonferroni)",
            "gozlem": "durağanlaştırma sonrası ortak gözlem sayısı",
            "donusum": "durağanlık için uygulanan dönüşüm: yok | fark | log-fark",
        }
        columns = [
            ColumnInfo(
                name=name,
                op="test_causality",
                formula=formulas.get(name),
                inputs=[args.column, args.against],
                source_series=sources,
                tool="test_causality",
                step_no=ctx.step_no,
            )
            for name in result.columns
        ]

        return ToolResult(
            ok=True,
            summary=(
                f"{summary} Gecikmeli korelasyon tablosu: {lag_ref} "
                f"({len(correlation)} satır)."
            ),
            table_ref=ref,
            preview=_preview_rows(result),
            columns=columns,
        )

    return ToolSpec(
        name="test_causality",
        description_tr=(
            "İki sütun arasında öncüllük sınar: Granger testi (iki yönde, "
            "durağanlaştırma ve çoklu test düzeltmesiyle) ve gecikmeli "
            "korelasyon. Nedensellik kanıtlamaz, hangi serinin önde gittiğini "
            "ölçer."
        ),
        args_model=TestCausalityArgs,
        fn=run,
        brief_name="Causality",
    )


#: `tools.toolset.make_registry` wires these into the production registry --
#: this module only builds the specs, it never registers itself.
TOOLS = [make_detect_anomalies(), make_detect_changes(), make_test_causality()]
