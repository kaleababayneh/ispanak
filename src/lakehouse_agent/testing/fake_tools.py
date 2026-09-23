"""Stand-in tools for testing the loop.

The loop's job is control flow: choose a tool, validate its arguments, run it,
merge what it wrote, decide whether to ask a human, keep going or stop. None of
that depends on real series data, and testing it against real data would make
every loop test also a test of the maths.

So these tools do the smallest thing that is still shaped like the real one:
they take the same kind of arguments, write into the same context, and return
the same `ToolResult`. Each also has a failure mode the loop has to survive.
"""

from __future__ import annotations

import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.contracts import (
    ColumnInfo,
    QualityReport,
    Source,
    ToolResult,
)
from lakehouse_agent.testing.fake_lake import SCENARIO_SERIES
from lakehouse_agent.tools.lakehouse import column_name, deflate_refusal
from lakehouse_agent.tools.registry import ToolRegistry, ToolSpec

FETCHED_AT = pd.Timestamp("2026-09-15T06:00:00Z").to_pydatetime()


# --------------------------------------------------------------------------
# search_catalog
# --------------------------------------------------------------------------


class SearchCatalogArgs(BaseModel):
    query: str = Field(
        ...,
        description="Dogal dilde arama metni, ornek 'konut kredisi faiz orani'.",
        examples=["konut kredisi faiz orani"],
    )


def _hits(ambiguous: bool) -> list[dict]:
    top = {
        "series_id": "evds:TP.KTF12",
        "name_tr": "Konut Kredisi Faiz Orani (TL, Akim)",
        "unit": "%",
        "freq": "W",
        "kind": "rate",
        "score": 0.0163,
    }
    if not ambiguous:
        return [top]
    return [
        top,
        {
            "series_id": "evds:TP.BKR.TRY.18",
            "name_tr": "Konut Kredisi Faiz Orani (TL, Stok)",
            "unit": "%",
            "freq": "M",
            "kind": "rate",
            "score": 0.0161,
        },
    ]


def make_search_catalog(*, ambiguous: bool = False) -> ToolSpec:
    """Catalog search. `ambiguous=True` returns the flow/stock pair.

    That pair is the whole reason the series-choice approval exists, so a test
    can turn it on and off and watch the gate fire or stay quiet.
    """

    def run(args: SearchCatalogArgs, ctx: ToolContext) -> ToolResult:
        hits = _hits(ambiguous)
        return ToolResult(
            ok=True,
            summary=f"{len(hits)} aday seri bulundu.",
            preview=hits,
        )

    return ToolSpec(
        name="search_catalog",
        description_tr="Katalogda dogal dille seri arar ve aday seri kimliklerini dondurur.",
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
        description="Katalogdaki series_id listesi, ornek ['evds:TP.KTF12']. SQL veya seri adi degil.",
        examples=[["evds:TP.KTF12"]],
    )
    start: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}$",
        description="Baslangic ayi YYYY-AA.",
        examples=["2021-01"],
    )
    end: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}$",
        description="Bitis ayi YYYY-AA.",
        examples=["2025-12"],
    )
    into: Literal["new", "active"] = Field(
        "new",
        description="'active': aktif tabloya sutun olarak ekle, tabloyu bozma.",
    )


def _months(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{start}-01", f"{end}-01", freq="MS")


#: The real tools' naming rule, not a copy of it: a fake that names columns its
#: own way lets a test pass against names production never produces.
_column_name = column_name


def make_get_series() -> ToolSpec:
    """Fetch series into a new table, or add them to the active one."""

    def run(args: GetSeriesArgs, ctx: ToolContext) -> ToolResult:
        periods = _months(args.start, args.end)
        columns: list[ColumnInfo] = []
        frame = pd.DataFrame({"period": periods})

        for index, series_id in enumerate(args.series_ids):
            name = _column_name(series_id)
            frame[name] = [10.0 + index + row * 0.5 for row in range(len(periods))]
            columns.append(
                ColumnInfo(
                    name=name,
                    # The real tool copies the catalog name into `label_tr`
                    # and the interfaces label charts and headers from it; a
                    # fake that left it empty would rehearse a screen the
                    # demo never shows. Unknown ids (tests invent them) stay
                    # unlabelled, as an uncatalogued series would.
                    label_tr=SCENARIO_SERIES.get(series_id, {}).get("name_tr"),
                    unit="%" if "KTF" in name else "milyon TL",
                    kind="rate" if "KTF" in name else "stock",
                    freq="M",
                    source_series=[series_id],
                    tool="get_series",
                    step_no=ctx.step_no,
                )
            )

        if args.into == "active" and ctx.has_table():
            existing = ctx.table()
            # Refused, not merged. `merge` renames a colliding column to
            # `<name>_x` / `<name>_y`, so the column the user was looking at
            # silently changes name -- and the loop's "existing columns came
            # back unchanged" check cannot see it, because the old name is no
            # longer in the frame to compare. Both lake implementations guard
            # the same case for the same reason.
            clash = [
                name
                for name in frame.columns
                if name != "period" and name in existing.columns
            ]
            if clash:
                return ToolResult(
                    ok=False,
                    summary=(
                        f"{', '.join(clash)} sutunu tabloda zaten var. "
                        "into='new' ile ayri bir tablo olustur."
                    ),
                )
            merged = existing.merge(frame, on="period", how="left")
            ref = ctx.active_table or ctx.new_table_ref("get_series")
            ctx.put_table(ref, merged)
        else:
            ref = ctx.new_table_ref("get_series")
            ctx.put_table(ref, frame)
            ctx.set_active(ref)

        return ToolResult(
            ok=True,
            summary=f"{len(args.series_ids)} seri, {len(periods)} satir.",
            table_ref=ref,
            preview=[{"period": str(periods[0].date())}],
            columns=columns,
            sources=[
                Source(series_id=series_id, fetched_at=FETCHED_AT)
                for series_id in args.series_ids
            ],
        )

    return ToolSpec(
        name="get_series",
        description_tr="Katalogdaki serileri belirtilen aralikta getirir.",
        args_model=GetSeriesArgs,
        fn=run,
        brief_name="Lakehouse",
    )


# --------------------------------------------------------------------------
# transform_table
# --------------------------------------------------------------------------


class TransformArgs(BaseModel):
    op: Literal["deflate", "diff"] = Field(
        ..., description="Uygulanacak islem: deflate veya diff."
    )
    column: str = Field(
        ...,
        description="Islenecek sutun adi. Onceki arac ciktisindan birebir kopyalayin.",
    )
    table_ref: str | None = Field(
        None, description="Hedef tablo; bos birakilirsa aktif tablo kullanilir."
    )


def make_transform_table() -> ToolSpec:
    """Adds one derived column. Never touches the ones already there."""

    def run(args: TransformArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table(args.table_ref)
        if args.column not in frame.columns:
            return ToolResult(
                ok=False,
                summary=(
                    f"'{args.column}' sutunu tabloda yok. "
                    f"Mevcut sutunlar: {', '.join(map(str, frame.columns))}."
                ),
            )

        ref = args.table_ref or ctx.active_table or ""
        warnings: list[str] = []
        upstream = ctx.lineage.get(f"{ref}.{args.column}")
        if args.op == "deflate":
            # The real tool's own rule, called rather than restated: a fake
            # that allowed a rate is how the 22 Sep live run produced a "real
            # rate", and a fake stricter or looser than production lets a test
            # pass that production would fail.
            refusal = deflate_refusal(args.column, upstream)
            if refusal:
                return ToolResult(ok=False, summary=refusal)
        new_name = (
            f"reel_{args.column}" if args.op == "deflate" else f"{args.column}_diff"
        )
        if new_name in frame.columns:
            # Deterministic here (no base month, no second argument), so the
            # same name is the same operation -- the real tool's message.
            return ToolResult(
                ok=False,
                summary=(
                    f"'{new_name}' sutunu ayni islemle zaten hesaplanmis; "
                    "mevcut sutunu kullan."
                ),
            )
        if args.op == "deflate":
            frame[new_name] = frame[args.column] * 0.8
            # A real deflated column descends from two series, the nominal one
            # and the price index, so the chain is read off the input column's
            # own lineage rather than asserted from nothing.
            column = ColumnInfo(
                name=new_name,
                unit=upstream.unit,
                kind=upstream.kind,
                freq=upstream.freq,
                op="deflate",
                formula="reel = nominal x TUFE_baz / TUFE_t",
                inputs=[args.column],
                source_series=[
                    *upstream.source_series,
                    "evds:TP.TUKFIY2025.GENEL",
                ],
                tool="transform_table",
                step_no=ctx.step_no,
            )
            # The one warning the scenario actually produces. CLAUDE.md #3 puts
            # conversions in `warnings` and planner.md rule 7 obliges the answer
            # to state the base period, so a fake that never warns leaves the
            # only route between those two rules untested.
            warnings.append(
                f"'{args.column}' 2025=100 bazina gore reellestirildi; "
                "rakamlar reel, nominal degil."
            )
        else:
            frame[new_name] = frame[args.column].diff()
            column = ColumnInfo(
                name=new_name,
                op="diff",
                formula="fark = deger_t - deger_t-1",
                inputs=[args.column],
                tool="transform_table",
                step_no=ctx.step_no,
            )

        ctx.put_table(ref, frame)
        return ToolResult(
            ok=True,
            summary=f"'{new_name}' sutunu eklendi.",
            table_ref=ref,
            columns=[column],
            warnings=warnings,
        )

    return ToolSpec(
        name="transform_table",
        description_tr="Aktif tabloya turetilmis bir sutun ekler; mevcut sutunlara dokunmaz.",
        args_model=TransformArgs,
        fn=run,
        brief_name="Lakehouse",
    )


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


class EmptyArgs(BaseModel):
    pass


def make_failing_tool() -> ToolSpec:
    """Raises. The loop must turn this into a tool message, not a dead turn."""

    def run(args: EmptyArgs, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("arac patladi")

    return ToolSpec(
        name="failing_tool",
        description_tr="Test icin her zaman hata veren arac.",
        args_model=EmptyArgs,
        fn=run,
    )


def make_sneaky_tool() -> ToolSpec:
    """Writes on the frame it was handed and never registers the change.

    Nothing stops a tool author from doing this by accident. The context hands
    out a shallow copy precisely so that when they do, the session table the
    user is already looking at does not change under them.
    """

    def run(args: EmptyArgs, ctx: ToolContext) -> ToolResult:
        frame = ctx.table()
        frame["gizli"] = 1.0
        frame.iloc[0, 1] = -999.0
        return ToolResult(ok=True, summary="hicbir sey yazmadim")

    return ToolSpec(
        name="sneaky_tool",
        description_tr="Test icin verilen tabloyu yerinde degistirmeye calisan arac.",
        args_model=EmptyArgs,
        fn=run,
    )


def make_corrupting_tool() -> ToolSpec:
    """Adds a column but also changes one that was already there.

    "Add a column without breaking the table" is the scenario's second
    question. A tool that quietly rewrites an existing column would produce an
    answer indistinguishable from a correct one, so the loop compares the old
    columns itself instead of trusting the tool's word.
    """

    def run(args: EmptyArgs, ctx: ToolContext) -> ToolResult:
        ref = ctx.active_table or ""
        frame = ctx.table(ref)
        target = [c for c in frame.columns if c != "period"][0]
        frame[target] = frame[target] * 2
        frame["yeni"] = 1.0
        ctx.put_table(ref, frame)
        return ToolResult(
            ok=True,
            summary="'yeni' sutunu eklendi.",
            table_ref=ref,
            columns=[
                ColumnInfo(name="yeni", tool="corrupting_tool", step_no=ctx.step_no)
            ],
        )

    return ToolSpec(
        name="corrupting_tool",
        description_tr="Test icin mevcut bir sutunu bozarak yeni sutun ekleyen arac.",
        args_model=EmptyArgs,
        fn=run,
    )


class SlowArgs(BaseModel):
    seconds: float = Field(1.0, description="Kac saniye bekleyecegi.")


def make_slow_tool() -> ToolSpec:
    """Blocks for real, so the step timeout is exercised the way it will be."""

    def run(args: SlowArgs, ctx: ToolContext) -> ToolResult:
        time.sleep(args.seconds)
        ref = ctx.new_table_ref("slow")
        ctx.put_table(ref, pd.DataFrame({"period": [pd.Timestamp("2021-01-01")]}))
        return ToolResult(ok=True, summary="bitti", table_ref=ref)

    return ToolSpec(
        name="slow_tool",
        description_tr="Test icin yavas calisan arac.",
        args_model=SlowArgs,
        fn=run,
    )


# --------------------------------------------------------------------------
# approval paths
# --------------------------------------------------------------------------


class IngestArgs(BaseModel):
    source: Literal["evds", "bddk_aylik"] = Field(
        ..., description="Verinin cekilecegi kaynak."
    )
    series_id: str = Field(..., description="Cekilecek seri kimligi.")


def make_ingest() -> ToolSpec:
    """Fetching from outside the pool. Always goes through the fetch gate."""

    def run(args: IngestArgs, ctx: ToolContext) -> ToolResult:
        ref = ctx.new_table_ref("ingest")
        ctx.put_table(
            ref, pd.DataFrame({"period": [pd.Timestamp("2021-01-01")], "value": [1.0]})
        )
        ctx.set_active(ref)
        return ToolResult(
            ok=True,
            summary=f"{args.series_id} canli cekildi.",
            table_ref=ref,
            sources=[Source(series_id=args.series_id, fetched_at=FETCHED_AT)],
        )

    def preview(args: IngestArgs, ctx: ToolContext) -> dict:
        return {
            "source": args.source.upper(),
            "detail": f"{args.series_id}, tahmini 6 cagri",
            "series_id": args.series_id,
        }

    return ToolSpec(
        name="ingest",
        description_tr="Havuzda olmayan bir seriyi canli olarak ceker.",
        args_model=IngestArgs,
        fn=run,
        brief_name="Lakehouse",
        approval="fetch",
        approval_preview=preview,
    )


def make_quality_table(*, missing_pct: float = 20.0) -> ToolSpec:
    """Produces a table with a bad quality card, to exercise the quality gate."""

    def run(args: EmptyArgs, ctx: ToolContext) -> ToolResult:
        ref = ctx.new_table_ref("quality")
        frame = pd.DataFrame(
            {
                "period": pd.date_range("2021-01-01", periods=6, freq="MS"),
                "value": range(6),
            }
        )
        ctx.put_table(ref, frame)
        ctx.set_active(ref)
        return ToolResult(
            ok=True,
            summary="Tablo hazir ama kalitesi tartismali.",
            table_ref=ref,
            quality=QualityReport(
                table_ref=ref,
                rows=6,
                freq="M",
                missing_periods=2,
                missing_pct=missing_pct,
                unit_guess=None,
            ),
        )

    return ToolSpec(
        name="quality_table",
        description_tr="Test icin kalite sorunlu tablo ureten arac.",
        args_model=EmptyArgs,
        fn=run,
    )


# --------------------------------------------------------------------------


def make_fake_registry(
    *, ambiguous_search: bool = False, missing_pct: float = 20.0
) -> ToolRegistry:
    """The default set of fakes. Tests override individual tools as needed."""
    return ToolRegistry.of(
        [
            make_search_catalog(ambiguous=ambiguous_search),
            make_get_series(),
            make_transform_table(),
            make_failing_tool(),
            make_sneaky_tool(),
            make_corrupting_tool(),
            make_slow_tool(),
            make_ingest(),
            make_quality_table(missing_pct=missing_pct),
        ]
    )
