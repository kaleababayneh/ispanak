"""The chart tool: the model describes a chart, this module builds its figure.

Nothing here imports plotly. A plotly figure's wire format is plain JSON --
`{"data": [...], "layout": {...}}` -- so the dict is built directly and the
library stays where it belongs, in the interface that renders it. That also
keeps `Event.payload` serialisable, which `docs/contracts.md` section 6
requires and which a `Figure` object would break.

Per the same section, the `chart` event carries `{chart_ref, plotly_json}`.
The chart goes back through `ctx.add_chart`, not through `ToolResult.artifacts`:
the loop merges both into the session, so returning it twice would count the
chart twice, and `contracts.Artifact` has no field to carry the figure in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from lakehouse_agent.agent.context import TableNotFound, ToolContext
from lakehouse_agent.agent.labels import column_units, table_labels
from lakehouse_agent.agent.state import ChartSeries, ChartSpec
from lakehouse_agent.contracts import TableRef, ToolResult
from lakehouse_agent.tools.registry import ToolSpec

#: Value-axis ticks are written out in full. Plotly's default abbreviates
#: with SI prefixes, and "300k" next to an axis titled "milyon TL" read as
#: three hundred million when the value is three hundred thousand million
#: (measured 23 Sep: 276,785 milyon TL in 2021-01). The unit is not rescaled
#: -- CLAUDE.md #3, and the table shows the same number -- the number is
#: just no longer abbreviated.
TICKS: dict[str, Any] = {"exponentformat": "none"}

#: Plotly's own default palette. Written out rather than imported so the
#: engine stays free of the library; the interface uses the same colours.
PALETTE = (
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
    "#b6e880",
)


def _cell(value: Any) -> Any:
    """One value as JSON. Missing stays missing.

    `None` rather than a number: CLAUDE.md #2 forbids filling gaps, and plotly
    draws a break in the line for a null, which is the honest picture. Dropping
    the row instead would join the two sides of a gap with a straight segment
    and invent the months in between.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def _axis_title(
    series: list[ChartSeries],
    side: str,
    labels: Mapping[str, str],
    units: Mapping[str, str],
) -> str:
    """One series: its label. Several sharing a unit: the unit -- two long
    labels joined by a comma ran off the top of the chart. Several with
    different units: the labels, since nothing shorter would be true."""
    mine = [one.column for one in series if one.axis == side]
    if len(mine) == 1:
        return labels.get(mine[0], mine[0])
    shared = {units.get(name, "") for name in mine}
    if len(shared) == 1 and next(iter(shared)):
        return next(iter(shared))
    return ", ".join(labels.get(name, name) for name in mine)


def figure_json(
    frame: pd.DataFrame,
    x: str,
    series: list[ChartSeries],
    title: str | None = None,
    labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """A plotly figure as a plain dict.

    `labels` maps a column name to what a person should read in the legend
    and on the axes (`agent/labels.py`); a column with no label keeps its
    name. The columns themselves are still addressed by name: the label is
    for the eye, the name is what the model works with.

    Raises `KeyError` naming the column when `x` or any series column is not
    in the frame. Tools must not quietly draw a different chart than the one
    they were asked for -- a missing column is the model having copied a name
    wrong, and the model can only fix what it is told about.
    """
    labels = labels or {}
    units = units or {}
    if x not in frame.columns:
        raise KeyError(x)
    missing = [one.column for one in series if one.column not in frame.columns]
    if missing:
        raise KeyError(missing[0])
    if not series:
        raise ValueError("Grafik icin en az bir seri gerekiyor.")

    xs = [_cell(value) for value in frame[x]]
    traces = []
    for index, one in enumerate(series):
        trace: dict[str, Any] = {
            "type": "scatter" if one.kind == "line" else "bar",
            "name": labels.get(one.column, one.column),
            "x": xs,
            "y": [_cell(value) for value in frame[one.column]],
            "marker": {"color": PALETTE[index % len(PALETTE)]},
        }
        if one.kind == "line":
            trace["mode"] = "lines"
        if one.axis == "right":
            trace["yaxis"] = "y2"
        traces.append(trace)

    layout: dict[str, Any] = {
        "xaxis": {"title": {"text": labels.get(x, x)}},
        "yaxis": {
            "title": {"text": _axis_title(series, "left", labels, units)},
            **TICKS,
        },
        "legend": {"orientation": "h"},
        "margin": {"t": 48, "r": 48, "b": 48, "l": 64},
        # Turkish: decimal comma, thousands point. Plotly's default is the
        # English pair, which put "40.3" on a Turkish screen.
        "separators": ",.",
    }
    if title:
        layout["title"] = {"text": title}
    if any(one.axis == "right" for one in series):
        layout["yaxis2"] = {
            "title": {"text": _axis_title(series, "right", labels, units)},
            "overlaying": "y",
            "side": "right",
            **TICKS,
        }
    return {"data": traces, "layout": layout}


class SeriesArg(BaseModel):
    """One line or bar on the chart."""

    column: str = Field(
        ...,
        description="Cizilecek sutun adi, onceki arac ciktisindan birebir kopyalanir.",
        examples=["TP_KTF12"],
    )
    axis: Literal["left", "right"] = Field(
        "left",
        description="Hangi eksende cizilecek. Farkli birimdeki sutun right'a.",
        examples=["left", "right"],
    )
    kind: Literal["line", "bar"] = Field(
        "line", description="Cizim turu.", examples=["line", "bar"]
    )
    label: str | None = Field(
        None,
        max_length=80,
        description=(
            "Lejantta okunacak kisa Turkce ad; anlami degistirmez, yalnizca "
            "okunur yazar. Bos birakilirsa katalog adi kullanilir."
        ),
        examples=["Konut kredisi faiz oranı"],
    )


class MakeChartArgs(BaseModel):
    table_ref: TableRef | None = Field(
        None,
        description="Grafigi cizilecek tablo. Bos birakilirsa aktif tablo.",
        examples=["t01_get_series"],
    )
    x: str = Field("period", description="Yatay eksendeki sutun.", examples=["period"])
    series: list[SeriesArg] = Field(
        ...,
        description="Cizilecek sutunlar. Farkli birimdeki sutunu right eksene koy.",
    )
    title: str | None = Field(
        None, description="Grafik basligi.", examples=["Konut kredisi faizi"]
    )


def make_make_chart() -> ToolSpec:
    def run(args: MakeChartArgs, ctx: ToolContext) -> ToolResult:
        try:
            frame = ctx.table(args.table_ref)
        except TableNotFound as exc:
            return ToolResult(ok=False, summary=str(exc))

        ref = args.table_ref or ctx.active_table or ""
        series = [
            ChartSeries(
                column=one.column, axis=one.axis, kind=one.kind, label=one.label
            )
            for one in args.series
        ]
        # The catalog's name for every column, then the model's word for
        # the ones it named. The model knows the question; the catalog
        # knows the series. A blank label is not an override.
        # The model's word for a column, where it gave one, reaches every
        # column derived from it too (`agent/labels.py`), so a `_diff` on
        # the same chart reads "<the model's label> — fark".
        named = {
            one.column: one.label.strip()
            for one in series
            if one.label and one.label.strip()
        }
        labels = table_labels(ref, frame.columns, ctx.lineage, named)
        units = column_units(ref, frame.columns, ctx.lineage)
        try:
            figure = figure_json(frame, args.x, series, args.title, labels, units)
        except KeyError as exc:
            available = ", ".join(map(str, frame.columns))
            return ToolResult(
                ok=False,
                summary=(
                    f"{ref} tablosunda {exc.args[0]!r} diye bir sutun yok. "
                    f"Var olanlar: {available}."
                ),
            )
        except ValueError as exc:
            return ToolResult(ok=False, summary=str(exc))

        spec = ChartSpec(
            chart_ref=ctx.new_chart_ref(),
            table_ref=ref,
            x=args.x,
            series=series,
            title=args.title,
            plotly_json=figure,
            step_no=ctx.step_no,
        )
        ctx.add_chart(spec)

        drawn = ", ".join(one.column for one in series)
        empty = [
            one.column
            for one, trace in zip(series, figure["data"], strict=True)
            if all(value is None for value in trace["y"])
        ]
        warnings = (
            [f"Bu sutunlarda cizilecek deger yok: {', '.join(empty)}."] if empty else []
        )
        return ToolResult(
            ok=True,
            summary=(
                f"Grafik {spec.chart_ref} hazir: {ref} tablosundan {drawn} "
                f"({len(frame)} donem, yatay eksen {args.x}). "
                "Grafik arayuzde gorunur, tablo degismedi."
            ),
            table_ref=None,
            warnings=warnings,
        )

    return ToolSpec(
        name="make_chart",
        description_tr=(
            "Oturumdaki bir tablodan grafik cizer. Sutun adlarini onceki arac "
            "ciktisindan birebir kopyala. Farkli birimdeki (orn. oran ve tutar) "
            "sutunlari ayni eksene koyma, birini right eksene al. Tabloyu "
            "degistirmez, yeni tablo kurmaz."
        ),
        args_model=MakeChartArgs,
        fn=run,
        brief_name="Chart",
    )


#: `tools.toolset.make_registry` wires these in; this module never registers
#: itself, the same as every other tool module.
TOOLS = [make_make_chart()]
