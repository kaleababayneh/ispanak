"""The chart tool and the figure it builds.

The figure is a plain dict -- plotly's wire format -- so it is compared field
by field against values worked out by hand, without the library. What the
tool writes goes through `ToolContext.add_chart` only: the loop merges both
that buffer and `ToolResult.artifacts`, so a chart returned both ways would
be counted twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.agent.state import ChartSeries, ChartSpec
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.tools import charts
from lakehouse_agent.tools.charts import MakeChartArgs, figure_json


def make_ctx(
    settings, tables=None, active=None, existing=(), lineage=None
) -> ToolContext:
    return ToolContext(
        session_id="s1",
        step_no=3,
        llm=FakeLLM(),
        lake=None,
        settings=settings,
        active_table=active,
        _tables=tables or {},
        lineage=lineage or {},
        charts=tuple(existing),
    )


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "period": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
            "faiz": [10.5, np.nan, 12.0],
            "bakiye": [100.0, 110.0, 121.0],
        }
    )


# --- figure_json ------------------------------------------------------------


def test_a_line_chart_is_one_scatter_trace_per_series(frame):
    figure = figure_json(frame, "period", [ChartSeries(column="faiz")], "Faiz")

    assert list(figure) == ["data", "layout"]
    (trace,) = figure["data"]
    assert trace["type"] == "scatter"
    assert trace["mode"] == "lines"
    assert trace["name"] == "faiz"
    assert trace["x"] == ["2021-01-01", "2021-02-01", "2021-03-01"]
    assert figure["layout"]["title"] == {"text": "Faiz"}
    assert figure["layout"]["xaxis"]["title"] == {"text": "period"}
    assert figure["layout"]["yaxis"]["title"] == {"text": "faiz"}
    assert "yaxis2" not in figure["layout"]


def test_a_missing_value_is_null_not_dropped_and_not_filled(frame):
    """CLAUDE.md #2: a gap stays a gap. Plotly draws a break for a null; a
    dropped row would join the two neighbours with a straight line and draw
    a February that was never measured."""
    (trace,) = figure_json(frame, "period", [ChartSeries(column="faiz")])["data"]

    assert trace["y"] == [10.5, None, 12.0]
    assert len(trace["y"]) == len(trace["x"])


def test_a_right_axis_series_gets_its_own_axis_and_a_bar_its_own_type(frame):
    figure = figure_json(
        frame,
        "period",
        [
            ChartSeries(column="faiz", axis="left", kind="line"),
            ChartSeries(column="bakiye", axis="right", kind="bar"),
        ],
    )

    left, right = figure["data"]
    assert "yaxis" not in left
    assert right["yaxis"] == "y2"
    assert right["type"] == "bar"
    assert "mode" not in right
    assert figure["layout"]["yaxis2"] == {
        "title": {"text": "bakiye"},
        "overlaying": "y",
        "side": "right",
        "exponentformat": "none",
    }
    assert left["marker"]["color"] != right["marker"]["color"]


def test_value_axes_write_numbers_out_in_turkish_form(frame):
    """ "300k" beside "milyon TL" was read as 300 million; the value was
    300 thousand million. Full digits, Turkish separators, on both axes."""
    figure = figure_json(
        frame,
        "period",
        [ChartSeries(column="faiz"), ChartSeries(column="bakiye", axis="right")],
    )

    layout = figure["layout"]
    assert layout["separators"] == ",."  # decimal comma, thousands point
    assert layout["yaxis"]["exponentformat"] == "none"
    assert layout["yaxis2"]["exponentformat"] == "none"
    assert "exponentformat" not in layout["xaxis"]  # dates are not numbers


def test_an_axis_with_several_series_is_titled_by_their_shared_unit(frame):
    """Two long labels joined by a comma ran off the top of the chart."""
    series = [ChartSeries(column="faiz"), ChartSeries(column="bakiye")]
    labels = {"faiz": "Konut kredisi faiz oranı", "bakiye": "Konut kredisi bakiyesi"}

    same = figure_json(
        frame, "period", series, labels=labels, units={"faiz": "%", "bakiye": "%"}
    )
    mixed = figure_json(
        frame, "period", series, labels=labels, units={"faiz": "%", "bakiye": "TL"}
    )
    alone = figure_json(frame, "period", series[:1], labels=labels, units={"faiz": "%"})

    assert same["layout"]["yaxis"]["title"] == {"text": "%"}
    assert mixed["layout"]["yaxis"]["title"] == {
        "text": "Konut kredisi faiz oranı, Konut kredisi bakiyesi"
    }
    assert alone["layout"]["yaxis"]["title"] == {"text": "Konut kredisi faiz oranı"}


def test_a_missing_column_is_named_not_skipped(frame):
    with pytest.raises(KeyError, match="yok"):
        figure_json(frame, "period", [ChartSeries(column="yok")])
    with pytest.raises(KeyError, match="tarih"):
        figure_json(frame, "tarih", [ChartSeries(column="faiz")])


def test_the_figure_is_json_serialisable(frame):
    """`Event.payload` must serialise; numpy scalars and Timestamps do not."""
    import json

    figure = figure_json(
        frame, "period", [ChartSeries(column="faiz"), ChartSeries(column="bakiye")]
    )

    json.dumps(figure)  # raises TypeError on a numpy scalar


def test_labels_name_the_traces_and_the_axes_but_not_the_data(frame):
    """A person reads the legend; the model reads the column. Both stay."""
    labels = {"period": "Dönem", "faiz": "Konut kredisi faizi (%)"}

    figure = figure_json(
        frame,
        "period",
        [ChartSeries(column="faiz"), ChartSeries(column="bakiye", axis="right")],
        labels=labels,
    )

    left, right = figure["data"]
    assert left["name"] == "Konut kredisi faizi (%)"
    assert right["name"] == "bakiye"  # no label: the name
    assert figure["layout"]["xaxis"]["title"] == {"text": "Dönem"}
    assert figure["layout"]["yaxis"]["title"] == {"text": "Konut kredisi faizi (%)"}
    assert figure["layout"]["yaxis2"]["title"] == {"text": "bakiye"}
    assert left["y"] == [10.5, None, 12.0]  # the data did not move


# --- the tool -----------------------------------------------------------------


def _tool():
    (spec,) = charts.TOOLS
    return spec


def test_the_tool_is_registered_under_its_name():
    assert _tool().name == "make_chart"
    assert _tool().brief_name == "Chart"


def test_make_chart_writes_a_spec_and_touches_no_table(agent_settings, frame):
    ctx = make_ctx(agent_settings, {"t01_get_series": frame}, active="t01_get_series")
    args = MakeChartArgs(
        series=[{"column": "faiz"}, {"column": "bakiye", "axis": "right"}],
        title="Faiz ve bakiye",
    )

    result = _tool().fn(args, ctx)

    assert result.ok, result.summary
    assert result.table_ref is None
    assert result.artifacts == []
    assert ctx.written == {}
    assert ctx.new_active is None
    (spec,) = ctx.written_charts
    assert isinstance(spec, ChartSpec)
    assert spec.chart_ref == "c01"
    assert spec.table_ref == "t01_get_series"
    assert spec.step_no == 3
    assert spec.title == "Faiz ve bakiye"
    assert [one.column for one in spec.series] == ["faiz", "bakiye"]
    assert spec.plotly_json and len(spec.plotly_json["data"]) == 2
    assert "c01" in result.summary and "t01_get_series" in result.summary


def test_make_chart_labels_from_the_sessions_lineage(agent_settings, frame):
    from lakehouse_agent.contracts import ColumnInfo

    lineage = {
        "t01_a.faiz": ColumnInfo(name="faiz", label_tr="Konut Kredisi Faizi", unit="%")
    }
    ctx = make_ctx(agent_settings, {"t01_a": frame}, active="t01_a", lineage=lineage)

    result = _tool().fn(MakeChartArgs(series=[{"column": "faiz"}]), ctx)

    assert result.ok
    (trace,) = ctx.written_charts[0].plotly_json["data"]
    assert trace["name"] == "Konut Kredisi Faizi (%)"
    assert ctx.written_charts[0].series[0].column == "faiz"  # the spec keeps the name


def test_the_models_label_wins_over_the_catalogs_and_is_recorded(agent_settings, frame):
    """The model knows the question; the catalog knows the series. A label
    the model gives is what the legend and the axis say, and the spec keeps
    it so the table header can say the same. A blank one is not a label."""
    from lakehouse_agent.contracts import ColumnInfo

    lineage = {
        "t01_a.faiz": ColumnInfo(name="faiz", label_tr="Konut Kredisi (TL, Akım, %)"),
        "t01_a.bakiye": ColumnInfo(
            name="bakiye", label_tr="Tüketici Kredileri - Konut"
        ),
    }
    ctx = make_ctx(agent_settings, {"t01_a": frame}, active="t01_a", lineage=lineage)
    args = MakeChartArgs(
        series=[
            {"column": "faiz", "label": "Konut kredisi faiz oranı"},
            {"column": "bakiye", "axis": "right", "label": "   "},
        ]
    )

    result = _tool().fn(args, ctx)

    assert result.ok
    spec = ctx.written_charts[0]
    left, right = spec.plotly_json["data"]
    assert left["name"] == "Konut kredisi faiz oranı"
    assert right["name"] == "Tüketici Kredileri - Konut"  # blank: the catalog
    assert spec.plotly_json["layout"]["yaxis"]["title"] == {
        "text": "Konut kredisi faiz oranı"
    }
    assert spec.series[0].label == "Konut kredisi faiz oranı"
    assert spec.series[0].column == "faiz"  # the name is still the name


def test_a_derived_series_on_the_chart_is_named_from_the_models_label(agent_settings):
    """Chart `faiz` as "Faiz oranı" and its unlabelled diff beside it: the
    diff's legend entry starts from the model's word, not the catalog's."""
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame(
        {"period": [1, 2], "faiz": [1.0, 2.0], "faiz_diff": [None, 1.0]}
    )
    lineage = {
        "t01_a.faiz": ColumnInfo(name="faiz", label_tr="Konut Kredisi (TL, Akım, %)"),
        "t01_a.faiz_diff": ColumnInfo(name="faiz_diff", op="diff", inputs=["faiz"]),
    }
    ctx = make_ctx(agent_settings, {"t01_a": frame}, active="t01_a", lineage=lineage)
    args = MakeChartArgs(
        series=[
            {"column": "faiz", "label": "Faiz oranı"},
            {"column": "faiz_diff", "axis": "right"},
        ]
    )

    assert _tool().fn(args, ctx).ok
    left, right = ctx.written_charts[0].plotly_json["data"]
    assert (left["name"], right["name"]) == ("Faiz oranı", "Faiz oranı — fark")


def test_the_schema_offers_label_and_bounds_it():
    from lakehouse_agent.tools.registry import tool_schema

    one = tool_schema(MakeChartArgs)["properties"]["series"]["items"]["properties"]

    assert "label" in one
    with pytest.raises(ValueError):
        MakeChartArgs(series=[{"column": "x", "label": "a" * 81}])


def test_the_chart_reference_counts_what_the_session_already_holds(
    agent_settings, frame
):
    existing = [ChartSpec(chart_ref="c02", table_ref="t01_a")]
    ctx = make_ctx(agent_settings, {"t01_a": frame}, active="t01_a", existing=existing)

    _tool().fn(MakeChartArgs(series=[{"column": "faiz"}]), ctx)
    _tool().fn(MakeChartArgs(series=[{"column": "bakiye"}]), ctx)

    assert [c.chart_ref for c in ctx.written_charts] == ["c03", "c04"]


def test_a_wrong_column_name_is_refused_with_the_real_names(agent_settings, frame):
    ctx = make_ctx(agent_settings, {"t01_a": frame}, active="t01_a")

    result = _tool().fn(MakeChartArgs(series=[{"column": "faizz"}]), ctx)

    assert not result.ok
    assert "'faizz'" in result.summary
    assert "faiz, bakiye" in result.summary
    assert ctx.written_charts == []


def test_no_active_table_is_a_refusal_not_a_crash(agent_settings):
    result = _tool().fn(
        MakeChartArgs(series=[{"column": "x"}]), make_ctx(agent_settings)
    )

    assert not result.ok
    assert "Aktif tablo yok" in result.summary


def test_a_series_with_nothing_to_draw_is_warned_about(agent_settings):
    empty = pd.DataFrame({"period": [1, 2], "bos": [np.nan, np.nan]})
    ctx = make_ctx(agent_settings, {"t01_a": empty}, active="t01_a")

    result = _tool().fn(MakeChartArgs(series=[{"column": "bos"}]), ctx)

    assert result.ok
    assert result.warnings == ["Bu sutunlarda cizilecek deger yok: bos."]


def test_the_schema_tells_the_model_the_allowed_axes_and_kinds():
    """A bad value must be refused at validation, where the model is told
    what it may send, not inside the tool as a bare pydantic error."""
    from lakehouse_agent.tools.registry import tool_schema

    schema = tool_schema(MakeChartArgs)
    one = schema["properties"]["series"]["items"]["properties"]

    assert one["axis"]["enum"] == ["left", "right"]
    assert one["kind"]["enum"] == ["line", "bar"]
    with pytest.raises(ValueError):
        MakeChartArgs(series=[{"column": "x", "axis": "middle"}])


def test_a_total_row_is_not_drawn_as_a_bar_and_the_tool_says_so(agent_settings):
    """Seen 23 Sep: Borsa Istanbul's gold table ends on "TOPLAM / TOTAL",
    and the chart drew it as a bar eight times the height of any month."""
    from lakehouse_agent.tools.charts import without_totals

    frame = pd.DataFrame(
        {
            "AY/MONTH": ["Ocak / January", "Şubat / February", "TOPLAM / TOTAL"],
            "hacim": [93.0, 90.0, 183.0],
        }
    )
    kept, dropped = without_totals(frame, "AY/MONTH")
    assert kept["AY/MONTH"].tolist() == ["Ocak / January", "Şubat / February"]
    assert dropped == ["TOPLAM / TOTAL"]

    ctx = make_ctx(agent_settings, {"t01_read_url": frame}, active="t01_read_url")
    result = _tool().fn(MakeChartArgs(x="AY/MONTH", series=[{"column": "hacim"}]), ctx)

    assert result.ok
    (trace,) = ctx.written_charts[0].plotly_json["data"]
    assert trace["x"] == ["Ocak / January", "Şubat / February"]
    assert any("Toplam satiri grafige alinmadi" in w for w in result.warnings)


def test_a_missing_axis_leaves_the_frame_alone():
    from lakehouse_agent.tools.charts import without_totals

    dated = pd.DataFrame({"period": pd.to_datetime(["2021-01-01"]), "v": [1.0]})
    kept, dropped = without_totals(dated, "yok")
    assert kept is dated and dropped == []
