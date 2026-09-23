"""The browser interface's testable half.

`app.py` at the repository root is the Chainlit shell and cannot be imported
here, so every decision it would otherwise make is in `agent/webui.py` and
pinned by these tests. `webui` itself must import neither `chainlit` nor
`plotly`; `test_import_discipline.py` enforces that for the whole package.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from lakehouse_agent.agent import webui
from lakehouse_agent.agent.state import ApprovalOption, ApprovalRequest, ChartSpec


def _request(default="approve"):
    return ApprovalRequest(
        request_id="r1",
        kind="fetch",
        question="Havuzda olmayan veri cekilsin mi?",
        options=[
            ApprovalOption(id="approve", label="Evet, cek", detail="3 istek"),
            ApprovalOption(id="cancel", label="Vazgec"),
        ],
        default=default,
        created_at=datetime.now(UTC),
    )


# --- the approval card --------------------------------------------------------


def test_every_option_becomes_a_button_and_the_default_is_marked():
    specs = webui.action_specs(_request())

    assert [spec["name"] for spec in specs] == ["approve", "cancel"]
    assert [spec["payload"] for spec in specs] == [
        {"value": "approve"},
        {"value": "cancel"},
    ]
    assert specs[0]["label"] == "Evet, cek (varsayilan)"
    assert specs[1]["label"] == "Vazgec"
    assert specs[0]["tooltip"] == "3 istek"
    assert specs[1]["tooltip"] == "Vazgec"  # no detail: the label, never empty


def test_a_button_press_is_a_persons_choice():
    decision = webui.decision(_request(), "cancel")

    assert decision.choice == "cancel"
    assert decision.automatic is False
    assert decision.request_id == "r1"


def test_no_answer_falls_to_the_default_and_says_so():
    """A timeout or a closed tab is not a choice. `automatic` decides what
    the model is told afterwards, so a default must never look chosen."""
    for value in (None, "", "yok_boyle_secenek", 42):
        decision = webui.decision(_request(), value)
        assert decision.choice == "approve", value
        assert decision.automatic is True, value
        assert "varsayilan" in (decision.note or "")


def test_a_card_without_a_default_falls_to_an_empty_choice():
    decision = webui.decision(_request(default=None), None)

    assert decision.choice == ""
    assert decision.automatic is True


# --- the fallback chart -------------------------------------------------------


def test_the_default_chart_draws_the_numeric_columns_over_period():
    frame = pd.DataFrame(
        {
            "period": pd.to_datetime(["2021-01-01", "2021-02-01"]),
            "faiz": [1.0, 2.0],
            "not": ["a", "b"],
            "bos": [np.nan, np.nan],
        }
    )

    spec = webui.default_chart(frame, "t01_a", [])

    assert spec is not None
    assert spec.chart_ref == "c01"
    assert spec.table_ref == "t01_a"
    assert spec.x == "period"
    assert [one.column for one in spec.series] == ["faiz"]  # text and empty skipped
    assert "arayuz cizdi" in (spec.title or "")
    assert spec.plotly_json["layout"]["title"]["text"] == spec.title


def test_the_default_chart_numbers_after_the_charts_the_session_holds():
    frame = pd.DataFrame({"period": [1, 2], "v": [1.0, 2.0]})
    held = [ChartSpec(chart_ref="c03", table_ref="t01_a")]

    assert webui.default_chart(frame, "t01_a", held).chart_ref == "c04"


def test_the_default_chart_caps_how_many_series_it_draws():
    columns = {f"s{i}": [float(i)] * 2 for i in range(webui.DEFAULT_SERIES_MAX + 3)}
    frame = pd.DataFrame({"period": [1, 2], **columns})

    spec = webui.default_chart(frame, "t01_a", [])

    assert len(spec.series) == webui.DEFAULT_SERIES_MAX


def test_nothing_to_draw_means_no_chart():
    no_axis = pd.DataFrame({"v": [1.0]})
    no_numbers = pd.DataFrame({"period": [1], "ad": ["x"]})
    all_missing = pd.DataFrame({"period": [1], "v": [np.nan]})

    assert webui.default_chart(no_axis, "t01_a", []) is None
    assert webui.default_chart(no_numbers, "t01_a", []) is None
    assert webui.default_chart(all_missing, "t01_a", []) is None


def test_the_default_chart_uses_labels_when_it_is_given_lineage():
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": [1, 2], "TP_KTF12": [1.0, 2.0]})
    lineage = {"t01_a.TP_KTF12": ColumnInfo(name="TP_KTF12", label_tr="Faiz", unit="%")}

    spec = webui.default_chart(frame, "t01_a", [], lineage=lineage)

    assert spec.plotly_json["data"][0]["name"] == "Faiz (%)"
    assert spec.plotly_json["layout"]["xaxis"]["title"] == {"text": "Dönem"}
    assert spec.series[0].column == "TP_KTF12"


def test_a_table_is_drawn_when_first_seen_or_when_a_column_is_added():
    """A `top` fires an update with no new columns; sixty rows again under
    "the three highest months" was noise."""
    known: set[str] = set()

    assert webui.note_table(known, "t01_a", []) == (True, True)  # first sighting
    assert webui.note_table(known, "t01_a", ["x_diff"]) == (False, True)
    assert webui.note_table(known, "t01_a", []) == (False, False)  # only read
    assert known == {"t01_a"}


def test_the_default_chart_uses_the_models_labels_when_given_them():
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": [1, 2], "TP_KTF12": [1.0, 2.0]})
    lineage = {
        "t01_a.TP_KTF12": ColumnInfo(
            name="TP_KTF12", label_tr="Konut Kredisi (TL, Akım, %)"
        )
    }

    spec = webui.default_chart(
        frame, "t01_a", [], lineage=lineage, overrides={"TP_KTF12": "Faiz oranı"}
    )

    assert spec.plotly_json["data"][0]["name"] == "Faiz oranı"


def test_a_shared_axis_is_titled_by_the_unit_not_two_labels():
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": [1, 2], "a": [1.0, 2.0], "a_diff": [None, 1.0]})
    lineage = {
        "t01_a.a": ColumnInfo(name="a", label_tr="Bakiye", unit="milyon TL"),
        "t01_a.a_diff": ColumnInfo(
            name="a_diff", op="diff", inputs=["a"], unit="milyon TL"
        ),
    }

    spec = webui.default_chart(frame, "t01_a", [], lineage=lineage)

    assert spec.plotly_json["layout"]["yaxis"]["title"] == {"text": "milyon TL"}


def test_a_fallback_is_only_for_a_table_this_turn_created():
    """Seen 23 Sep, twice: a page-read answer got the previous question's
    chart, and a follow-up that only read the table got its chart again.
    The active table outlives the turn; the fallback must not."""
    assert webui.fallback_table("t03_a", {"t03_a"}, charts_added=0) == "t03_a"
    assert webui.fallback_table("t03_a", set(), charts_added=0) is None  # follow-up
    assert webui.fallback_table("t03_a", {"t04_b"}, charts_added=0) is None
    assert webui.fallback_table("t03_a", {"t03_a"}, charts_added=1) is None  # drawn
    assert webui.fallback_table(None, {"t03_a"}, charts_added=0) is None


def test_the_default_chart_puts_a_second_unit_on_the_right_axis_and_drops_a_third():
    """Counts and a percentage on one axis drew the percentage flat at zero."""
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame(
        {
            "period": [1, 2],
            "ipotekli": [10.0, 20.0],
            "toplam": [100.0, 200.0],
            "pay": [10.0, 10.0],
            "tutar": [5.0, 6.0],
        }
    )
    lineage = {
        "t01_a.ipotekli": ColumnInfo(name="ipotekli", unit="Adet"),
        "t01_a.toplam": ColumnInfo(name="toplam", unit="adet"),  # case-folded
        "t01_a.pay": ColumnInfo(name="pay", unit="%"),
        "t01_a.tutar": ColumnInfo(name="tutar", unit="milyon TL"),
    }

    spec = webui.default_chart(frame, "t01_a", [], lineage=lineage)

    assert [(one.column, one.axis) for one in spec.series] == [
        ("ipotekli", "left"),
        ("toplam", "left"),
        ("pay", "right"),
    ]  # `tutar`, a third unit, is not drawn on someone else's scale
    assert "yaxis2" in spec.plotly_json["layout"]


def test_the_default_chart_without_lineage_keeps_everything_on_one_axis():
    frame = pd.DataFrame({"period": [1, 2], "a": [1.0, 2.0], "b": [3.0, 4.0]})

    spec = webui.default_chart(frame, "t01_a", [])

    assert [one.axis for one in spec.series] == ["left", "left"]


# --- the displayed table ------------------------------------------------------


def test_the_displayed_table_has_labelled_headers_and_untouched_values():
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": [1], "TP_KTF12": [40.3], "x": [1]})
    lineage = {"t01_a.TP_KTF12": ColumnInfo(name="TP_KTF12", label_tr="Faiz", unit="%")}

    shown = webui.display_frame(frame, "t01_a", lineage)

    assert list(shown.columns) == ["Dönem", "Faiz (%)", "x"]
    assert shown.iloc[0].tolist() == [1, 40.3, 1]
    assert list(frame.columns) == ["period", "TP_KTF12", "x"]  # the session's frame


def test_the_displayed_table_shows_dates_without_a_time_of_day():
    """Seen 23 Sep in the browser: `2021-01-01T00:00:00.000` for a month.
    `period` is a date by contract; the time is serialisation noise."""
    frame = pd.DataFrame(
        {"period": pd.to_datetime(["2021-01-01", "2021-02-01"]), "v": [1.0, None]}
    )

    shown = webui.display_frame(frame, "t01_a", {})

    assert shown["Dönem"].tolist() == ["2021-01-01", "2021-02-01"]
    assert str(frame["period"].dtype).startswith("datetime64")  # the session's
    assert shown["v"].isna().tolist() == [False, True]  # gaps untouched


def test_chart_labels_come_from_the_tables_latest_chart_only():
    from lakehouse_agent.agent.state import ChartSeries

    charts = [
        ChartSpec(
            chart_ref="c01",
            table_ref="t01_a",
            series=[ChartSeries(column="faiz", label="Eski ad")],
        ),
        ChartSpec(
            chart_ref="c02",
            table_ref="t02_b",
            series=[ChartSeries(column="faiz", label="Baska tablo")],
        ),
        ChartSpec(
            chart_ref="c03",
            table_ref="t01_a",
            series=[
                ChartSeries(column="faiz", label="Konut kredisi faiz oranı"),
                ChartSeries(column="bakiye"),  # unlabelled: nothing
            ],
        ),
    ]

    assert webui.chart_labels(charts, "t01_a") == {"faiz": "Konut kredisi faiz oranı"}
    assert webui.chart_labels(charts, "t09_yok") == {}
    assert webui.chart_labels(
        [{"table_ref": "t01_a", "series": [{"column": "x", "label": "Y"}]}], "t01_a"
    ) == {"x": "Y"}


def test_the_displayed_table_uses_the_models_labels_over_the_catalogs():
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": [1], "faiz": [1.0]})
    lineage = {
        "t01_a.faiz": ColumnInfo(name="faiz", label_tr="Konut Kredisi (TL, Akım, %)")
    }

    shown = webui.display_frame(
        frame, "t01_a", lineage, {"faiz": "Konut kredisi faiz oranı", "yok": "x"}
    )

    assert list(shown.columns) == ["Dönem", "Konut kredisi faiz oranı"]


def test_the_displayed_table_rounds_floats_but_the_session_keeps_every_digit():
    """1716481.7634920634 in a cell is noise. Display only: the eval and the
    next tool read the frame, and 19.998 must not become 20.0 there."""
    frame = pd.DataFrame({"period": [1], "reel": [1716481.7634920634], "n": [3]})

    shown = webui.display_frame(frame, "t01_a", {})

    assert shown["reel"].tolist() == [1716481.7635]
    assert shown["n"].tolist() == [3]  # integers untouched
    assert frame["reel"].tolist() == [1716481.7634920634]


def test_the_models_label_reaches_the_derived_columns_of_the_displayed_table():
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": [1], "TP_KTF12": [1.0], "TP_KTF12_diff": [0.5]})
    lineage = {
        "t01_a.TP_KTF12": ColumnInfo(
            name="TP_KTF12", label_tr="Konut Kredisi (TL, Akım, %)"
        ),
        "t01_a.TP_KTF12_diff": ColumnInfo(
            name="TP_KTF12_diff", op="diff", inputs=["TP_KTF12"]
        ),
    }

    shown = webui.display_frame(frame, "t01_a", lineage, {"TP_KTF12": "Faiz oranı"})

    assert list(shown.columns) == ["Dönem", "Faiz oranı", "Faiz oranı — fark"]


def test_two_columns_with_the_same_label_keep_their_own_names():
    """A rename that collapses two headers into one loses a column."""
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"a": [1], "b": [2]})
    lineage = {
        "t01_a.a": ColumnInfo(name="a", label_tr="Ayni"),
        "t01_a.b": ColumnInfo(name="b", label_tr="Ayni"),
    }

    assert list(webui.display_frame(frame, "t01_a", lineage).columns) == ["a", "b"]


# --- text helpers -------------------------------------------------------------


def test_the_step_line_is_keyed_on_node_and_shows_the_first_summary_line():
    """`step_end` carries no tool name (docs/contracts.md section 6)."""
    line = webui.step_line(
        {"node": "act", "ok": True, "duration_ms": 120, "summary": "ilk\nikinci"}
    )
    assert line == "✓ act · 120 ms · ilk"
    assert webui.step_line({"node": "act", "ok": False}).startswith("✗ act")


def test_flags_and_sources_become_bullets_with_the_overflow_counted():
    flags = webui.flag_lines({"flags": [{"text": "a"}, {"text": "b"}]})
    assert flags == ["- a", "- b"]

    many = [{"series_id": f"evds:S{i}", "name": f"seri {i}"} for i in range(15)]
    lines = webui.source_lines({"sources": many})
    assert len(lines) == webui.SOURCES_SHOWN + 1
    assert lines[0] == "- `evds:S0` seri 0"
    assert lines[-1] == "- … ve 3 kaynak daha"

    url = webui.source_lines({"sources": [{"url": "https://x.y/z"}]})
    assert url == ["- `https://x.y/z`"]
