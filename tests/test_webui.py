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
    one_row = pd.DataFrame({"period": [1], "v": [18.1775]})
    assert webui.default_chart(one_row, "t02_a", []) is None  # a number, not a line

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


def test_the_step_line_names_the_tool_the_caller_saw_at_step_start():
    """`step_end` carries no tool name (docs/contracts.md section 6); a
    screen of "act · 1707 ms" lines told nobody which tool ran."""
    payload = {"node": "act", "ok": True, "duration_ms": 120, "summary": "ilk\nikinci"}

    assert webui.step_line(payload, "get_series") == "✓ get_series · 120 ms"
    assert webui.step_line(payload) == "✓ act · 120 ms"  # no tool known: the node
    assert webui.step_line({"node": "act", "ok": False}, "read_url").startswith(
        "✗ read_url"
    )


def test_an_analysis_table_is_titled_by_its_analysis_and_the_series_analysed():
    """`t02_detect_changes` meant nothing on screen (seen 23 Sep)."""
    from lakehouse_agent.agent.state import ChartSeries
    from lakehouse_agent.contracts import ColumnInfo

    lineage = {
        "t01_get_series.TP_KTF12": ColumnInfo(
            name="TP_KTF12", label_tr="Konut Kredisi (TL, Akım, %)"
        ),
        "t01_get_series.TP_KFE_TR": ColumnInfo(
            name="TP_KFE_TR", label_tr="Konut Fiyat Endeksi (KFE)"
        ),
        "t02_detect_changes.tarih": ColumnInfo(
            name="tarih", inputs=["TP_KTF12"], tool="detect_changes"
        ),
        "t02_detect_changes.sicrama_yuzde": ColumnInfo(
            name="sicrama_yuzde", inputs=["TP_KTF12"]
        ),
        "t03_causality.yon": ColumnInfo(name="yon", inputs=["TP_KTF12", "TP_KFE_TR"]),
    }

    # catalog name when the model never labelled the column
    assert (
        webui.table_title("t02_detect_changes", ["tarih", "sicrama_yuzde"], lineage)
        == "Kırılma analizi — Konut Kredisi (TL, Akım, %)"
    )
    # the model's own word wins, from any chart in the session
    charts = [
        ChartSpec(
            chart_ref="c01",
            table_ref="t01_get_series",
            series=[ChartSeries(column="TP_KTF12", label="Konut kredisi faiz oranı")],
        )
    ]
    assert (
        webui.table_title(
            "t02_detect_changes", ["tarih", "sicrama_yuzde"], lineage, charts
        )
        == "Kırılma analizi — Konut kredisi faiz oranı"
    )
    assert (
        webui.table_title("t03_causality", ["yon"], lineage, charts)
        == "Granger öncüllük testi — Konut kredisi faiz oranı ve Konut Fiyat Endeksi (KFE)"
    )


def test_a_series_table_is_titled_by_its_columns_and_unknown_tables_by_their_suffix():
    from lakehouse_agent.contracts import ColumnInfo

    lineage = {
        "t01_get_series.TP_KTF12": ColumnInfo(
            name="TP_KTF12", label_tr="Faiz", unit="%"
        ),
        "t04_read_url.a": ColumnInfo(name="a", source_url="https://x.y/z.pdf"),
    }

    assert (
        webui.table_title("t01_get_series", ["period", "TP_KTF12"], lineage)
        == "Seri tablosu — Faiz"
    )
    assert (
        webui.table_title("t04_read_url", ["a", "b"], lineage)
        == "Adresten okunan tablo — z.pdf"
    )
    assert webui.table_title("t05_lag_corr", ["gecikme"], {}) == "Gecikmeli korelasyon"
    assert webui.table_title("t06_yeni_arac", ["x"], {}) == "yeni arac"


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


# --- carried over from feat/chainlit-ui (Burak) ------------------------------


def test_bare_series_ids_are_fenced_for_the_markdown_renderer():
    """Chainlit's markdown reads `:TP` as a directive and drops it."""
    assert (
        webui.protect_series_ids("Kaynak evds:TP.KTF12 idi.")
        == "Kaynak `evds:TP.KTF12` idi."
    )
    assert (
        webui.protect_series_ids("bddk:aylik:tuketici:konut:TL:sektor ve")
        == "`bddk:aylik:tuketici:konut:TL:sektor` ve"
    )
    assert (
        webui.protect_series_ids("zaten `evds:TP.KFE.TR` var")
        == "zaten `evds:TP.KFE.TR` var"
    )
    assert (
        webui.protect_series_ids("evds:TP.KTF12 ile evds:TP.KFE.TR")
        == "`evds:TP.KTF12` ile `evds:TP.KFE.TR`"
    )
    assert webui.protect_series_ids("hic kimlik yok") == "hic kimlik yok"
    assert webui.protect_series_ids("") == ""


def test_a_known_error_kind_gets_a_short_message_and_the_log_gets_the_detail():
    user, log = webui.error_message(
        {"kind": "LlmTimeoutError", "message": "Traceback ..."}
    )
    assert user == webui.ERROR_MESSAGES["LlmTimeoutError"]
    assert "Traceback" in log and "Traceback" not in user

    user, log = webui.error_message({"kind": "ValueError", "message": "x"})
    assert user == webui.GENERIC_ERROR and log == "ValueError: x"
    assert webui.error_message(None) == (webui.GENERIC_ERROR, "")


def test_a_removed_table_is_announced_not_ignored():
    assert webui.removed_note(
        {"table_ref": "t02_a", "removed": True, "reason": "quality_cancel"}
    ) == ("Tablo t02_a kaldırıldı (quality_cancel).")
    assert (
        webui.removed_note({"table_ref": "t02_a", "removed": True})
        == "Tablo t02_a kaldırıldı."
    )
    assert webui.removed_note({"table_ref": "t02_a", "new_columns": ["x"]}) is None
    assert webui.removed_note(None) is None


def test_sources_from_every_step_are_merged_once_each_in_order_of_first_sight():
    """Three fetches put three source blocks above the answer (seen 23 Sep)."""
    a = {"sources": [{"series_id": "evds:TP.KTF12", "name": "Faiz"}]}
    b = {
        "sources": [
            {"series_id": "evds:TP.KTF12", "name": "Faiz yine"},
            {"url": "https://x/y"},
        ]
    }
    c = {"sources": [{"url": "https://x/y"}, "bozuk", {"name": "kimliksiz"}]}

    merged = webui.merge_sources([a, b, None, c])

    assert merged == [
        {"series_id": "evds:TP.KTF12", "name": "Faiz"},
        {"url": "https://x/y"},
    ]
    assert webui.merge_sources([]) == []
    assert len(webui.source_lines({"sources": merged})) == 2


def test_a_series_table_title_lists_the_fetched_series_not_their_derivatives():
    """U2's title ran to three lines: nominal, real, rebased and percent
    change of one balance. The table holds one fetched series."""
    from lakehouse_agent.contracts import ColumnInfo

    lineage = {
        "t03_get_series.bak": ColumnInfo(
            name="bak", label_tr="Tüketici Kredileri - Konut (Toplam)", unit="milyon TL"
        ),
        "t03_get_series.reel_bak": ColumnInfo(
            name="reel_bak",
            op="deflate",
            inputs=["bak"],
            unit="milyon TL, 2021-01 fiyatlariyla",
        ),
        "t03_get_series.reel_bak_endeks_202101": ColumnInfo(
            name="reel_bak_endeks_202101",
            op="rebase",
            inputs=["reel_bak"],
            unit="endeks",
        ),
        "t03_get_series.reel_bak_degisim_202101": ColumnInfo(
            name="reel_bak_degisim_202101",
            op="change_since",
            inputs=["reel_bak"],
            unit="%",
        ),
    }
    columns = [
        "period",
        "bak",
        "reel_bak",
        "reel_bak_endeks_202101",
        "reel_bak_degisim_202101",
    ]

    assert webui.table_title("t03_get_series", columns, lineage) == (
        "Seri tablosu — Tüketici Kredileri - Konut (Toplam)"
    )
    # nothing fetched-looking at all (a fixture): every column, as before
    assert webui.table_title("t09_get_series", ["a", "b"], {}) == "Seri tablosu — a, b"


def test_a_dot_in_a_header_is_replaced_for_the_browser_but_not_in_the_frame():
    """Seen 23 Sep: every header with a dot rendered an empty column."""
    frame = pd.DataFrame(
        {
            "AY/MONTH": ["Ocak"],
            "Sayısı/ Trans.": [1017.0],
            "8.İşsizlik oranı (%)": [9.1],
        }
    )

    shown = webui.display_frame(frame, "t01_read_url", {})

    assert list(shown.columns) == [
        "AY/MONTH",
        "Sayısı/ Trans\u2024",
        "8\u2024İşsizlik oranı (%)",
    ]
    assert "." not in "".join(shown.columns)
    assert shown.iloc[0].tolist() == ["Ocak", 1017.0, 9.1]
    assert list(frame.columns) == ["AY/MONTH", "Sayısı/ Trans.", "8.İşsizlik oranı (%)"]
