"""`ui/render.py` -- pure, so tested without chainlit, a browser or a network.

Every event type the Chainlit adapter draws gets one normal-payload test and
one missing/empty-payload test: the interface must not crash on an event
shape it has not seen a full example of yet.
"""

from __future__ import annotations

from lakehouse_agent.ui.render import (
    render_approval_card,
    render_approval_decision,
    render_chart,
    render_error,
    render_sources,
    render_step_end,
    render_step_start,
    render_table_note,
    render_verification,
)

# --- step_start / step_end --------------------------------------------------


def test_step_start_names_the_tool_and_its_arguments():
    view = render_step_start(
        {
            "step_no": 3,
            "node": "act",
            "tool": "get_series",
            "args": {"series_ids": ["a"]},
        }
    )
    assert view["label"] == "get_series"
    assert "series_ids" in view["input_text"]


def test_step_start_with_no_tool_names_the_node():
    assert render_step_start({"node": "plan", "tool": None})["label"] == "Planlama"
    assert render_step_start({"node": "observe"})["label"] == "Degerlendirme"
    assert render_step_start({"node": "verify"})["label"] == "verify"


def test_step_start_on_an_empty_payload_does_not_crash():
    view = render_step_start(None)
    assert view["label"]
    assert view["input_text"] == ""


def test_step_end_reports_success_and_duration():
    view = render_step_end(
        {"ok": True, "summary": "3 satir\nikinci satir", "duration_ms": 120}
    )
    assert view["output_text"] == "3 satir"
    assert view["duration_ms"] == 120
    assert view["is_error"] is False
    assert view["error_text"] == ""


def test_step_end_reports_failure():
    view = render_step_end({"ok": False, "summary": "hata oldu", "error": "timeout"})
    assert view["is_error"] is True
    assert view["error_text"] == "timeout"


def test_step_end_on_an_empty_payload_does_not_crash():
    view = render_step_end({})
    assert view["output_text"] == ""
    assert view["duration_ms"] == 0
    assert view["is_error"] is False


# --- table_update ------------------------------------------------------------


def test_table_note_lists_new_columns():
    note = render_table_note(
        {"table_ref": "t01_get_series", "new_columns": ["yoy", "mom"]}
    )
    assert "t01_get_series" in note
    assert "yoy" in note and "mom" in note


def test_table_note_with_no_new_columns_still_names_the_table():
    assert "t01_get_series" in render_table_note(
        {"table_ref": "t01_get_series", "new_columns": []}
    )


def test_table_note_for_a_removed_table():
    note = render_table_note(
        {"table_ref": "t01_get_series", "removed": True, "reason": "quality_cancel"}
    )
    assert "kaldirildi" in note
    assert "quality_cancel" in note


def test_table_note_on_an_empty_payload_does_not_crash():
    assert render_table_note(None) == "tablo  kuruldu"


# --- sources -------------------------------------------------------------


def test_sources_lists_series_id_and_name():
    block = render_sources(
        {
            "sources": [
                {
                    "series_id": "evds:TP.KTF12",
                    "name": "Konut kredisi faizi",
                    "fetched_at": "2026-09-23T10:00:00Z",
                    "metadata_url": "https://evds2.tcmb.gov.tr/...",
                    "note": "aylik ortalama",
                }
            ]
        }
    )
    assert block is not None
    assert "evds:TP.KTF12" in block
    assert "Konut kredisi faizi" in block
    assert "metaveri" in block
    assert "aylik ortalama" in block


def test_sources_with_an_empty_list_is_none():
    assert render_sources({"sources": []}) is None


def test_sources_on_an_empty_payload_is_none():
    assert render_sources(None) is None


def test_sources_skips_a_malformed_entry_instead_of_crashing():
    assert render_sources({"sources": ["not-a-dict"]}) == "**Kaynaklar**"


# --- verification --------------------------------------------------------


def test_verification_flags_are_shown():
    block = render_verification(
        {"passed": False, "flags": [{"kind": "number_not_found", "text": "'9'"}]}
    )
    assert block is not None
    assert "'9'" in block


def test_a_clean_verification_renders_nothing():
    assert render_verification({"passed": True, "flags": []}) is None


def test_verification_on_an_empty_payload_renders_nothing():
    assert render_verification(None) is None


# --- chart -----------------------------------------------------------------


def test_a_valid_plotly_chart_is_kept():
    chart = render_chart(
        {
            "chart_ref": "c01",
            "title": "Faiz",
            "plotly_json": {
                "data": [{"type": "scatter", "y": [1, 2, 3]}],
                "layout": {},
            },
        }
    )
    assert chart is not None
    assert chart["chart_ref"] == "c01"
    assert chart["title"] == "Faiz"


def test_a_chart_with_no_plotly_json_is_skipped():
    assert render_chart({"chart_ref": "c01"}) is None


def test_a_chart_with_empty_data_is_skipped():
    assert render_chart({"plotly_json": {"data": []}}) is None


def test_a_chart_with_malformed_plotly_json_is_skipped():
    assert render_chart({"plotly_json": {"data": "not-a-list"}}) is None
    assert render_chart({"plotly_json": "not-a-dict"}) is None


def test_chart_on_an_empty_payload_is_skipped():
    assert render_chart(None) is None


# --- approvals ---------------------------------------------------------------


def test_approval_card_marks_the_default_option():
    card = render_approval_card(
        {
            "question": "Devam edelim mi?",
            "kind": "fetch",
            "options": [
                {"id": "approve", "label": "Onayla"},
                {"id": "cancel", "label": "Iptal"},
            ],
            "default": "approve",
        }
    )
    assert card["question"] == "Devam edelim mi?"
    options = {option["id"]: option for option in card["options"]}
    assert options["approve"]["is_default"] is True
    assert options["cancel"]["is_default"] is False


def test_approval_card_on_an_empty_payload_does_not_crash():
    card = render_approval_card(None)
    assert card["question"] == ""
    assert card["options"] == []


def test_approval_decision_line():
    line = render_approval_decision(
        {"kind": "fetch", "choice": "approve", "automatic": False}
    )
    assert "fetch" in line and "approve" in line and "secildi" in line

    automatic = render_approval_decision(
        {"kind": "fetch", "choice": "approve", "automatic": True}
    )
    assert "otomatik" in automatic


# --- errors -------------------------------------------------------------


def test_a_known_error_kind_gets_a_short_turkish_message():
    view = render_error({"kind": "LlmTimeoutError", "message": "socket read timed out"})
    assert "zaman asimi" in view["user_message"].lower()
    assert "stack" not in view["user_message"].lower()
    assert "socket read timed out" in view["log_message"]


def test_an_unknown_error_kind_gets_a_generic_message():
    view = render_error({"kind": "SomeNewException", "message": "detail"})
    assert view["user_message"] == "Bir hata olustu. Lutfen tekrar deneyin."
    assert "detail" in view["log_message"]


def test_error_on_an_empty_payload_does_not_crash():
    view = render_error(None)
    assert view["user_message"]
    assert view["log_message"] == ""
