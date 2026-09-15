"""What the model reads instead of the table.

Everything here serves one rule: the model gets a description, never the data.
Context stays small, and a model that never holds the digits cannot copy them
into an answer with a decimal place slipped.
"""

import pandas as pd

from lakehouse_agent.agent.formatting import (
    TOOL_MESSAGE_MAX_CHARS,
    build_tables_summary,
    describe_table,
    format_tool_message,
    preview_rows,
)
from lakehouse_agent.contracts import QualityReport, ToolResult


def frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": pd.date_range("2021-01-01", periods=rows, freq="MS"),
            "TP_KTF12": [10.0 + i * 0.5 for i in range(rows)],
        }
    )


def test_a_table_is_described_by_shape_not_by_content():
    line = describe_table("t01_get_series", frame())

    assert line == (
        "t01_get_series: 60 satir, 2021-01 - 2025-12, sutunlar: period, TP_KTF12"
    )
    assert "10.0" not in line


def test_an_unreadable_period_column_does_not_cost_the_step():
    """This runs after the tool has already succeeded and produced a table.

    An exception here would throw away work that was done, and it would do it
    outside the try block that catches tool failures, so the turn would die
    with the model's tool call left unanswered.
    """
    broken = pd.DataFrame({"period": ["bu bir tarih degil"], "v": [1.0]})
    line = describe_table("t01_x", broken)

    assert "donem araligi okunamadi" in line
    assert "sutunlar: period, v" in line


def test_the_summary_marks_which_table_is_active():
    """Without this the model cannot tell that a follow-up question can build on
    the table it made two turns ago, and it starts over from the catalog."""
    summary = build_tables_summary({"t01_a": frame(2), "t02_b": frame(2)}, "t02_b")

    assert "t01_a" in summary
    assert summary.count("(aktif)") == 1
    assert "(aktif)" in [line for line in summary.splitlines() if "t02_b" in line][0]


def test_an_empty_session_says_so_plainly():
    assert "tablo yok" in build_tables_summary({}, None)


def test_the_preview_shows_both_ends_of_the_range():
    """Both ends, not just the head: the model is usually asked about a range
    and the last period is as informative as the first."""
    rows = preview_rows(frame(60))

    assert len(rows) == 10
    assert rows[0]["period"] == "2021-01-01"
    assert rows[-1]["period"] == "2025-12-01"


def test_a_short_table_is_shown_whole_without_duplicating_rows():
    rows = preview_rows(frame(4))
    assert [row["period"] for row in rows] == [
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
        "2021-04-01",
    ]


def test_a_failed_result_is_marked_so_the_model_cannot_miss_it():
    message = format_tool_message(ToolResult(ok=False, summary="Sutun bulunamadi."))
    assert message.startswith("[HATA]")


def test_quality_problems_travel_with_the_table():
    """A unit nobody could determine changes what every number in the answer
    means, so it goes in the message rather than staying in a report."""
    message = format_tool_message(
        ToolResult(
            ok=True,
            summary="Tablo hazir.",
            quality=QualityReport(
                rows=60, missing_periods=2, duplicates=3, unit_guess=None
            ),
        )
    )

    assert "2 eksik donem" in message
    assert "3 tekrarli donem" in message
    assert "birim bilinmiyor" in message


def test_a_chatty_tool_cannot_crowd_out_the_conversation():
    """Capped at 2000 characters, as written in docs/contracts.md §4.

    One tool returning a huge preview would otherwise push the question itself
    out of the model's attention. Long output is a sign the tool should
    summarise better, not a reason to widen the cap.
    """
    message = format_tool_message(
        ToolResult(
            ok=True,
            summary="Tamam.",
            preview=[{"period": f"2021-{i:02d}", "v": "x" * 200} for i in range(30)],
        )
    )

    assert len(message) <= TOOL_MESSAGE_MAX_CHARS
    assert message.endswith("... (kisaltildi)")
