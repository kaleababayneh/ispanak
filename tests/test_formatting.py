"""What the model reads instead of the table.

Everything here serves one rule: the model gets a description, never the data.
Context stays small, and a model that never holds the digits cannot copy them
into an answer with a decimal place slipped.
"""

import datetime as dt
import json
from decimal import Decimal

import numpy as np
import pandas as pd

from lakehouse_agent.agent.formatting import (
    TOOL_MESSAGE_MAX_CHARS,
    build_tables_summary,
    describe_table,
    format_sources,
    format_tool_message,
    json_safe,
    preview_rows,
)
from lakehouse_agent.contracts import QualityReport, Source, ToolResult

FETCHED_AT = pd.Timestamp("2026-09-15T06:00:00Z").to_pydatetime()


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
    """Capped at 2000 characters, as written in docs/contracts.md section 4.

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


def test_what_gets_dropped_is_the_middle_of_the_preview_not_its_end():
    """The preview is assembled last, so slicing the finished message always
    deleted the newest rows first -- and silently.

    `preview_rows` keeps both ends on purpose: the last period is as
    informative as the first. Cutting from the middle keeps that true, and the
    label says how many rows went.
    """
    rows = [{"period": f"2021-{i + 1:02d}-01", "v": "x" * 200} for i in range(30)]
    message = format_tool_message(ToolResult(ok=True, summary="Tamam.", preview=rows))

    assert len(message) <= TOOL_MESSAGE_MAX_CHARS
    assert "2021-01-01" in message  # first row survives
    assert "2021-30-01" in message  # and so does the last
    assert "satir atlandi" in message
    assert not message.endswith("(kisaltildi)")


def test_a_preview_that_cannot_fit_at_all_says_so_instead_of_half_a_row():
    """Truncating JSON mid-object hands the model something it cannot read."""
    message = format_tool_message(
        ToolResult(ok=True, summary="Tamam.", preview=[{"v": "x" * 4000}])
    )

    assert "sigmadi" in message
    assert "sutunlar: v" in message


def test_the_preview_survives_json_dumps_for_every_column_gold_can_hand_us():
    """The claim in the docstring, actually checked.

    Only `datetime64` columns were normalised, and gold does not use one: the
    contract stores `period` as `datetime.date`, which lands in an object
    column no dtype check looks at. So the preview of a real gold table could
    not be serialised, and the failure would surface on the event channel in
    the middle of a turn rather than here.
    """
    frame = pd.DataFrame(
        {
            "period": [dt.date(2021, 1, 1), dt.date(2021, 2, 1)],  # gold's dtype
            "stamp": pd.date_range("2021-01-01", periods=2, freq="MS"),
            "span": pd.to_timedelta([1, 2], unit="D"),
            "quarter": pd.period_range("2021Q1", periods=2, freq="Q"),
            "money": [Decimal("1.5"), Decimal("2.5")],
            "value": [1.0, float("nan")],
        }
    )

    rows = preview_rows(frame)
    json.dumps(rows)  # the assertion: this used to raise TypeError

    assert rows[0]["period"] == "2021-01-01"
    assert rows[0]["stamp"] == "2021-01-01"
    assert rows[0]["quarter"] == "2021Q1"
    assert rows[0]["money"] == 1.5
    assert "1 days" in rows[0]["span"]


def test_a_missing_value_becomes_null_not_the_word_nan():
    """`json.dumps` writes NaN as a bare `NaN`, which is not JSON.

    `JSON.parse` on the interface side rejects it outright, so the event that
    carried it is lost. Missing stays missing -- it becomes `null`; nothing is
    filled in (CLAUDE.md #2).
    """
    frame = pd.DataFrame({"value": [1.0, float("nan"), float("inf")]})
    rows = preview_rows(frame)

    assert rows[1]["value"] is None
    assert rows[2]["value"] is None
    assert "NaN" not in json.dumps(rows)
    assert json.loads(json.dumps(rows))[1]["value"] is None


def test_a_timestamp_carrying_a_time_does_not_lose_it():
    """Dates print short, because every series here is monthly or coarser.

    An intraday stamp is another matter: silently dropping the time would make
    two different rows read as the same period.
    """
    frame = pd.DataFrame({"period": [pd.Timestamp("2021-01-01T14:30:00")]})
    assert preview_rows(frame)[0]["period"] == "2021-01-01T14:30:00"


def test_an_integer_period_column_is_not_reported_as_1970():
    """`to_datetime` reads plain integers as nanoseconds since the epoch.

    A column of 1, 2, 3 was described as "1970-01 - 1970-01" -- a confident,
    wrong answer in the one line the model reads about the table. Refusing to
    read it is the honest output.
    """
    line = describe_table("t01_x", pd.DataFrame({"period": [1, 2, 3], "v": [1.0] * 3}))

    assert "donem araligi okunamadi" in line
    assert "1970" not in line


def test_an_integer_period_column_is_caught_whatever_its_dtype():
    """The guard asks the values, not the dtype.

    The first version keyed on `is_numeric_dtype`, which sees `int64` and not
    an object column holding the same integers -- so the fix missed exactly the
    column shape that hides from dtype checks, which is the reason `json_safe`
    in the same file converts per value.
    """
    for column in (pd.Series([1, 2, 3]), pd.Series([1, 2, 3], dtype=object)):
        line = describe_table("t01_x", pd.DataFrame({"period": column}))
        assert "donem araligi okunamadi" in line
        assert "1970" not in line

    dates = pd.DataFrame({"period": [dt.date(2021, 1, 1), None, dt.date(2021, 3, 1)]})
    assert "2021-01 - 2021-03" in describe_table("t01_x", dates)


def test_a_non_finite_decimal_does_not_slip_past_as_infinity():
    """`is_nan` let `Decimal("Infinity")` through, and json.dumps writes that
    as a bare `Infinity` -- the same invalid JSON `NaN` produces."""
    rows = preview_rows(pd.DataFrame({"v": [Decimal("Infinity"), Decimal("NaN")]}))

    assert rows == [{"v": None}, {"v": None}]
    assert json.loads(json.dumps(rows)) == rows


def test_pandas_na_becomes_null_rather_than_the_text_na():
    assert json_safe(pd.NA) is None
    assert json_safe({"a": pd.NA}) == {"a": None}


def test_the_model_reads_preview_rows_as_json_not_as_a_python_repr():
    """This is the only place a preview reaches the model, and every tool
    reaches it -- including ones that did not build their preview here.

    Left as a repr the model is handed `datetime.date(2020, 9, 1)` and `nan`
    where a date and a null belong.
    """
    message = format_tool_message(
        ToolResult(
            ok=True,
            summary="Tablo hazir.",
            preview=[{"dönem": dt.date(2020, 9, 1), "değer": float("nan")}],
        )
    )

    assert '"dönem": "2020-09-01"' in message
    assert '"değer": null' in message
    assert "datetime.date" not in message
    assert "nan" not in message


def test_a_warning_travels_all_the_way_into_the_model_message():
    """The only mechanism CLAUDE.md #3 has.

    Units are never converted in the lake; conversion happens in the tool layer
    and is reported in `warnings`. A warning that stops here becomes an answer
    that states a converted number as if it were the original.
    """
    message = format_tool_message(
        ToolResult(
            ok=True,
            summary="Tablo hazir.",
            warnings=["Birim bin TL'den milyon TL'ye cevrildi.", "Kumulatif seri."],
        ),
        table_note="t02_deflate: 48 satir",
    )

    assert "t02_deflate: 48 satir" in message
    assert "bin TL'den milyon TL'ye" in message
    assert "Kumulatif seri." in message


def test_sources_are_listed_so_citations_are_copied_not_invented():
    sources = [
        Source(series_id="evds:TP.KTF12", name="Konut kredisi", fetched_at=FETCHED_AT),
        Source(url="https://www.bddk.org.tr/BultenAylik", fetched_at=FETCHED_AT),
    ]
    listed = format_sources(sources)

    assert "evds:TP.KTF12 (Konut kredisi)" in listed
    assert "https://www.bddk.org.tr/BultenAylik" in listed
    assert format_sources([]) == "Kaynak yok."


def test_a_nested_or_numpy_cell_is_converted_too():
    """A cell is not always a scalar: a tool may put a list or a small mapping
    in one, and numpy hands back its own scalar types."""
    assert json_safe([np.int64(3), np.float64(1.5), np.bool_(True)]) == [3, 1.5, True]
    assert json_safe({"a": (dt.date(2021, 1, 1), np.float64("nan"))}) == {
        "a": ["2021-01-01", None]
    }
    assert json_safe(np.datetime64("2021-01-01")) == "2021-01-01"
    assert json_safe(object()).startswith("<object object")


def test_a_duplicate_column_label_is_kept_rather_than_eaten():
    """`to_dict` collapses duplicate labels to the last one and warns on
    stderr, which nothing on the model or event path reads.

    `describe_table` still lists the label twice, so the model would be told
    the table has two such columns and handed one value for them.
    """
    frame = pd.DataFrame([[1.0, 2.0]], columns=["value", "value"])
    rows = preview_rows(frame)

    assert rows == [{"value": 1.0, "value__2": 2.0}]
    json.dumps(rows)


def test_a_nanosecond_numpy_timestamp_does_not_arrive_as_a_19_digit_number():
    """`.item()` on `np.datetime64[ns]` has no lossless Python target and
    returns a bare integer. Day resolution converts cleanly, which is why the
    only case the suite covered stayed green."""
    assert json_safe(np.datetime64("2021-01-01T12:00:00", "ns")) == (
        "2021-01-01T12:00:00"
    )
    assert json_safe(np.datetime64("2021-01-01", "D")) == "2021-01-01"
    assert json_safe(np.timedelta64(1, "D")) == "1 days 00:00:00"
    assert json_safe(np.datetime64("NaT", "ns")) is None


def test_an_array_in_a_cell_falls_back_to_text_instead_of_raising():
    """A groupby/apply result is the ordinary way to get one. The branch was
    marked defensive; it is reached."""
    assert json_safe(np.array([1, 2])) == "[1 2]"


def test_one_number_among_the_dates_is_enough_to_refuse_the_range():
    """The guard asked whether every value was a number. One integer is enough
    for `to_datetime` to place that row in 1970 and drag the range with it."""
    mixed = pd.DataFrame({"period": [dt.date(2021, 1, 1), 2, dt.date(2021, 3, 1)]})

    line = describe_table("t01_x", mixed)
    assert "donem araligi okunamadi" in line
    assert "1970" not in line


def test_the_cap_still_holds_when_it_is_not_the_preview_that_is_long():
    """Fitting the preview cannot save a message whose other parts already
    overflow, so the blunt slice stays as the last line of defence."""
    message = format_tool_message(
        ToolResult(ok=True, summary="Tamam.", warnings=["u" * 500] * 6)
    )

    assert len(message) <= TOOL_MESSAGE_MAX_CHARS
    assert message.endswith("... (kisaltildi)")
