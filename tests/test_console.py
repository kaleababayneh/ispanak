"""The terminal interface (agent/console.py).

No terminal and no network: the console writes to a `StringIO` and reads from
one, which is what `rich.Console(file=..., stream=...)` is for. What is tested
is the three seams the loop uses -- draw an event, ask a question, answer an
approval card -- not how pretty the frame is.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import StringIO

from rich.console import Console

from lakehouse_agent.agent.console import Terminal
from lakehouse_agent.agent.events import Event
from lakehouse_agent.agent.state import ApprovalOption, ApprovalRequest


def _terminal(typed: str = "") -> tuple[Terminal, StringIO]:
    screen = StringIO()
    console = Console(file=screen, width=100, force_terminal=False, no_color=True)
    return Terminal(console=console, stream=StringIO(typed)), screen


def _event(type_: str, **payload) -> Event:
    return Event(
        type=type_,
        ts=datetime(2026, 9, 23, tzinfo=UTC),
        session_id="s",
        payload=payload,
    )


def _drawn(type_: str, **payload) -> str:
    terminal, screen = _terminal()
    asyncio.run(terminal.emit(_event(type_, **payload)))
    return screen.getvalue()


# --- the screen ------------------------------------------------------------


def test_a_table_arrives_with_its_values():
    """The headless recap prints a one-line description; on a screen the
    numbers are the point. Measured 23 Sep: an answer reasoned about one month
    of thirteen because the values never reached it.

    Drawn straight away here because no table store is wired: the event's own
    preview is all there will ever be."""
    drawn = _drawn(
        "table_update",
        table_ref="t01_get_series",
        columns=["period", "TP_KTF12"],
        rows=[{"period": "2026-06-01", "TP_KTF12": 40.335}],
        new_columns=[],
    )

    assert "t01_get_series" in drawn
    assert "TP_KTF12" in drawn
    assert "40.335" in drawn


def test_an_answer_streams_without_a_newline_per_piece():
    terminal, screen = _terminal()

    for piece in ("Faiz ", "yukseldi."):
        asyncio.run(terminal.emit(_event("answer_delta", text=piece)))

    assert "Faiz yukseldi." in screen.getvalue()


def test_model_text_is_never_read_as_markup():
    """`rich` eats `[...]`. An answer citing a bracket would lose it, and a
    page could colour text we did not write."""
    drawn = _drawn("answer_delta", text="kaynak [evds:TP.KTF12] alindi")

    assert "[evds:TP.KTF12]" in drawn


def test_flags_are_drawn_and_a_clean_verification_is_not():
    flagged = _drawn(
        "verification",
        passed=False,
        flags=[{"kind": "number_not_found", "text": "'9'"}],
    )
    clean = _drawn("verification", passed=True, flags=[])

    assert "Dogrulanamayan" in flagged and "'9'" in flagged
    assert clean.strip() == ""


def test_an_event_the_interface_does_not_know_is_ignored():
    """The event list grows. A demo must not die because the screen has not
    caught up with it."""
    terminal, screen = _terminal()

    asyncio.run(terminal.emit(_event("done", ok=True)))

    assert screen.getvalue() == ""


# --- the keyboard ----------------------------------------------------------


def test_a_question_is_read_and_stripped():
    terminal, _ = _terminal("  konut kredisi faizi  \n")

    assert asyncio.run(terminal.ask()) == "konut kredisi faizi"


def test_end_of_input_ends_the_session():
    """Ctrl-D, or a stream that ran out."""
    terminal, _ = _terminal("")

    assert asyncio.run(terminal.ask()) is None


def test_a_blank_line_with_nothing_written_ends_the_session_too():
    """Nothing collected and nothing more coming is the same as Ctrl-D."""
    terminal, _ = _terminal("\n")

    assert asyncio.run(terminal.ask()) is None


# --- the approval card -----------------------------------------------------


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        request_id="r1",
        kind="series_choice",
        question="Hangi seriyle devam edelim?",
        options=[
            ApprovalOption(
                id="evds:TP.KBK.TRY.18", label="Kar payi", detail="Kredi Kar Oranlari"
            ),
            ApprovalOption(
                id="evds:TP.KTF12", label="Faiz", detail="Kredi Faiz Oranlari"
            ),
        ],
        default="evds:TP.KBK.TRY.18",
        created_at=datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_the_card_is_drawn_with_numbered_options_and_their_detail():
    terminal, screen = _terminal("1\n")

    asyncio.run(terminal.approve(_request()))
    drawn = screen.getvalue()

    assert "1." in drawn and "2." in drawn
    assert "Kredi Faiz Oranlari" in drawn  # the detail that tells the twins apart
    assert "varsayilan" in drawn


def test_the_number_typed_picks_that_option():
    terminal, _ = _terminal("2\n")

    decision = asyncio.run(terminal.approve(_request()))

    assert decision.choice == "evds:TP.KTF12"
    assert decision.request_id == "r1"
    assert not decision.automatic  # a person answered


def test_an_empty_answer_takes_the_default():
    terminal, _ = _terminal("\n")

    assert asyncio.run(terminal.approve(_request())).choice == "evds:TP.KBK.TRY.18"


def test_a_bad_answer_is_asked_again_rather_than_guessed():
    terminal, screen = _terminal("sekiz\n9\n2\n")

    decision = asyncio.run(terminal.approve(_request()))

    assert decision.choice == "evds:TP.KTF12"
    assert screen.getvalue().count("Gecerli bir numara yaz.") == 2


# --- a pasted question is one question -------------------------------------


def test_a_pasted_paragraph_is_one_question_not_three():
    """Measured 23 Sep on the first chat run: the brief's own scenario is
    three lines, and pasting it ran three turns -- the second one on the
    fragment "gosteriniz. Buna ek olarak ...", which is not a question
    anybody asked."""
    terminal, _ = _terminal(
        "2021-2025 arasindaki 60 aylik veriden konut kredilerini\n"
        "aylik goster. Faiz oranlarini da ekle.\n"
        "Faizin dustugu donemlerde kredi nasil degismis?\n"
        "\n"
    )

    question = asyncio.run(terminal.ask())

    assert question is not None
    assert question.startswith("2021-2025")
    assert question.endswith("degismis?")
    assert "\n" not in question


def test_the_blank_line_ends_the_question_not_the_ones_after_it():
    """The paragraph ends where the person left a gap; what follows is the
    next question, not part of this one."""
    terminal, _ = _terminal("ilk soru satiri\nikinci satiri\n\nbambaska bir soru\n\n")

    assert asyncio.run(terminal.ask()) == "ilk soru satiri ikinci satiri"
    assert asyncio.run(terminal.ask()) == "bambaska bir soru"


def test_a_question_is_still_asked_when_the_input_ends_without_a_blank_line():
    """Ctrl-D on a half-written question sends it rather than losing it."""
    terminal, _ = _terminal("tek satirlik soru\n")

    assert asyncio.run(terminal.ask()) == "tek satirlik soru"


# --- the whole table, not a preview ----------------------------------------


def _frame(rows: int):
    import pandas as pd

    return pd.DataFrame(
        {
            "period": pd.period_range("2021-01", periods=rows, freq="M").to_timestamp(),
            "aylik_tuketici_tuketici_kredileri_konut_toplam_sektor": [
                1000.0 + n for n in range(rows)
            ],
        }
    )


def _drawn_table(frame, **payload) -> str:
    screen = StringIO()
    terminal = Terminal(
        console=Console(file=screen, width=110, no_color=True),
        tables={"t01_get_series": frame}.get,
    )
    asyncio.run(
        terminal.emit(
            _event(
                "table_update",
                table_ref="t01_get_series",
                columns=list(frame.columns),
                rows=[],  # the event's preview is deliberately empty here
                new_columns=[],
                **payload,
            )
        )
    )
    # The table is drawn when the turn ends, not after every column added to
    # it: a turn adding four columns to sixty months printed it five times.
    asyncio.run(terminal.emit(_event("done", truncated_reason=None)))
    return screen.getvalue()


def test_a_sixty_month_question_shows_sixty_rows():
    """The event carries ten rows, enough for the model. A person asking for
    sixty months and seeing ten reads it as missing data -- which is what the
    first real run looked like. `docs/contracts.md` section 4 keeps the whole
    frame in the session store so the interface can draw it."""
    drawn = _drawn_table(_frame(60))

    assert (
        drawn.count("│ 2021-")
        + drawn.count("│ 2022-")
        + drawn.count("│ 2023-")
        + drawn.count("│ 2024-")
        + drawn.count("│ 2025-")
        == 60
    )
    assert "gosteriliyor" not in drawn  # nothing was cut, so nothing is claimed


def test_a_huge_table_is_cut_and_says_so():
    """The ceiling is a guard against a tool that built forty thousand rows,
    and a cut that does not announce itself is the failure it replaces."""
    drawn = _drawn_table(_frame(500))

    assert "500 satirin ilk ve son 100'i gosteriliyor" in drawn


def test_a_long_column_name_is_folded_not_truncated():
    """It is the name the next question has to use, so it has to be readable
    in full. Drawn on a narrow screen on purpose: with room to spare nothing
    is cut either way, and the test would pass without the folding."""
    screen = StringIO()
    frame = _frame(2)
    terminal = Terminal(
        console=Console(file=screen, width=54, no_color=True),
        tables={"t01_get_series": frame}.get,
    )
    asyncio.run(
        terminal.emit(
            _event(
                "table_update",
                table_ref="t01_get_series",
                columns=list(frame.columns),
                rows=[],
                new_columns=[],
            )
        )
    )
    asyncio.run(terminal.emit(_event("done", truncated_reason=None)))
    drawn = screen.getvalue()

    assert "…" not in drawn
    # Every piece of the name survives, wrapped across lines.
    assert "toplam_sektor" in drawn.replace("\n", "").replace(" ", "")


def test_float_noise_never_reaches_the_screen():
    """Binary arithmetic turns a monthly difference into 0.4029999999999999,
    and a person cannot tell noise from precision."""
    import pandas as pd

    frame = pd.DataFrame({"period": ["2021-02-01"], "fark": [-0.40299999999999997]})

    drawn = _drawn_table(frame)

    assert "-0.403" in drawn
    assert "0.40299999" not in drawn


def test_without_a_table_store_the_preview_is_still_drawn():
    """An older caller, or a test, wires no store; the event's own rows are
    the fallback rather than an empty table."""
    drawn = _drawn(
        "table_update",
        table_ref="t01_get_series",
        columns=["period", "deger"],
        rows=[{"period": "2021-01-01", "deger": 1.0}],
        new_columns=[],
    )

    assert "2021-01-01" in drawn


def test_the_prompt_names_the_number_the_blank_line_picks():
    """ "bos=varsayilan" made a person look for the word rather than the
    option."""
    terminal, screen = _terminal("\n")

    asyncio.run(terminal.approve(_request()))

    assert "bos=1" in screen.getvalue()


def test_a_table_is_drawn_once_at_the_end_not_after_every_column():
    """Measured 23 Sep: a turn adding four columns to sixty months printed the
    same table five times, some three hundred lines, and the answer scrolled
    off the top. What a person wants is the table the turn ended with."""
    screen = StringIO()
    frame = _frame(3)
    terminal = Terminal(
        console=Console(file=screen, width=100, no_color=True),
        tables={"t01_get_series": frame}.get,
    )

    for fresh in (["period"], ["a_diff"], ["b_diff"]):
        asyncio.run(
            terminal.emit(
                _event(
                    "table_update",
                    table_ref="t01_get_series",
                    columns=list(frame.columns),
                    rows=[],
                    new_columns=fresh,
                )
            )
        )
    mid = screen.getvalue()
    asyncio.run(terminal.emit(_event("done", truncated_reason=None)))
    end = screen.getvalue()

    assert mid.count("2021-01-01") == 0  # nothing drawn while the turn runs
    assert "a_diff eklendi" in mid  # but the person is told what changed
    assert end.count("2021-01-01") == 1  # and the table lands once, at the end


def test_a_chart_is_announced_not_drawn():
    """A terminal has no plotly; it says the chart exists and where it shows,
    and counts the traces so a person can tell an empty figure from a full
    one without opening the browser."""
    figure = {"data": [{"name": "a"}, {"name": "b"}], "layout": {}}

    screen = _drawn("chart", chart_ref="c02", plotly_json=figure)

    assert "grafik c02 hazir (2 seri)" in screen
    assert "tarayici" in screen
    assert _drawn("chart", chart_ref="c03").count("c03") == 1  # no figure: no crash


def test_a_table_header_shows_the_label_over_the_name_when_lineage_knows_it():
    """The name stays because the next question has to use it; the label
    goes above it because a person is reading."""
    import pandas as pd

    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": ["2026-06-01"], "TP_KTF12": [40.3]})
    lineage = {
        "t01_get_series.TP_KTF12": ColumnInfo(
            name="TP_KTF12", label_tr="Faiz", unit="%"
        )
    }
    screen = StringIO()
    terminal = Terminal(
        console=Console(file=screen, width=110, no_color=True),
        tables={"t01_get_series": frame}.get,
        lineage=lineage,
    )
    asyncio.run(
        terminal.emit(
            _event("table_update", table_ref="t01_get_series", columns=[], rows=[])
        )
    )
    asyncio.run(terminal.emit(_event("done")))
    drawn = screen.getvalue()

    assert "Faiz (%)" in drawn
    assert "TP_KTF12" in drawn
    assert "Dönem" in drawn


def test_the_table_header_says_what_the_legend_said():
    """A person who read "Konut kredisi faiz oranı" on the chart must not
    meet the catalog's technical name in the table under it."""
    import pandas as pd

    from lakehouse_agent.agent.state import ChartSeries, ChartSpec
    from lakehouse_agent.contracts import ColumnInfo

    frame = pd.DataFrame({"period": ["2026-06-01"], "TP_KTF12": [40.3]})
    lineage = {
        "t01_get_series.TP_KTF12": ColumnInfo(
            name="TP_KTF12", label_tr="Konut Kredisi (TL, Akım, %)"
        )
    }
    charts = [
        ChartSpec(
            chart_ref="c01",
            table_ref="t01_get_series",
            series=[ChartSeries(column="TP_KTF12", label="Konut kredisi faiz oranı")],
        )
    ]
    screen = StringIO()
    terminal = Terminal(
        console=Console(file=screen, width=110, no_color=True),
        tables={"t01_get_series": frame}.get,
        lineage=lineage,
        charts=charts,
    )
    asyncio.run(
        terminal.emit(
            _event("table_update", table_ref="t01_get_series", columns=[], rows=[])
        )
    )
    asyncio.run(terminal.emit(_event("done")))
    drawn = screen.getvalue()

    assert "Konut kredisi faiz oranı" in drawn
    assert "Konut Kredisi (TL, Akım, %)" not in drawn
    assert "TP_KTF12" in drawn


def test_a_follow_up_that_only_reads_the_table_does_not_draw_it_again():
    """`top` fires a `table_update` with no new columns. The first turn
    draws the table; the second must not print sixty rows again."""
    import pandas as pd

    frame = pd.DataFrame({"period": ["2026-06-01"], "TP_KTF12": [40.3]})
    screen = StringIO()
    terminal = Terminal(
        console=Console(file=screen, width=110, no_color=True),
        tables={"t01_get_series": frame}.get,
    )
    update = dict(table_ref="t01_get_series", columns=["period", "TP_KTF12"], rows=[])

    asyncio.run(
        terminal.emit(_event("table_update", new_columns=["TP_KTF12"], **update))
    )
    asyncio.run(terminal.emit(_event("done")))
    first = screen.getvalue()
    asyncio.run(terminal.emit(_event("table_update", new_columns=[], **update)))
    asyncio.run(terminal.emit(_event("done")))
    second = screen.getvalue()[len(first) :]

    assert "40.3" in first and "kuruldu" in first
    assert "40.3" not in second and "tablo" not in second
