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

import pytest
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
    of thirteen because the values never reached it."""
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


@pytest.mark.parametrize("typed", ["", "\n"])
def test_an_empty_line_or_end_of_input_ends_the_session(typed):
    terminal, _ = _terminal(typed)

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
