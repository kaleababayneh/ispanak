"""When the agent stops and asks a person, and when it must not.

Both directions matter equally. KKB called human approval a differentiator, so
the gates have to fire on the cases a person genuinely decides better. But an
agent that asks at every step is unusable on stage, and the scenario is
supposed to run start to finish without a single interruption, so the tests
below pin the quiet cases just as tightly as the noisy ones.
"""

import asyncio

import pytest

from lakehouse_agent.agent import approvals
from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.agent.state import SessionState
from lakehouse_agent.contracts import QualityReport, ToolResult
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.testing.fake_tools import (
    IngestArgs,
    make_get_series,
    make_ingest,
)

FLOW_RATE = {
    "series_id": "evds:TP.KTF12",
    "name_tr": "Konut Kredisi Faiz Orani (Akim)",
    "unit": "%",
    "freq": "W",
    "kind": "rate",
    "score": 0.0163,
}
STOCK_RATE = {
    "series_id": "evds:TP.BKR.TRY.18",
    "name_tr": "Konut Kredisi Faiz Orani (Stok)",
    "unit": "%",
    "freq": "M",
    "kind": "rate",
    "score": 0.0161,
}


@pytest.fixture
def state() -> SessionState:
    return SessionState(session_id="s1", turn_no=1)


@pytest.fixture
def ctx(agent_settings) -> ToolContext:
    return ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM(),
        lake=None,
        settings=agent_settings,
    )


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def test_fetching_outside_data_asks_and_shows_what_it_will_do(state, ctx):
    """The card has to say what is about to happen, not just "proceed?".

    This is the one action with a cost outside our process: it hits a public
    service, takes real time and can fail mid-demo. "Approve?" with no detail
    is theatre; naming the source and the series is a decision.
    """
    spec = make_ingest()
    args = IngestArgs(source="evds", series_id="evds:TP.KTF12")

    request = approvals.needs_fetch_approval(spec, args, ctx, state)

    assert request is not None
    assert request.kind == "fetch"
    assert "EVDS" in request.question
    assert "evds:TP.KTF12" in request.question
    assert [option.id for option in request.options] == ["approve", "cancel"]
    assert request.default == "approve"


def test_a_tool_that_only_reads_the_pool_never_asks(state, ctx):
    """`get_series` reads data we already hold; stopping for it would be noise."""
    spec = make_get_series()
    args = spec.args_model.model_validate(
        {"series_ids": ["evds:TP.KTF12"], "start": "2021-01", "end": "2025-12"}
    )
    assert approvals.needs_fetch_approval(spec, args, ctx, state) is None


def test_a_broken_preview_still_lets_the_question_through(state, ctx):
    """Losing the detail is bad; skipping the gate because of it is worse."""
    spec = make_ingest()
    broken = type(spec)(
        **{
            **{field: getattr(spec, field) for field in spec.__dataclass_fields__},
            "approval_preview": lambda args, ctx: 1 / 0,
        }
    )
    request = approvals.needs_fetch_approval(
        broken, IngestArgs(source="evds", series_id="x"), ctx, state
    )
    assert request is not None
    assert request.kind == "fetch"


# --------------------------------------------------------------------------
# series choice
# --------------------------------------------------------------------------


def test_one_clear_match_runs_without_asking(state):
    """The scenario's own path. Three turns, zero interruptions."""
    result = ToolResult(ok=True, summary="1 aday", preview=[FLOW_RATE])
    assert approvals.needs_series_choice(result, state, 1) is None


def test_the_flow_stock_pair_stops_and_asks(state):
    """The trap the scenario is built around.

    "Housing loan interest rate" matches both the flow series and the stock
    series. They answer different questions, and choosing wrong produces a
    confident, wrong analysis that reads exactly like a correct one. A person
    settles it in one click; the model would guess.
    """
    result = ToolResult(ok=True, summary="2 aday", preview=[FLOW_RATE, STOCK_RATE])
    request = approvals.needs_series_choice(result, state, 1)

    assert request is not None
    assert request.kind == "series_choice"
    assert [option.id for option in request.options] == [
        "evds:TP.KTF12",
        "evds:TP.BKR.TRY.18",
    ]
    assert request.default == "evds:TP.KTF12"
    assert "%" in request.options[0].detail


def test_two_far_apart_candidates_of_the_same_shape_do_not_ask(state):
    """A clear winner in the same unit and kind is not an ambiguity."""
    runner_up = {**FLOW_RATE, "series_id": "evds:TP.OTHER", "score": 0.004}
    result = ToolResult(ok=True, summary="2 aday", preview=[FLOW_RATE, runner_up])
    assert approvals.needs_series_choice(result, state, 1) is None


def test_ambiguity_is_measured_as_a_ratio_not_a_gap(state):
    """Hybrid search returns fused ranks, roughly `1/(60+rank)`.

    Those values are only comparable inside one query, so a fixed gap like
    0.05 would either fire on every search or on none. The ratio holds whatever
    scale the real catalog ends up returning.
    """
    close = [FLOW_RATE, {**FLOW_RATE, "series_id": "evds:B", "score": 0.0150}]
    far = [FLOW_RATE, {**FLOW_RATE, "series_id": "evds:B", "score": 0.0100}]

    assert (
        approvals.needs_series_choice(
            ToolResult(ok=True, summary="", preview=close), state, 1, ratio=0.9
        )
        is not None
    )
    assert (
        approvals.needs_series_choice(
            ToolResult(ok=True, summary="", preview=far), state, 1, ratio=0.9
        )
        is None
    )


def test_the_choice_card_shows_at_most_four_options(state):
    """A card with twelve buttons is a card nobody reads."""
    many = [{**FLOW_RATE, "series_id": f"evds:S{i}"} for i in range(8)]
    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=many), state, 1
    )
    assert request is not None
    assert len(request.options) == 4


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------


def _quality(**kwargs) -> ToolResult:
    defaults = {
        "table_ref": "t01_x",
        "rows": 60,
        "freq": "M",
        "unit_guess": "milyon TL",
    }
    return ToolResult(
        ok=True,
        summary="",
        table_ref="t01_x",
        quality=QualityReport(**defaults | kwargs),
    )


def test_a_clean_table_passes_without_asking(state):
    assert approvals.needs_quality_gate(_quality(), state, 1) is None


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"missing_pct": 18.0}, "eksik veri"),
        ({"duplicates": 3}, "tekrarli donem"),
        ({"unit_guess": None}, "birim belirlenemedi"),
    ],
)
def test_each_quality_problem_is_named_in_the_question(state, kwargs, expected):
    """The card says what is wrong. "Quality issues, continue?" tells nobody
    anything, and the whole point of the gate is an informed decision."""
    request = approvals.needs_quality_gate(_quality(**kwargs), state, 1)
    assert request is not None
    assert expected in request.question


def test_a_tool_that_returns_no_quality_card_is_not_gated(state):
    """Most tools do not produce a table and have nothing to judge."""
    assert (
        approvals.needs_quality_gate(ToolResult(ok=True, summary=""), state, 1) is None
    )


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------


def test_request_ids_are_unique_within_a_turn(state):
    """The interface matches a person's answer back to a card by this id."""
    first = approvals.next_request_id(state)
    state.approvals.append(
        asyncio.run(
            approvals.auto_default(
                approvals.needs_quality_gate(_quality(missing_pct=50.0), state, 1)
            )
        )
    )
    assert approvals.next_request_id(state) != first


def test_the_automatic_mode_takes_the_default_and_says_so(state):
    """Used by tests and the eval runner, never in a demo: the point of a gate
    is that a person answers it, so the note records that nobody did."""
    request = approvals.needs_quality_gate(_quality(missing_pct=50.0), state, 1)
    decision = asyncio.run(approvals.auto_default(request))
    assert decision.choice == "continue"
    assert "otomatik" in decision.note


def test_a_series_choice_is_reported_back_as_a_person_speaking(state):
    """Written as a user message, not a tool result.

    A person overrode the model's ranking. Feeding that back as tool output
    would let the model weigh it like any other observation and search again;
    as a user message it carries the weight it should.
    """
    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=[FLOW_RATE, STOCK_RATE]), state, 1
    )
    decision = asyncio.run(approvals.auto_default(request))
    message = approvals.decision_message(request, decision)
    assert "evds:TP.KTF12" in message
    assert "tekrar katalog aramasi yapma" in message


def test_continuing_past_a_quality_warning_obliges_the_answer_to_mention_it(state):
    """Seeing the card and going on is fine. Going on silently is not."""
    request = approvals.needs_quality_gate(_quality(missing_pct=50.0), state, 1)
    decision = asyncio.run(approvals.auto_default(request))
    assert "sinirlari belirt" in approvals.decision_message(request, decision)
