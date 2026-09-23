"""When the agent stops and asks a person, and when it must not.

Both directions matter equally. KKB called human approval a differentiator, so
the gates have to fire on the cases a person genuinely decides better. But an
agent that asks at every step is unusable on stage, and the scenario is
supposed to run start to finish without a single interruption, so the tests
below pin the quiet cases just as tightly as the noisy ones.
"""

import asyncio
import types

import pytest

from lakehouse_agent.agent import approvals
from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.agent.state import ApprovalDecision, SessionState
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
    """The pool search scores by cosine similarity and falls back to keyword
    scores on another scale; a ratio means the same on both, a fixed gap
    like 0.05 would not.
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


def test_a_clear_winner_does_not_ask_even_in_another_unit(state):
    """The gate also fired whenever the top two differed in unit or kind,
    whatever the gap: 31 of the 36 pairs in the nine-series pool, so nearly
    every search stopped. The model sees the units; a winner by score is its
    call (planner rule 12)."""
    winner = {**FLOW_RATE, "score": 0.80}
    other = {**STOCK_RATE, "unit": "milyon TL", "kind": "stock", "score": 0.45}
    result = ToolResult(ok=True, summary="2 aday", preview=[winner, other])

    assert approvals.needs_series_choice(result, state, 1) is None


def test_series_the_query_named_by_code_are_not_a_choice(state):
    """ "evds:TP.KTF12 ve evds:TP.KFE.TR" returns both at 1.0. Asking which one
    and sending back "use this, do not search again" dropped the other series
    the person had asked for by name."""
    named = [
        {**FLOW_RATE, "score": 1.0, "eslesme": "kod"},
        {**STOCK_RATE, "score": 1.0, "eslesme": "kod"},
    ]
    result = ToolResult(ok=True, summary="2 aday", preview=named)

    assert approvals.needs_series_choice(result, state, 1) is None


def test_a_fallen_back_answer_is_not_a_persons_choice(state):
    """A stale or invalid answer falls back to the card's default -- the top
    search hit -- which nobody chose, so the model must not hear it as "a
    person picked this, do not search again"."""
    result = ToolResult(ok=True, summary="2 aday", preview=[FLOW_RATE, STOCK_RATE])
    request = approvals.needs_series_choice(result, state, 1)
    settled = approvals.settle(
        request, ApprovalDecision(request_id=request.request_id, choice="evds:YOK")
    )

    assert settled.automatic is True
    assert approvals.decision_message(request, settled) is None


def test_an_automatic_quality_default_does_not_claim_a_person_saw_it(state):
    request = approvals.needs_quality_gate(
        ToolResult(
            ok=True,
            summary="",
            quality=QualityReport(
                table_ref="t01", rows=10, missing_periods=5, unit_guess=None
            ),
        ),
        state,
        1,
    )
    decision = asyncio.run(approvals.auto_default(request))

    message = approvals.decision_message(request, decision)

    assert message.startswith("[Otomatik]")
    assert "gorulup" not in message
    assert "sinirlari belirt" in message


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
    decision = ApprovalDecision(request_id=request.request_id, choice="evds:TP.KTF12")
    message = approvals.decision_message(request, decision)
    assert "evds:TP.KTF12" in message
    assert "tekrar katalog aramasi yapma" in message


def test_a_default_series_choice_is_not_reported_as_a_person_speaking(state):
    """The default is just the top search hit. Saying a person chose it would
    overrule the model's own reading of the candidates with the ranker's."""
    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=[FLOW_RATE, STOCK_RATE]), state, 1
    )
    decision = asyncio.run(approvals.auto_default(request))
    assert decision.automatic is True
    assert approvals.decision_message(request, decision) is None


def test_changing_the_assumption_carries_the_person_s_own_words_forward(state):
    """The third quality option is not "continue" or "cancel" but "I disagree
    with how you read this". Whatever they typed has to reach the model, or the
    button does nothing."""
    request = approvals.needs_quality_gate(_quality(missing_pct=50.0), state, 1)
    decision = ApprovalDecision(
        request_id=request.request_id,
        choice=approvals.CHOICE_CHANGE_ASSUMPTION,
        note="Eksik aylari bos birak, sifir sayma.",
    )
    message = approvals.decision_message(request, decision)

    assert "varsayimi degistirdi" in message
    assert "sifir sayma" in message


def test_continuing_past_a_quality_warning_obliges_the_answer_to_mention_it(state):
    """Seeing the card and going on is fine. Going on silently is not."""
    request = approvals.needs_quality_gate(_quality(missing_pct=50.0), state, 1)
    decision = asyncio.run(approvals.auto_default(request))
    assert "sinirlari belirt" in approvals.decision_message(request, decision)


# --------------------------------------------------------------------------
# The answer is checked against the card it answers
# --------------------------------------------------------------------------


def _card(state):
    return approvals.needs_quality_gate(_quality(duplicates=3), state, 1)


def test_an_option_that_was_never_on_the_card_falls_back_to_the_default(state):
    """The decision arrives from outside the engine -- an interface, an eval
    runner, a test.

    An id nobody was offered is not a choice a person could have made, and
    acting on it would mean doing something nobody approved. `cancel` in
    particular deletes a table.
    """
    request = _card(state)
    settled = approvals.settle(
        request,
        ApprovalDecision(request_id=request.request_id, choice="uydurma_secenek"),
    )

    assert settled.choice == request.default
    assert "kartta olmayan secenek" in settled.note


def test_an_answer_to_a_different_card_is_not_applied_to_this_one(state):
    """Two gates can be open in one turn; the ids are what tells them apart."""
    request = _card(state)
    settled = approvals.settle(
        request, ApprovalDecision(request_id="baska-kart", choice="cancel")
    )

    assert settled.choice == request.default
    assert settled.request_id == request.request_id
    assert "farkli karta ait" in settled.note


def test_a_valid_answer_is_passed_through_untouched(state):
    request = _card(state)
    decision = ApprovalDecision(
        request_id=request.request_id, choice="cancel", note="kalite yetersiz"
    )

    assert approvals.settle(request, decision) is decision


# --------------------------------------------------------------------------
# The card has to survive the event channel
# --------------------------------------------------------------------------


def test_a_card_carrying_numpy_or_a_missing_date_still_serialises(state, ctx):
    """An approval card goes out as an `Event`, whose payload must serialise.

    The loop treats a failed gate as "skip the gate", so an unserialisable
    value in a catalog hit did not surface as an error -- it silently removed
    the human from the decision.
    """
    import json

    import numpy as np
    import pandas as pd

    result = ToolResult(
        ok=True,
        summary="",
        preview=[
            # Named as variants of one quantity, which is what the gate
            # now asks about; the test itself is about serialisation.
            {
                "series_id": "evds:A",
                "name_tr": "Konut Kredisi (TL, Akım, %)",
                "score": np.float64(0.91),
                "unit": "%",
            },
            {
                "series_id": "evds:B",
                "name_tr": "Konut Kredisi (TL, Stok, %)",
                "score": np.float64(0.90),
                "unit": "TL",
            },
        ],
    )
    request = approvals.needs_series_choice(result, state, 1)
    assert request is not None
    json.dumps(request.model_dump(mode="json"))

    fetch_spec = types.SimpleNamespace(
        approval="fetch",
        approval_preview=lambda args, ctx: {"source": "EVDS", "when": pd.NaT},
    )
    request = approvals.needs_fetch_approval(fetch_spec, None, ctx, state)
    assert request is not None
    assert request.context["when"] is None
    json.dumps(request.model_dump(mode="json"))


def test_a_preview_that_returns_the_wrong_shape_does_not_take_the_gate_down(state, ctx):
    """The existing guard covered a preview that raises. One that simply
    returns a list reached `.get` and lost the approval the same way."""
    spec = types.SimpleNamespace(
        approval="fetch", approval_preview=lambda args, ctx: ["EVDS"]
    )

    request = approvals.needs_fetch_approval(spec, None, ctx, state)

    assert request is not None
    assert request.context == {}


# --------------------------------------------------------------------------
# The missing-data test must work against either profiler
# --------------------------------------------------------------------------


def test_a_gap_is_caught_however_the_profiler_scales_its_percentage(state):
    """`lake/quality.py` reports a quarter missing as 0.25; the fake reports
    25.0. A threshold of ten is dead against the first: no fraction exceeds it.

    Counting periods is coarser but unambiguous, so the gate takes whichever
    of the two signals is larger.
    """
    as_fraction = _quality(rows=9, missing_periods=3, missing_pct=0.25)
    as_percent = _quality(rows=9, missing_periods=0, missing_pct=25.0)

    for result in (as_fraction, as_percent):
        request = approvals.needs_quality_gate(result, state, 1)
        assert request is not None
        assert "eksik veri %25.0" in request.question


def test_the_series_card_shows_the_table_before_anything_else():
    """Measured 23 Sep: two candidates both read "Ağırlıklı ortalama, W,
    rate" and a person could not tell the loan rate from the profit share."""
    from lakehouse_agent.agent import approvals
    from lakehouse_agent.agent.state import initial_state
    from lakehouse_agent.contracts import ToolResult

    result = ToolResult(
        ok=True,
        summary="2 aday",
        preview=[
            {
                "series_id": "evds:TP.KBK.TRY.18",
                "name_tr": "Konut Kredisi, (TL, Akım, %)",
                "tablo": "Kredi Kâr Oranları (Akım)",
                "unit": "Ağırlıklı ortalama",
                "freq": "W",
                "kind": "rate",
                "score": 0.7153,
            },
            {
                "series_id": "evds:TP.KTF12",
                "name_tr": "Konut Kredisi (TL, Akım, %)",
                "tablo": "Kredi Faiz Oranları (Akım)",
                "unit": "Ağırlıklı ortalama",
                "freq": "W",
                "kind": "rate",
                "score": 0.6684,
            },
        ],
    )

    request = approvals.needs_series_choice(result, initial_state("s"), 1)

    assert request is not None
    assert request.options[0].detail.startswith("Kredi Kâr Oranları (Akım), ")
    assert request.options[1].detail.startswith("Kredi Faiz Oranları (Akım), ")


# --- a card is for a choice, not for the search being unsure ---------------


def _hits(*names: str) -> ToolResult:
    """Candidates whose scores are close enough to open the gate, so what is
    left under test is whether they are a choice at all."""
    return ToolResult(
        ok=True,
        summary="",
        preview=[
            {"series_id": f"evds:S{n}", "name_tr": name, "score": 0.70 - n * 0.01}
            for n, name in enumerate(names)
        ],
    )


def test_one_quantity_written_several_ways_is_a_choice(state):
    """The run that made this matter: interest rate or profit share, flow or
    stock. Four rows reading "Konut Kredisi (...)", and only a person knows
    which the question meant."""
    result = _hits(
        "Konut Kredisi, (TL, Akım, %)",
        "Konut Kredisi (TL, Stok, %)",
        "Konut Kredisi (TL, Akım, %)",
        "Konut Kredisi (TL, Stok, %)",
    )

    assert approvals.needs_series_choice(result, state, 1) is not None


def test_different_quantities_are_not_a_choice(state):
    """The card a person actually got: non-performing housing loans, consumer
    housing loans, total loans and short-term total loans. Four different
    quantities, none of them obviously the answer, and the default was the
    wrong one. The model reading their names decides better than a person
    reading four lines."""
    result = _hits(
        "Takipteki Konut Kredileri (Toplam)",
        "Tüketici Kredileri - Konut (Toplam)",
        "Toplam Krediler (Toplam)",
        "Toplam Krediler (Kısa Toplam)",
    )

    assert approvals.needs_series_choice(result, state, 1) is None


def test_the_top_two_alone_would_have_kept_the_wrong_card(state):
    """Why the measure looks at every candidate shown: the top two of that bad
    card share three words out of five, as high as the good card's four
    together. It is the whole set that tells them apart."""
    top_two_only = approvals._same_quantity(
        ["Takipteki Konut Kredileri (Toplam)", "Tüketici Kredileri - Konut (Toplam)"]
    )
    all_four = approvals._same_quantity(
        [
            "Takipteki Konut Kredileri (Toplam)",
            "Tüketici Kredileri - Konut (Toplam)",
            "Toplam Krediler (Toplam)",
            "Toplam Krediler (Kısa Toplam)",
        ]
    )

    assert top_two_only >= approvals.MIN_SHARED_WORDS
    assert all_four < approvals.MIN_SHARED_WORDS


def test_a_candidate_without_a_name_is_read_by_its_id(state):
    """The card labels an unnamed candidate with its id, so the gate has to
    weigh the same text or the two would disagree."""
    result = ToolResult(
        ok=True,
        summary="",
        preview=[
            {"series_id": "evds:TP.KTF12", "score": 0.70},
            {"series_id": "evds:TP.KTF12", "score": 0.69},
        ],
    )

    assert approvals.needs_series_choice(result, state, 1) is not None


# --- name twins (23 Sep) ------------------------------------------------------


def _hit(series_id, name, score, tablo):
    return {
        "series_id": series_id,
        "name_tr": name,
        "score": score,
        "tablo": tablo,
        "unit": "Ağırlıklı ortalama",
        "freq": "W",
        "kind": "rate",
        "eslesme": "anlamsal",
    }


def test_two_candidates_with_one_name_are_a_card_even_with_a_stranger_between(state):
    """Measured 23 Sep: profit share first, the loan rate third, a stock
    series between them; the old rule saw four names and stayed quiet."""
    preview = [
        _hit(
            "evds:TP.KBK.TRY.18",
            "Konut Kredisi, (TL, Akım, %)",
            0.7128,
            "Kredi Kâr Oranları (Akım)",
        ),
        _hit(
            "evds:TP.KKP.TRY.18",
            "Konut Kredisi (TL, Stok, %)",
            0.6792,
            "Kredi Kâr Oranları (Stok)",
        ),
        _hit(
            "evds:TP.KTF12",
            "Konut Kredisi (TL, Akım, %)",
            0.6650,
            "Kredi Faiz Oranları (Akım)",
        ),
        {
            **_hit("evds:TP.KM.B11", "3.2.1.1. Konut Kredileri", 0.6454, "Krediler"),
            "kind": "stock",
        },
    ]

    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=preview), state, 1
    )

    assert request is not None
    assert [o.id for o in request.options] == ["evds:TP.KBK.TRY.18", "evds:TP.KTF12"]
    assert request.options[0].detail.startswith("Kredi Kâr Oranları (Akım)")
    assert request.options[1].detail.startswith("Kredi Faiz Oranları (Akım)")


def test_names_differing_only_in_a_number_are_not_twins(state):
    """ "3 Aya Kadar Vadeli" and "6 Aya Kadar Vadeli" are different maturities."""
    preview = [
        _hit("evds:A", "3 Aya Kadar Vadeli (Euro, %)", 0.70, "Mevduat Faiz"),
        _hit("evds:B", "6 Aya Kadar Vadeli (Euro, %)", 0.69, "Mevduat Faiz"),
    ]
    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=preview), state, 1
    )
    # the old rule may still ask (same quantity, close scores) -- but not as twins
    assert approvals._name_twins(preview, 0.9, 4) == []
    assert request is None or len(request.options) == 2


def test_a_twin_scoring_far_below_the_top_is_left_to_the_model(state):
    preview = [
        _hit("evds:A", "Konut Kredisi, (TL, Akım, %)", 0.70, "Kâr"),
        _hit("evds:B", "Konut Kredisi (TL, Akım, %)", 0.50, "Faiz"),
    ]
    assert approvals._name_twins(preview, 0.9, 4) == []


def _searched(state, query):
    """The history as the loop leaves it: the search call already made."""
    import json as _json

    state.messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "search_catalog",
                        "arguments": _json.dumps({"query": query}),
                    },
                }
            ],
        }
    )
    return state


def test_the_twin_whose_table_matches_the_question_is_the_default(state):
    """ "faiz" in the question, "Faiz" in the table: the loan rate, not the
    profit share that happened to score higher. In the auto path the
    default is what gets applied, so this is what the eval fetches."""
    preview = [
        _hit(
            "evds:TP.KBK.TRY.18",
            "Konut Kredisi, (TL, Akım, %)",
            0.7128,
            "Kredi Kâr Oranları (Akım)",
        ),
        _hit(
            "evds:TP.KTF12",
            "Konut Kredisi (TL, Akım, %)",
            0.6650,
            "Kredi Faiz Oranları (Akım)",
        ),
    ]
    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=preview),
        _searched(state, "konut kredisi faiz oranı"),
        1,
    )

    assert request is not None
    assert request.default == "evds:TP.KTF12"
    assert [o.id for o in request.options] == [
        "evds:TP.KBK.TRY.18",
        "evds:TP.KTF12",
    ]  # score order kept


def test_without_a_matching_word_the_top_score_stays_the_default(state):
    preview = [
        _hit("evds:A", "8.İşsizlik oranı (%)", 0.70, "İşgücü İstatistikleri"),
        _hit("evds:B", "8.İşsizlik oranı (%)", 0.69, "Mevsim Etkisinden Arındırılmış"),
    ]
    request = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=preview),
        _searched(state, "işsizlik oranı"),
        1,
    )
    assert request is not None and request.default == "evds:A"

    # no search call in the history at all: still the top
    from lakehouse_agent.agent.state import initial_state

    untouched = initial_state("s2")
    fresh = approvals.needs_series_choice(
        ToolResult(ok=True, summary="", preview=preview), untouched, 1
    )
    assert fresh is not None and fresh.default == "evds:A"
    assert approvals._query_hint(untouched) == ""
