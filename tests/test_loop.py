"""The turn loop: what it does, and what it refuses to do.

These tests run against `fake_tools`, not `FakeLake`, on purpose. The loop's job
is control flow — pick a tool, validate its arguments, run it, merge what it
wrote, decide whether to ask a person, keep going or stop — and none of that
depends on the arithmetic inside a tool. Testing it against real series would
turn every control-flow failure into a debugging session about aggregation.

The model is scripted. `FakeLLM` raises `ScriptExhausted` when the loop asks for
one reply more than a test provides, so "the loop took an extra step" fails
loudly instead of passing quietly.
"""

import asyncio
import json

import pandas as pd
import pytest

from lakehouse_agent.agent import loop as loop_module
from lakehouse_agent.agent.events import ListSink
from lakehouse_agent.agent.loop import AgentConfig, AgentDeps, AgentError, run_turn
from lakehouse_agent.agent.prompts import load_prompt
from lakehouse_agent.agent.state import ApprovalDecision, initial_state
from lakehouse_agent.llm import MIN_ATTEMPT_S, LlmServerError, LlmTimeoutError
from lakehouse_agent.testing.fake_llm import FakeLLM, ScriptExhausted
from lakehouse_agent.testing.fake_tools import make_fake_registry

SEARCH = ("search_catalog", {"query": "konut kredisi faiz orani"})
CREDIT = "aylik_tuketici_konut_TL_sektor"
FETCH_CREDIT = (
    "get_series",
    {
        "series_ids": ["bddk:aylik:tuketici:konut:TL:sektor"],
        "start": "2021-01",
        "end": "2025-12",
    },
)
FETCH_RATE = (
    "get_series",
    {
        "series_ids": ["evds:TP.KTF12"],
        "start": "2021-01",
        "end": "2025-12",
        "into": "new",
    },
)


def build(
    script, *, sink=None, config=None, approve=None, clock=None, **registry_kwargs
):
    """A turn's worth of dependencies, with everything swappable."""
    from lakehouse_agent.config import AgentSettings

    deps = AgentDeps(
        llm=FakeLLM(script),
        registry=make_fake_registry(**registry_kwargs),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        sink=sink,
        config=config or AgentConfig(),
        approve=approve,
    )
    if clock is not None:
        deps.clock = clock
    return deps


def turn(state, deps, question="Konut kredisi faizi nasil seyretti?"):
    return asyncio.run(run_turn(state, deps, question))


def tool_messages(state):
    return [m for m in state.messages if m.get("role") == "tool"]


def assistant_tool_calls(state):
    return [m for m in state.messages if m.get("tool_calls")]


def orphan_call_ids(state):
    """Tool calls the model made that nobody answered.

    One of these makes the *next* request to MIA a protocol error, so it is not
    a cosmetic issue: it fails the turn after this one and every turn after
    that. Asserted wherever a turn can end early.
    """
    answered = {m["tool_call_id"] for m in tool_messages(state)}
    return [
        call["id"]
        for message in state.messages
        for call in message.get("tool_calls") or []
        if call["id"] not in answered
    ]


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_one_tool_then_an_answer():
    """The shape everything else is a deviation from."""
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, content="Seriyi getiriyorum."),
            FakeLLM.text("Veri hazir."),
            FakeLLM.text("Faiz 2021-01 ile 2025-12 arasinda yukseldi."),
        ],
        sink=sink,
    )

    result = turn(state, deps)

    assert result.answer.startswith("Faiz")
    assert result.truncated_reason is None
    assert result.table_refs == ["t01_get_series"]
    assert state.active_table == "t01_get_series"
    assert list(state.tables["t01_get_series"].columns) == ["period", "TP_KTF12"]
    assert [step.node for step in state.steps] == [
        "plan",
        "act",
        "observe",
        "answer",
        "verify",
    ]
    # Every step opens and closes a card, the model calls included: they are
    # the only steps that carry a token count, and docs/contracts.md puts that
    # count in the `step_end` payload.
    assert sink.types() == [
        "step_start",  # plan
        "plan",
        "step_end",
        "step_start",  # get_series
        "step_end",
        "table_update",
        "sources",
        "step_start",  # observe
        "step_end",
        "answer_delta",
        "verification",
        "done",
    ]
    model_steps = [
        event
        for event in sink.of_type("step_end")
        if event.payload["node"] in ("plan", "observe")
    ]
    assert model_steps and all(
        event.payload["tokens"] is not None for event in model_steps
    )
    assert sink.of_type("step_start")[1].payload["tool"] == "get_series"


def test_the_model_never_receives_the_table_only_a_description_of_it():
    """The rule the whole formatting module exists for.

    A model that holds the digits copies them into its answer with one decimal
    place slipped, and nothing downstream can tell. It gets row counts, a date
    range and column names instead.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )
    turn(state, deps)

    message = tool_messages(state)[0]["content"]
    assert "60 satir" in message
    assert "2021-01 - 2025-12" in message
    assert "TP_KTF12" in message
    # The 37.5 in row 54 is in the table and must not be in the model's context.
    assert "37.5" not in message


def test_two_tools_in_sequence_share_one_table_and_build_a_lineage_chain():
    """The follow-up case: add a derived column to the table already on screen.

    The assertion that matters is the last one. A deflated column descends from
    both the nominal series and the price index, and if the chain only recorded
    one of them the traceability panel would show a number nobody can account
    for.
    """
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_CREDIT, call_id="c1"),
            FakeLLM.tool_call(
                "transform_table", {"op": "deflate", "column": CREDIT}, call_id="c2"
            ),
            FakeLLM.text("Hazir."),
            FakeLLM.text("Reel kredi hesaplandi."),
        ],
        sink=sink,
    )

    turn(state, deps)

    frame = state.tables["t01_get_series"]
    assert list(frame.columns) == ["period", CREDIT, f"reel_{CREDIT}"]
    assert state.active_table == "t01_get_series"
    assert sink.of_type("table_update")[-1].payload["new_columns"] == [f"reel_{CREDIT}"]

    column = state.lineage[f"t01_get_series.reel_{CREDIT}"]
    assert column.op == "deflate"
    assert column.inputs == [CREDIT]
    assert column.source_series == [
        "bddk:aylik:tuketici:konut:TL:sektor",
        "evds:TP.TUKFIY2025.GENEL",
    ]


def test_the_fake_refuses_to_deflate_a_rate_like_the_real_tool():
    """The 22 Sep live run deflated an interest rate through the fake, because
    the fake allowed what the real tool refuses. The loop tests judge control
    flow on these fakes, so they call the real tool's rule
    (`tools.lakehouse.deflate_refusal`) instead of restating it."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.tool_call(
                "transform_table", {"op": "deflate", "column": "TP_KTF12"}, call_id="c2"
            ),
            FakeLLM.text("Oran reellestirilemez."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert "oran" in tool_messages(state)[-1]["content"]
    assert "reel_TP_KTF12" not in state.tables["t01_get_series"].columns


# --------------------------------------------------------------------------
# What a tool is not allowed to do
# --------------------------------------------------------------------------


def test_a_tool_cannot_change_the_session_table_under_the_user():
    """`ToolContext.table()` hands out a shallow copy, and this is why.

    The user is looking at that table. A tool that assigns to the frame it was
    given, without registering the write, must not be able to change what is on
    screen — least of all half way through a step that then fails.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.tool_call("sneaky_tool", {}, call_id="c2"),
            FakeLLM.text("Hazir."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    frame = state.tables["t01_get_series"]
    assert "gizli" not in frame.columns
    assert frame["TP_KTF12"].iloc[0] == 10.0


def test_changing_an_existing_column_is_caught_and_reported():
    """The loop verifies the promise instead of trusting it.

    A tool that adds a column and also rewrites one that was there produces an
    answer that reads exactly like a correct one. The old columns are compared
    value by value, and the mismatch reaches both the model and the answer.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.tool_call("corrupting_tool", {}, call_id="c2"),
            FakeLLM.text("Hazir."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert any("mevcut sutunlar degisti" in w for w in state.warnings)
    assert "mevcut sutunlar degisti" in tool_messages(state)[1]["content"]


# --------------------------------------------------------------------------
# Failures the loop has to absorb
# --------------------------------------------------------------------------


def test_a_tool_that_raises_becomes_a_message_not_a_dead_turn():
    """The model gets told what broke and carries on with what it has."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("failing_tool", {}, call_id="c1"),
            FakeLLM.text("Anladim."),
            FakeLLM.text("Bu adim basarisiz oldu, elimdekiyle devam ediyorum."),
        ]
    )

    result = turn(state, deps)

    message = tool_messages(state)[0]["content"]
    assert message.startswith("[HATA]")
    assert "arac patladi" in message
    assert result.answer
    assert state.steps[1].ok is False


def test_an_unknown_tool_is_answered_with_the_list_of_real_ones():
    """Open-weights models invent tool names. Naming the real ones is how the
    next step gets to be a correct one."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("sql_query", {"sql": "select 1"}, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    message = tool_messages(state)[0]["content"]
    assert "Bilinmeyen arac" in message
    assert "get_series" in message


def test_bad_arguments_are_fed_back_once_and_only_once():
    """MIA was observed writing SQL into an unexplained field.

    Showing it the validation error works the first time. Insisting a second
    time means it is not reading the error, and a third attempt would burn
    quota we need for the demo, so the second message tells it to stop.
    """
    bad = ("get_series", {"series_ids": "evds:TP.KTF12", "start": "2021", "end": "x"})
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*bad, call_id="c1"),
            FakeLLM.tool_call(*bad, call_id="c2"),
            FakeLLM.text("Vazgectim."),
            FakeLLM.text("Seriyi getiremedim."),
        ]
    )

    turn(state, deps)

    messages = [m["content"] for m in tool_messages(state)]
    assert len(messages) == 2
    assert "Duzeltip tekrar dene" in messages[0]
    assert "tekrar deneme" in messages[1]
    assert state.tables == {}


def test_arguments_that_are_not_json_are_handled_like_any_other_bad_argument():
    """vLLM emits malformed tool arguments under load; it must not end the turn."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.bad_json_tool_call("get_series", '{"series_ids": [', call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert "[HATA]" in tool_messages(state)[0]["content"]
    assert state.tables == {}


def test_a_step_that_overruns_is_abandoned_and_writes_nothing():
    """The thread cannot be killed, so the context it holds is thrown away.

    Whatever the abandoned tool writes afterwards lands in an orphan object.
    The session never sees it, which is the entire reason writes go through a
    per-step buffer rather than into the session directly.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("slow_tool", {"seconds": 0.4}, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Bu adim zaman asimina ugradi."),
        ],
        config=AgentConfig(step_timeout_s=0.05),
    )

    result = turn(state, deps)

    assert "0.05 saniye icinde bitmedi" in tool_messages(state)[0]["content"]
    assert state.steps[1].ok is False
    assert state.steps[1].error == "timeout"
    assert state.tables == {}
    assert result.answer


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------


def test_the_step_limit_stops_the_loop_and_the_answer_says_so():
    """Truncating is fine. Truncating and presenting the result as complete is
    the failure the trust layer exists to prevent, so the reason travels into
    the answer prompt."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("search_catalog", {"query": f"q{i}"}, call_id=f"c{i}")
            for i in range(4)
        ]
        + [FakeLLM.text("Yarim kaldi ama soyle diyebilirim.")],
        config=AgentConfig(max_steps=3),
    )

    result = turn(state, deps)

    assert result.truncated_reason == "step_limit"
    assert state.steps_this_turn == 3
    # Three tools ran; the fourth call the model had already made is answered
    # with a refusal rather than left dangling.
    assert len(tool_messages(state)) == 4
    assert orphan_call_ids(state) == []
    assert "calistirilmadi" in tool_messages(state)[3]["content"]
    system_prompt = deps.llm.calls[-1]["messages"][0]["content"]
    assert "Adim sinirina ulasildi" in system_prompt


def test_the_same_call_three_times_over_is_treated_as_a_loop():
    """Re-calling one tool with identical arguments is not a plan."""
    same = ("search_catalog", {"query": "konut"})
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*same, call_id="c1"),
            FakeLLM.tool_call(*same, call_id="c2"),
            FakeLLM.tool_call(*same, call_id="c3"),
            FakeLLM.text("Cevap."),
        ]
    )

    result = turn(state, deps)

    assert result.truncated_reason == "repeat_loop"
    assert "tekrarlama" in tool_messages(state)[2]["content"]


def test_waiting_for_a_person_is_not_charged_to_the_turn_budget():
    """A jury reading an approval card is not the agent being slow.

    Measured on a fake clock that jumps five hundred seconds while the card is
    open. On wall time that is far past the budget; on working time it is one
    fast step, which is the only reading under which an approval gate and a
    time limit can coexist.
    """
    ticks = [0.0]

    def clock() -> float:
        return ticks[0]

    async def slow_human(request):
        ticks[0] += 500.0
        return ApprovalDecision(request_id=request.request_id, choice="approve")

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(
                "ingest", {"source": "evds", "series_id": "evds:TP.YENI"}, call_id="c1"
            ),
            FakeLLM.text("Geldi."),
            FakeLLM.text("Seri cekildi."),
        ],
        approve=slow_human,
        clock=clock,
    )

    result = turn(state, deps)

    assert result.truncated_reason is None
    assert state.approval_wait_s >= 500.0
    assert state.active_s < 180.0
    assert result.duration_ms >= 500_000  # wall time really did pass
    assert state.tables


def test_the_budget_does_stop_a_turn_that_is_genuinely_slow():
    """The other half of the test above: working time still counts.

    Slow is expressed as a slow model rather than a jumping clock, because that
    is the real shape of it: thinking-on calls against a 27B model are where a
    turn actually spends its minutes.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(
                "search_catalog", {"query": "a"}, call_id="c1", duration_ms=100_000
            ),
            FakeLLM.tool_call(
                "search_catalog", {"query": "b"}, call_id="c2", duration_ms=100_000
            ),
            FakeLLM.text("Cevap."),
        ],
        config=AgentConfig(total_budget_s=150.0),
    )

    result = turn(state, deps)

    assert result.truncated_reason == "time_budget"
    assert state.active_s >= 150.0
    assert orphan_call_ids(state) == []


def _times_out(messages, tools, thinking):
    raise LlmTimeoutError("MIA zaman asimi: Request timed out.")


def test_every_model_call_is_held_to_what_is_left_of_the_budget():
    """The 22 Sep live run: 57 s of work, then one answer call that retried
    its way to 138 s -- 195 s against a 180 s budget, because the budget was
    only checked between tool steps and each retry got a fresh 45 s."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1", duration_ms=140_000),
            FakeLLM.text("Tamam.", duration_ms=5_000),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    plan, observe, answer = (call["timeout_s"] for call in deps.llm.calls)
    # Planning is capped at the client timeout, so it leaves the turn time.
    assert plan == pytest.approx(deps.settings.llm_timeout_s)
    # 180 - 140 = 40 left: under the cap, so that is the limit.
    assert observe == pytest.approx(40.0, abs=0.5)
    # 180 - 140 - 5 = 35 left, and the answer, the last call, gets all of it.
    assert answer == pytest.approx(35.0, abs=0.5)


def test_the_answer_still_gets_its_minimum_after_the_budget_is_spent():
    """A turn that did its work and ran out of time to say so is the worst
    way for one to end. The answer is the one call allowed past the budget."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1", duration_ms=100_000),
            FakeLLM.text("Tamam.", duration_ms=100_000),
            FakeLLM.text("Cevap."),
        ],
        config=AgentConfig(total_budget_s=150.0),
    )

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert deps.llm.calls[-1]["timeout_s"] == pytest.approx(30.0)


def test_with_no_budget_left_to_observe_the_turn_goes_straight_to_the_answer():
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1", duration_ms=175_000),
            FakeLLM.text("Cevap."),
        ]
    )

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert result.truncated_reason == "time_budget"
    assert [call["tag"] for call in deps.llm.calls] == [
        f"plan@{load_prompt('planner').tag}",
        f"answer@{load_prompt('answer').tag}",
    ]
    assert orphan_call_ids(state) == []


def test_a_timed_out_answer_is_retried_once_without_thinking():
    """Retrying the identical request cannot help when the server is slow:
    the same answer takes as long again. Thinking is where the time goes, so
    the one retry turns it off and caps the output instead."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam."),
            _times_out,
            FakeLLM.text("Cevap, dusunmeden."),
        ]
    )

    result = turn(state, deps)

    assert result.answer == "Cevap, dusunmeden."
    first, retry = deps.llm.calls[-2:]
    assert (first["thinking"], first["max_tokens"]) == (True, None)
    assert (retry["thinking"], retry["max_tokens"]) == (False, 1024)
    assert state.steps[-2].node == "answer"
    assert state.steps[-2].thinking is False


def test_a_timed_out_plan_is_retried_once_without_thinking():
    state = initial_state("s1")
    deps = build(
        [
            _times_out,
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert [call["thinking"] for call in deps.llm.calls[:2]] == [True, False]


def test_a_timed_out_observe_is_not_retried_and_the_turn_still_answers():
    """Observe runs with thinking off, so there is no cheaper request to fall
    back to, and a second identical attempt is what not to send. But observing
    only decides the next move: the tool results are already in the session.
    Measured 22 Sep: this timeout came right after the scenario's last tool
    had listed the months, and the finished analysis was thrown away."""
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            _times_out,
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
    )

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert result.truncated_reason == "llm_error"
    assert [call["tag"].split("@")[0] for call in deps.llm.calls] == [
        "plan",
        "observe",
        "answer",
    ]
    assert state.tables  # the work done before the timeout is kept
    assert orphan_call_ids(state) == []
    # The observe card that opened is closed, as failed.
    failed = [e for e in sink.of_type("step_end") if e.payload["ok"] is False]
    assert [e.payload["node"] for e in failed] == ["observe"]
    assert failed[0].payload["error"] == "LlmTimeoutError"


def test_a_server_error_while_observing_still_fails_the_turn():
    """Only slowness is recovered from. A 5xx after every retry means MIA is
    down, and the answer call would meet the same server. (A 5xx whose retry
    the budget stopped is slowness: the client reports it as a timeout, see
    test_llm.)"""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            lambda *_: (_ for _ in ()).throw(LlmServerError("MIA 503")),
        ]
    )

    with pytest.raises(AgentError, match="MIA 503"):
        turn(state, deps)


class _Clock:
    """A clock the scripted model moves: a timeout takes the time it was given."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _times_out_on(clock, holder, overshoot=0.0):
    """Script entry for a call that runs its whole timeout and fails.

    `overshoot` is what a real timeout costs beyond the number it was given:
    httpx fires the read timeout at or after the deadline, and the exception
    still has to travel back through the worker thread. It is small and it is
    never zero, which is the whole of the bug the caller below reproduces.
    """

    def reply(messages, tools, thinking):
        clock.now += holder[0].llm.calls[-1]["timeout_s"] + overshoot
        raise LlmTimeoutError("MIA zaman asimi: Request timed out.")

    return reply


def _timed_turn(script_for, *, config=None, sink=None, overshoot=0.0):
    """Run one turn whose failing calls advance a shared clock."""
    clock = _Clock()
    holder = []
    deps = build(
        script_for(_times_out_on(clock, holder, overshoot)),
        clock=clock,
        config=config,
        sink=sink,
    )
    holder.append(deps)
    state = initial_state("s1")
    return state, deps, turn(state, deps)


def test_the_answer_floor_is_one_allowance_not_one_per_attempt():
    """The floor was re-applied to the retry: a thinking-on answer that timed
    out at 30 s got a fresh 30 s for its retry, 60 s past the budget. The two
    attempts now share one allowance."""
    state, deps, result = _timed_turn(
        lambda times_out: [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            # Slow in time, fast in tokens: the budget is spent, and the model
            # is quick enough that the answer tries thinking first.
            FakeLLM.text("Tamam.", duration_ms=170_000, completion_tokens=100_000),
            times_out,
            FakeLLM.text("Cevap."),
        ]
    )

    first, retry = (call["timeout_s"] for call in deps.llm.calls[-2:])
    assert result.answer == "Cevap."
    # 10 s of budget left, so the answer's 30 s floor is its allowance; the
    # thinking attempt leaves the retry `min_call_s` of it.
    assert first == pytest.approx(20.0, abs=0.5)
    assert retry == pytest.approx(10.0, abs=0.5)
    assert first + retry == pytest.approx(deps.config.answer_min_s, abs=0.5)


def test_the_fallback_still_runs_when_the_timeout_overran_its_window():
    """The blocker this guards against: the thinking-on attempt is capped at
    `allowance - min_call_s`, so a timeout that fires a hair late leaves just
    under `min_call_s`. Tested against a clock that advanced by exactly the
    timeout, the old `left() < min_call_s` check passed; against any real
    clock it raised and the turn ended with no answer at all -- and only on
    the turns tight enough to need the fallback, which is every turn that had
    already spent 125 s of its 180."""
    state, deps, result = _timed_turn(
        lambda times_out: [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam.", duration_ms=170_000, completion_tokens=100_000),
            times_out,
            FakeLLM.text("Cevap."),
        ],
        overshoot=0.05,
    )

    first, retry = deps.llm.calls[-2:]
    assert result.answer == "Cevap."
    assert (first["thinking"], retry["thinking"]) == (True, False)
    # What is left after the overshoot: under `min_call_s`, over what the
    # client needs to start an attempt.
    assert 5.0 <= retry["timeout_s"] < 10.0


def test_the_last_chance_answer_gets_all_the_time_that_is_left():
    """The third 22 Sep run: the thinking-off answer was capped at the 45 s
    client timeout and timed out with about a minute of the budget unused."""
    state, deps, result = _timed_turn(
        lambda times_out: [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam.", duration_ms=80_000, completion_tokens=100_000),
            times_out,
            FakeLLM.text("Cevap, dusunmeden."),
        ]
    )

    first, retry = deps.llm.calls[-2:]
    assert result.answer == "Cevap, dusunmeden."
    assert (first["thinking"], first["timeout_s"]) == (True, pytest.approx(45.0))
    # 180 - 80 - 45 = 55: more than the client timeout, and all of what is left.
    assert (retry["thinking"], retry["max_tokens"]) == (False, 1024)
    assert retry["timeout_s"] == pytest.approx(55.0, abs=0.5)


def test_a_slow_model_answers_without_thinking_from_the_start():
    """Measured in the third 22 Sep run: plan and observe calls at 7-10
    tokens/s, and a thinking-on answer (750 tokens on 15 Sep) cannot fit in
    45 s at that speed. Trying it anyway cost 45 s and then the answer."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1", duration_ms=27_321),
            FakeLLM.text("Tamam.", completion_tokens=109, duration_ms=12_784),
            FakeLLM.text("Cevap."),
        ]
    )

    result = turn(state, deps)

    answer = deps.llm.calls[-1]
    assert result.answer == "Cevap."
    assert (answer["thinking"], answer["max_tokens"]) == (False, 1024)
    # Everything left, in one attempt: 180 - 27.3 - 12.8.
    assert answer["timeout_s"] == pytest.approx(139.9, abs=0.5)
    assert state.steps[-2].thinking is False


def test_a_failed_model_step_reports_the_time_it_took():
    """A timed-out observe call reported 0 ms: the step card and the recap
    showed nothing for a quarter of the budget."""
    sink = ListSink()
    state, deps, result = _timed_turn(
        lambda times_out: [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            times_out,
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
    )

    failed = [e for e in sink.of_type("step_end") if e.payload["ok"] is False]
    assert [e.payload["duration_ms"] for e in failed] == [45_000]
    observe = next(step for step in state.steps if step.node == "observe")
    assert (observe.duration_ms, observe.thinking) == (45_000, False)


def test_a_retried_step_reports_both_attempts():
    """After a timeout and a successful retry the step is the whole of it,
    not only the attempt that answered, and it says thinking was off."""
    state, deps, result = _timed_turn(
        lambda times_out: [
            times_out,
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1", duration_ms=2_000),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    plan = state.steps[0]
    assert plan.node == "plan"
    assert plan.duration_ms == 47_000
    assert plan.thinking is False


def test_a_failed_retry_records_the_attempts_that_ran():
    """The failed step recorded the thinking it was asked for (on), though
    its last attempt, the one that failed it, ran with thinking off, and a
    duration of 0 for two full timeouts."""
    clock = _Clock()
    holder = []
    times_out = _times_out_on(clock, holder)
    deps = build([times_out, times_out], clock=clock)
    holder.append(deps)
    state = initial_state("s1")

    with pytest.raises(AgentError):
        turn(state, deps)

    plan = state.steps[0]
    assert (plan.node, plan.ok, plan.thinking) == ("plan", False, False)
    assert plan.duration_ms == 90_000  # 45 s thinking on, then 45 s off


def test_the_answer_request_does_not_end_on_the_models_own_note():
    """Ending on the observe step's "Veriler hazir." made DeepSeek continue
    that message instead of answering (22 Sep: 16 tokens of nonsense)."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Veriler hazir."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    sent = deps.llm.calls[-1]["messages"]
    assert sent[-1]["role"] == "tool"
    assert all(m.get("content") != "Veriler hazir." for m in sent)
    # The session keeps it; only the request leaves it out.
    assert any(m.get("content") == "Veriler hazir." for m in state.messages)


def test_a_failed_answer_is_in_the_step_log():
    """Both answer attempts timing out ended the turn with no step saying
    where it ended or what the answer had cost."""
    clock = _Clock()
    holder = []
    times_out = _times_out_on(clock, holder)
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam."),
            times_out,
            times_out,
        ],
        clock=clock,
    )
    holder.append(deps)
    state = initial_state("s1")

    with pytest.raises(AgentError):
        turn(state, deps)

    answer = state.steps[-1]
    assert (answer.node, answer.ok, answer.error) == (
        "answer",
        False,
        "LlmTimeoutError",
    )
    # 45 s thinking on, then the rest of the allowance: 180 s less the 10 ms
    # the plan and observe calls took.
    assert answer.duration_ms == 179_990
    assert answer.thinking is False


def test_a_model_call_that_crashes_still_closes_its_card():
    """Only `LlmError` closed the card: a cancelled turn or a client bug left
    the plan step running forever on the screen."""
    sink = ListSink()
    state = initial_state("s1")

    def crash(messages, tools, thinking):
        raise RuntimeError("istemci hatasi")

    deps = build([crash], sink=sink)

    with pytest.raises(RuntimeError):
        turn(state, deps)

    ends = sink.of_type("step_end")
    assert [(e.payload["node"], e.payload["ok"]) for e in ends] == [("plan", False)]
    assert state.steps[0].error == "RuntimeError"


def test_a_turn_stopped_at_the_approval_card_answers_the_call():
    """Cancellation is not an `Exception`, so the approval gate's handler did
    not see it and the call stayed unanswered: every later request a 400."""

    async def stopped(request):
        raise asyncio.CancelledError

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(
                "ingest", {"source": "evds", "series_id": "evds:TP.YENI"}, call_id="c1"
            ),
        ],
        approve=stopped,
    )

    with pytest.raises(asyncio.CancelledError):
        turn(state, deps)

    assert orphan_call_ids(state) == []
    assert "durduruldu" in tool_messages(state)[0]["content"]


def test_a_stopped_turn_does_not_close_the_channel_as_a_finished_one():
    """Cancellation is not an `Exception` either, so it fell through to the
    `finally` and published `done` with `truncated_reason: null` -- the same
    payload a turn that answered publishes. An interface reading that draws a
    stopped turn as a success."""
    sink = ListSink()

    async def stopped(request):
        raise asyncio.CancelledError

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(
                "ingest", {"source": "evds", "series_id": "evds:TP.YENI"}, call_id="c1"
            ),
        ],
        approve=stopped,
        sink=sink,
    )

    with pytest.raises(asyncio.CancelledError):
        turn(state, deps)

    assert state.truncated_reason == "cancelled"
    assert sink.last("done").payload["truncated_reason"] == "cancelled"
    assert sink.of_type("error")


# --------------------------------------------------------------------------
# Protocol details that break the next request if we get them wrong
# --------------------------------------------------------------------------


def test_every_tool_call_the_model_made_gets_exactly_one_reply():
    """The trimming contract.

    MIA sometimes returns several tool calls at once. We keep the first, and we
    have to trim the stored assistant message the same way: an assistant
    message carrying a call id with no matching tool message makes the *next*
    request a protocol error, which surfaces as an unexplained 400 mid-demo.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_calls(
                [FETCH_RATE, SEARCH],
                ids=["c1", "c2"],
                content="Iki sey birden yapayim.",
            ),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    stored = assistant_tool_calls(state)
    assert len(stored) == 1
    ids_asked = [call["id"] for call in stored[0]["tool_calls"]]
    ids_answered = [m["tool_call_id"] for m in tool_messages(state)]
    assert ids_asked == ids_answered == ["c1"]


def test_a_cancelled_call_still_gets_its_tool_message():
    """Refusing to run a tool is still an answer the protocol requires."""

    async def refuse(request):
        return ApprovalDecision(request_id=request.request_id, choice="cancel")

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(
                "ingest", {"source": "evds", "series_id": "evds:TP.YENI"}, call_id="c1"
            ),
            FakeLLM.text("Peki."),
            FakeLLM.text("Kullanici cekimi iptal etti."),
        ],
        approve=refuse,
    )

    turn(state, deps)

    messages = tool_messages(state)
    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "c1"
    assert "iptal etti" in messages[0]["content"]
    assert state.tables == {}


def test_the_assistant_message_sent_back_carries_no_vllm_extras():
    """Only role, content and tool_calls go back to MIA.

    vLLM adds fields of its own to the message. Echoing them back has been seen
    to produce a 400 from the very same server that sent them.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert set(assistant_tool_calls(state)[0]) <= {"role", "content", "tool_calls"}


def test_thinking_is_on_where_judgement_is_needed_and_off_where_it_is_not():
    """Measured: thinking costs about four times the time and the tokens.

    Worth it while planning and while writing the answer. Wasteful while
    reading back a tool result the model already decided to ask for.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    planner, answer = load_prompt("planner").tag, load_prompt("answer").tag
    calls = [(call["tag"], call["thinking"]) for call in deps.llm.calls]
    assert calls == [
        (f"plan@{planner}", True),
        (f"observe@{planner}", False),
        (f"answer@{answer}", True),
    ]


def test_the_loop_takes_no_more_steps_than_the_script_expects():
    """`ScriptExhausted` is the assertion; this test makes it explicit."""
    state = initial_state("s1")
    deps = build([FakeLLM.tool_call(*FETCH_RATE, call_id="c1")])
    with pytest.raises(ScriptExhausted):
        turn(state, deps)


# --------------------------------------------------------------------------
# Approvals, in the loop rather than in isolation
# --------------------------------------------------------------------------


def test_an_ambiguous_search_stops_and_the_choice_comes_back_as_a_person_speaking():
    """The flow/stock trap, end to end."""

    async def person(request):
        return ApprovalDecision(
            request_id=request.request_id, choice=request.options[0].id
        )

    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*SEARCH, call_id="c1"),
            FakeLLM.text("Secimi aldim."),
            FakeLLM.text("Akim serisiyle devam ettim."),
        ],
        sink=sink,
        approve=person,
        ambiguous_search=True,
    )

    result = turn(state, deps)

    assert result.approvals == 1
    assert sink.last("approval_request").payload["kind"] == "series_choice"
    assert any(
        m.get("role") == "user" and "[Onay] Seri secimi" in m.get("content", "")
        for m in state.messages
    )


def test_a_series_choice_nobody_made_is_not_put_in_a_persons_mouth():
    """Without a person (CLI, eval) the card's default -- the top search hit --
    is applied and recorded, but the model is not told a person chose it. The
    search ranks by text similarity and on the real pool ranks non-performing
    housing loans above the loan stock; the model reading the candidates is
    the better judge, and "a person said so" would overrule it."""
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*SEARCH, call_id="c1"),
            FakeLLM.text("Adaylardan akim serisini seciyorum."),
            FakeLLM.text("Akim serisiyle devam ettim."),
        ],
        sink=sink,
        ambiguous_search=True,
    )

    result = turn(state, deps)

    assert result.approvals == 1
    assert state.approvals[0].automatic is True
    assert sink.last("approval_decision").payload["automatic"] is True
    assert not any(
        m.get("role") == "user" and "[Onay]" in m.get("content", "")
        for m in state.messages
    )


def test_a_table_rejected_at_the_quality_gate_does_not_stay_in_the_session():
    """Cancelling has to actually remove it, or the next turn analyses the very
    table a person just refused."""

    async def reject(request):
        return ApprovalDecision(request_id=request.request_id, choice="cancel")

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("quality_table", {}, call_id="c1"),
            FakeLLM.text("Peki."),
            FakeLLM.text("Tablo kalite nedeniyle iptal edildi."),
        ],
        approve=reject,
        missing_pct=40.0,
    )

    turn(state, deps)

    assert state.tables == {}
    assert state.active_table is None


def test_an_approval_that_never_comes_back_does_not_poison_the_session():
    """A closed browser or a timed-out card must not cost more than the step.

    This was the worst failure in the loop. The approval callback raising left
    the tool call unanswered, and because the message history carries across
    turns, every later question in that session would fail with an unexplained
    400 from MIA. Now it reads as a failed step and the session survives.
    """

    async def never_answers(request):
        raise TimeoutError("kullanici cevap vermedi")

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(
                "ingest", {"source": "evds", "series_id": "evds:TP.YENI"}, call_id="c1"
            ),
            FakeLLM.text("Peki."),
            FakeLLM.text("Onay alinamadigi icin cekim yapilmadi."),
        ],
        approve=never_answers,
    )

    result = turn(state, deps)

    assert result.answer
    assert orphan_call_ids(state) == []
    assert "Onay alinamadi" in tool_messages(state)[0]["content"]
    assert state.steps[1].error == "approval_failed"
    assert state.tables == {}


def test_a_tool_that_reports_failure_leaves_nothing_behind():
    """Same rule as a timed-out step, applied to an honest failure.

    A tool that writes a table and then returns `ok=False` would otherwise
    leave a half-built frame in the session for the next turn to analyse.
    """
    from lakehouse_agent.contracts import ToolResult
    from lakehouse_agent.testing.fake_tools import EmptyArgs
    from lakehouse_agent.tools.registry import ToolSpec

    def half_done(args, ctx):
        ref = ctx.new_table_ref("half")
        ctx.put_table(ref, pd.DataFrame({"period": [1]}))
        ctx.set_active(ref)
        return ToolResult(ok=False, summary="Yarida kaldi.", table_ref=ref)

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("half_tool", {}, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )
    deps.registry.add(
        ToolSpec(
            name="half_tool",
            description_tr="Yarida biten arac.",
            args_model=EmptyArgs,
            fn=half_done,
        )
    )

    turn(state, deps)

    assert state.tables == {}
    assert state.active_table is None


@pytest.mark.parametrize(
    "script_item, expected_error",
    [
        (FakeLLM.tool_call("sql_query", {}, call_id="c1"), "unknown_tool"),
        (
            FakeLLM.tool_call("get_series", {"start": "2021-01"}, call_id="c1"),
            "invalid_args",
        ),
        (FakeLLM.bad_json_tool_call("get_series", "{[", call_id="c1"), "invalid_args"),
    ],
)
def test_every_failed_step_still_draws_a_step_card(script_item, expected_error):
    """Bad arguments and invented tool names are what a jury will actually see.

    Without an end event the interface has a step card that never closes, so
    the screen simply stops moving at the moment something went wrong.
    """
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [script_item, FakeLLM.text("Tamam."), FakeLLM.text("Cevap.")], sink=sink
    )

    turn(state, deps)

    failed = [
        event for event in sink.of_type("step_end") if event.payload["ok"] is False
    ]
    assert len(failed) == 1
    assert failed[0].payload["error"] == expected_error
    # Opened and closed: an interface that never gets the end event is left
    # with a card that stops moving at the moment something went wrong.
    opened = {event.payload["step_no"] for event in sink.of_type("step_start")}
    closed = {event.payload["step_no"] for event in sink.of_type("step_end")}
    assert opened == closed


def test_a_preview_with_missing_values_still_serialises(capsys):
    """`diff` puts a NaN in the first row of every derived column.

    NaN is not valid JSON. It survives only because the payload goes through
    pydantic's json mode on the way out, which turns it into null; a plain
    `json.dumps` of the raw rows would emit `NaN` and any browser reading the
    stream would fail on it.
    """
    import json

    from lakehouse_agent.agent.loop import _cli

    assert _cli(["--fake", "--question", "fark"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    updates = [line for line in lines if line["type"] == "table_update"]
    first_row = updates[-1]["payload"]["rows"][0]

    assert first_row["TP_KTF12_diff"] is None
    assert "NaN" not in json.dumps(updates[-1], allow_nan=False)


def test_a_model_failure_ends_the_turn_honestly_instead_of_half_answering():
    """MIA goes down mid-turn. It will, at some point, in front of someone.

    The turn cannot be salvaged -- there is no answer without the model -- but
    it has to end as a stated failure with a `done` event, not as a traceback
    the interface has to guess at.
    """

    def raises(exc):
        def reply(messages, tools, thinking):
            raise exc

        return reply

    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            raises(LlmServerError("MIA 503")),
        ],
        sink=sink,
    )

    with pytest.raises(AgentError, match="MIA 503"):
        turn(state, deps)

    assert state.truncated_reason == "llm_error"
    assert sink.last("error").payload["kind"] == "LlmServerError"
    assert sink.types()[-1] == "done"
    # The step that did succeed is still in the session.
    assert state.tables


def test_no_prompt_placeholder_ever_reaches_the_model_unfilled():
    """`render` leaves an unknown `{{name}}` in place, visibly and on purpose.

    Visibly to us, that is -- if nobody looks, the model just reads the literal
    braces and quietly loses whatever that slot was supposed to tell it. Adding
    a placeholder to planner.md without wiring it up is a one-line change that
    nothing else would catch.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    for call in deps.llm.calls:
        system = call["messages"][0]["content"]
        assert "{{" not in system, f"{call['tag']} prompt'unda dolmamis yer tutucu var"


# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------


def test_three_turns_in_one_session_with_no_interruptions():
    """The demo, compressed.

    Turn one builds a table, turn two adds a derived column to it, turn three
    adds a second series to the same table. Nothing is rebuilt from scratch, no
    existing column changes, and the approval count is zero: the gates exist
    for genuine ambiguity, and this path has none.
    """
    state = initial_state("s1")
    settings_script = [
        # turn 1: find the series, fetch it, answer
        FakeLLM.tool_call(*SEARCH, call_id="c1"),
        FakeLLM.tool_call(*FETCH_RATE, call_id="c2"),
        FakeLLM.text("Hazir."),
        FakeLLM.text("Faiz serisi getirildi."),
        # turn 2: add a derived column without breaking the table
        FakeLLM.tool_call(
            "transform_table", {"op": "diff", "column": "TP_KTF12"}, call_id="c3"
        ),
        FakeLLM.text("Eklendi."),
        FakeLLM.text("Aylik fark sutunu eklendi."),
        # turn 3: add a second series to the same table
        FakeLLM.tool_call(
            "get_series",
            {
                "series_ids": ["bddk:aylik:tuketici:konut:TL:sektor"],
                "start": "2021-01",
                "end": "2025-12",
                "into": "active",
            },
            call_id="c4",
        ),
        FakeLLM.text("Eklendi."),
        FakeLLM.text("Kredi bakiyesi de ayni tabloda."),
    ]
    deps = build(settings_script)

    turn(state, deps, "Konut kredisi faizini getir.")
    before = state.tables["t01_get_series"].copy(deep=True)

    turn(state, deps, "Aylik farkini da ekle, tabloyu bozma.")
    after_two = state.tables["t01_get_series"]
    assert list(after_two.columns) == ["period", "TP_KTF12", "TP_KTF12_diff"]
    pd.testing.assert_frame_equal(after_two[list(before.columns)], before)

    turn(state, deps, "Kredi bakiyesini de ekle.")
    final = state.tables["t01_get_series"]
    assert list(final.columns) == [
        "period",
        "TP_KTF12",
        "TP_KTF12_diff",
        "aylik_tuketici_konut_TL_sektor",
    ]
    pd.testing.assert_frame_equal(final[list(before.columns)], before)

    assert state.approvals == []
    assert state.tables.keys() == {"t01_get_series"}
    assert state.active_table == "t01_get_series"
    assert state.turn_no == 3
    assert len({step.step_no for step in state.steps}) == len(state.steps)
    assert [source.key() for source in state.sources] == [
        "evds:TP.KTF12",
        "bddk:aylik:tuketici:konut:TL:sektor",
    ]


# --------------------------------------------------------------------------
# The headless entry point
# --------------------------------------------------------------------------


def test_the_fake_cli_writes_one_json_event_per_line_and_ends_with_done(capsys):
    """The done criterion for this stage, asserted rather than eyeballed."""
    import json

    from lakehouse_agent.agent.loop import _cli

    assert _cli(["--fake", "--question", "Konut kredisi faizi nasil seyretti?"]) == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert [line["type"] for line in lines[:2]] == ["step_start", "plan"]
    assert lines[-1]["type"] == "done"
    assert lines[-1]["payload"]["truncated_reason"] is None
    assert all(line["session_id"] == "cli" for line in lines)


def test_the_cli_makes_you_say_whether_you_want_the_real_model(capsys):
    """`--fake` and `--live` differ by whether real quota is spent.

    Defaulting to either one would be wrong: a default of live would make an
    innocent-looking command hit MIA, and a default of fake would let someone
    believe they had tested against the model when they had not.
    """
    from lakehouse_agent.agent.loop import _cli

    with pytest.raises(SystemExit) as exit_info:
        _cli([])
    assert exit_info.value.code == 2
    assert "--fake" in capsys.readouterr().err


def test_live_mode_says_what_is_missing_instead_of_traceback(capsys, monkeypatch):
    """The most likely first run: no key. It has to be a sentence, not a stack."""
    from lakehouse_agent.agent.loop import _cli

    monkeypatch.setattr(
        "lakehouse_agent.config.get_agent_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("MIA_API_KEY yok")),
    )
    assert _cli(["--live"]) == 2
    assert "MIA_API_KEY" in capsys.readouterr().err


def test_real_mode_says_what_is_missing_instead_of_traceback(capsys, monkeypatch):
    from lakehouse_agent.agent.loop import _cli

    # A developer's real .env must not leak into the test process.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(
        "lakehouse_agent.config.get_agent_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("MIA_API_KEY yok")),
    )
    assert _cli(["--real"]) == 2
    assert "MIA_API_KEY" in capsys.readouterr().err


def test_real_mode_wires_real_tools_and_a_lake_and_no_fakes(monkeypatch):
    """The demo's dependency set: the production registry, a real lake."""
    from lakehouse_agent.agent.loop import _real_deps
    from lakehouse_agent.config import AgentSettings
    from lakehouse_agent.tools.lake_adapter import LakeAdapter

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(
        "lakehouse_agent.config.get_agent_settings",
        lambda: AgentSettings(MIA_API_KEY="test-only-not-a-real-key"),
    )

    deps = _real_deps()

    assert isinstance(deps.lake, LakeAdapter)
    assert {"search_catalog", "get_series", "transform_table"} <= set(
        deps.registry.names()
    )
    assert "failing_tool" not in deps.registry.names()


def test_several_questions_are_turns_of_one_session(capsys):
    """A follow-up like "tabloyu bozmadan ..." only works if it sees the table
    the previous question built."""
    from lakehouse_agent.agent.loop import _cli

    assert _cli(["--fake", "--question", "bir", "--question", "iki"]) == 0

    out, err = capsys.readouterr()
    done = [json.loads(line) for line in out.splitlines()]
    done = [event for event in done if event["type"] == "done"]
    assert len(done) == 2
    # Only one shared session produces these: `usage` in `done` is the
    # session total, and table numbers count up across its turns.
    assert [event["payload"]["usage"]["calls"] for event in done] == [5, 10]
    assert "t02_get_series" in err


def test_the_recap_prints_every_address_in_full(capsys):
    """An answer carries whatever address the model wrote, and a model
    shortens a long one. The list beside it is built from the session, so a
    person watching sees the address they can open."""
    from datetime import UTC, datetime

    from lakehouse_agent.agent.loop import _summarise
    from lakehouse_agent.contracts import Source

    url = "https://www.tcmb.gov.tr/wps/wcm/connect/tr/duyurular/2023/duy2023-25"
    state = initial_state("cli")
    state.turn_no = 1
    state.sources = [
        Source(
            series_id="evds:TP.KTF12",
            name="Konut Kredisi",
            fetched_at=datetime(2026, 9, 22, tzinfo=UTC),
        ),
        Source(
            url=url, name="TCMB duyurusu", fetched_at=datetime(2026, 9, 22, tzinfo=UTC)
        ),
    ]

    _summarise(state, None)

    err = capsys.readouterr().err
    assert "KAYNAKLAR (bakilan)" in err
    assert url in err  # not shortened
    assert "evds:TP.KTF12" in err and "Konut Kredisi" in err


def test_a_session_with_no_sources_prints_no_heading(capsys):
    from lakehouse_agent.agent.loop import _summarise

    state = initial_state("cli")
    state.turn_no = 1

    _summarise(state, None)

    assert "KAYNAKLAR" not in capsys.readouterr().err


def test_a_failed_turn_still_shows_what_it_built(capsys):
    """The 22 Sep live run ended on a bare error line although its table was
    built. The recap now prints the tables either way."""
    from lakehouse_agent.agent.loop import _summarise

    state = initial_state("cli")
    state.turn_no = 1
    state.tables["t01_get_series"] = pd.DataFrame(
        {"period": pd.date_range("2021-01-01", periods=3, freq="MS"), "x": [1, 2, 3]}
    )
    state.active_table = "t01_get_series"

    _summarise(state, None)

    err = capsys.readouterr().err
    assert "TABLO t01_get_series: 3 satir" in err
    assert "CEVAP YOK" in err


# --------------------------------------------------------------------------
# What a rejected table takes with it
# --------------------------------------------------------------------------


def _reject_quality(request):
    async def answer(req):
        if req.kind == "quality":
            return ApprovalDecision(request_id=req.request_id, choice="cancel")
        return ApprovalDecision(request_id=req.request_id, choice=req.default)

    return answer


QUALITY = ("quality_table", {})


def test_a_rejected_table_takes_its_lineage_citation_and_chart_with_it():
    """Popping the frame was only the visible part of the rollback.

    What stayed behind mattered more. `answer.md` tells the model to cite from
    `state.sources` and nothing else, so a series left in that list after the
    person rejected it can be named in the answer as if it had been used. The
    lineage rows were the same problem for the traceability panel.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.tool_call(*QUALITY),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        approve=_reject_quality(None),
    )

    turn(state, deps)

    assert "t02_quality" not in state.tables
    assert [key for key in state.lineage if key.startswith("t02_quality.")] == []
    assert all("SUPHELI" not in source.key() for source in state.sources)
    assert all(chart.table_ref != "t02_quality" for chart in state.charts)
    # Back to the table the person was working on, not to nothing.
    assert state.active_table == "t01_get_series"


def test_a_rejected_table_does_not_hand_its_number_to_the_next_one():
    """`decisions.md` records max-based numbering as the fix for a deleted
    table freeing its number, and names the quality gate as the deleter.

    The gate always rejects the newest table, which is exactly the number the
    scan counts from, so the recorded fix did not cover its own motivating
    case: the next table took the rejected one's name and inherited its
    lineage rows.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.tool_call(*QUALITY),
            FakeLLM.tool_call(*QUALITY),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        approve=_reject_quality(None),
    )

    turn(state, deps)

    # Both were rejected, so both are retired -- and the second one got its
    # own number rather than stepping into the first one's.
    assert state.retired_tables == {"t02_quality", "t03_quality"}
    assert not any(ref.startswith("t02_") for ref in state.tables)


def test_the_interface_is_told_the_table_went_away():
    """A card the person cancelled has to disappear from the screen too."""
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.tool_call(*QUALITY),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
        approve=_reject_quality(None),
    )

    turn(state, deps)

    removals = [
        event for event in sink.of_type("table_update") if event.payload.get("removed")
    ]
    assert [event.payload["table_ref"] for event in removals] == ["t02_quality"]
    assert removals[0].payload["reason"] == "quality_cancel"


# --------------------------------------------------------------------------
# Warnings belong to the turn that produced them
# --------------------------------------------------------------------------


def test_a_failure_in_one_turn_is_not_reported_as_live_in_the_next():
    """`state.warnings` is cleared every turn and `state.steps` is not, so the
    two halves of the answer's warning list described different windows.

    `answer.md` rule 8 tells the model to open with what went wrong, so turn 3
    apologised for a tool that failed in turn 1.
    """
    from lakehouse_agent.agent.loop import _collected_warnings

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("failing_tool", {}),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Ilk cevap."),
        ]
    )
    turn(state, deps)
    assert any("failing_tool" in warning for warning in _collected_warnings(state))

    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Ikinci."),
        ]
    )
    turn(state, deps, question="Peki enflasyon?")

    assert _collected_warnings(state) == []


def test_a_unit_conversion_is_not_pushed_out_by_later_warnings():
    """`get_series` puts its conversions first. Keeping the last four let one
    call's coverage notes and a later deflate push them out of the answer
    prompt, although the answer is obliged to repeat them (CLAUDE.md #3)."""
    from lakehouse_agent.agent.loop import MAX_ANSWER_WARNINGS, _collected_warnings

    state = initial_state("s1")
    state.warnings = [
        "TP_KTF12 haftalik yayimlaniyor; aylik degere ortalama alinarak cevrildi.",
        *[f"uyari {i}" for i in range(MAX_ANSWER_WARNINGS + 1)],
    ]

    collected = _collected_warnings(state)

    assert collected[0].startswith("TP_KTF12 haftalik")
    assert len(collected) == MAX_ANSWER_WARNINGS + 1
    assert collected[-1].startswith("(+2 uyari daha")


def test_a_tool_warning_reaches_the_answer_prompt_in_its_own_turn():
    """The route CLAUDE.md #3 depends on: a conversion the answer must repeat."""
    from lakehouse_agent.agent.loop import _collected_warnings

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_CREDIT),
            FakeLLM.tool_call(
                "transform_table",
                {"table_ref": "t01_get_series", "column": CREDIT, "op": "deflate"},
            ),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert any("2025=100" in warning for warning in _collected_warnings(state))


# --------------------------------------------------------------------------
# A failed step publishes nothing
# --------------------------------------------------------------------------


def test_a_failed_step_does_not_publish_a_table_a_citation_or_a_chart():
    """The rule `_merge_context` states in its own docstring, applied to the
    event channel as well as to the session.

    Publishing anyway left the interface showing a provenance entry for data
    the session had refused to keep.
    """
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("failing_tool", {}),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
    )

    turn(state, deps)

    assert sink.of_type("table_update") == []
    assert sink.of_type("sources") == []
    assert sink.of_type("quality_report") == []
    assert sink.of_type("chart") == []


# --------------------------------------------------------------------------
# The channel has to stay honest when things go wrong
# --------------------------------------------------------------------------


def test_an_unexpected_crash_is_reported_as_an_error_not_as_a_clean_finish():
    """Only `LlmError` was handled, so any other crash reached the `finally`
    and published `done` with `truncated_reason: null` -- which an interface
    reads as a turn that finished successfully."""
    sink = ListSink()
    state = initial_state("s1")
    deps = build([FakeLLM.text("Plan."), FakeLLM.text("Cevap.")], sink=sink)

    def explode(*args, **kwargs):
        raise RuntimeError("beklenmedik")

    deps.llm.chat = explode

    with pytest.raises(RuntimeError):
        turn(state, deps)

    assert sink.of_type("error")
    assert sink.last("done").payload["truncated_reason"] == "llm_error"


def test_a_verifier_that_crashes_does_not_cost_the_turn_its_answer():
    """The one crash that must not propagate. The answer is already written
    when the verifier runs, and throwing away finished work to report a doubt
    is the worst trade in the turn."""
    sink = ListSink()
    state = initial_state("s1")
    deps = build([FakeLLM.text("Plan."), FakeLLM.text("Cevap.")], sink=sink)

    def explode(*args, **kwargs):
        raise RuntimeError("beklenmedik")

    deps.verifier.verify = explode

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert result.truncated_reason is None
    assert sink.last("verification").payload["passed"] is True


def test_a_model_that_returns_no_answer_is_not_a_finished_turn():
    """The failure `AgentConfig` names in its own docstring. Two empties: the
    first is retried with thinking off, and a model that answers nothing
    twice has nothing to say."""
    state = initial_state("s1")
    deps = build([FakeLLM.text("Plan."), FakeLLM.text(""), FakeLLM.text("")])

    result = turn(state, deps)

    assert result.truncated_reason == "llm_error"
    assert "cevap uretmedi" in result.answer


def test_an_empty_answer_is_asked_for_once_more_without_thinking():
    """Measured 23 Sep on a turn whose tools had all succeeded: the answer
    call spent 374 completion tokens on reasoning and wrote no content at
    all, and the person got "cevap uretmedi" under a finished analysis.
    Thinking off is a different request, with no reasoning channel for the
    answer to disappear into."""
    state = initial_state("s1")
    deps = build([FakeLLM.text("Plan."), FakeLLM.text(""), FakeLLM.text("Cevap.")])

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert result.truncated_reason is None
    retry = deps.llm.calls[-1]
    assert retry["thinking"] is False


def test_the_delta_stream_joins_back_into_the_answer():
    """Rebuilding each chunk with `strip()` ate one space per boundary, so a
    consumer concatenating the deltas got two words glued together."""
    from lakehouse_agent.agent.loop import _sentences

    text = " ".join(f"kelime{i:03d}" for i in range(60))
    chunks = _sentences(text)

    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_the_turn_reports_its_own_cost_not_the_session_total():
    """Reporting the running total made every turn look more expensive than
    the one before it."""
    state = initial_state("s1")
    first = turn(state, build([FakeLLM.text("Plan."), FakeLLM.text("Bir.")]))
    second = turn(
        state, build([FakeLLM.text("Plan."), FakeLLM.text("Iki.")]), question="Peki?"
    )

    assert second.usage.calls == first.usage.calls
    assert state.usage.calls == first.usage.calls + second.usage.calls


# --------------------------------------------------------------------------
# Charts: three carriers, two output channels
# --------------------------------------------------------------------------


def _chart_tool(chart):
    from pydantic import BaseModel

    from lakehouse_agent.contracts import ToolResult
    from lakehouse_agent.tools.registry import ToolSpec

    class NoArgs(BaseModel):
        pass

    def run(args, ctx):
        ctx.add_chart(chart)
        return ToolResult(ok=True, summary="Grafik hazir.")

    return ToolSpec(
        name="make_chart", description_tr="Grafik.", args_model=NoArgs, fn=run
    )


@pytest.mark.parametrize(
    "chart",
    [
        pytest.param({"chart_ref": "c01", "table_ref": "t01_a"}, id="dict"),
        pytest.param(
            {"chart_ref": "c01", "table_ref": "t01_a", "plotly_json": {"x": None}},
            id="dict-with-figure",
        ),
    ],
)
def test_a_chart_carried_as_a_plain_dict_does_not_kill_the_turn(chart):
    """`ToolContext.add_chart` is annotated `Any`, `test_context.py` pins that a
    dict lands in the buffer, and `state.chart_ref_of` exists specifically to
    read `ref` as well as `chart_ref`.

    The loop was the one place reading the narrow name, so the carriers the
    codebase advertises as accepted crashed `_answer` with a raw traceback --
    after `done` had already been emitted.
    """
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("make_chart", {}),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
    )
    deps.registry.add(_chart_tool(chart))

    result = turn(state, deps)

    assert result.answer == "Cevap."
    assert sink.last("chart").payload["chart_ref"] == "c01"


def test_a_figure_full_of_numpy_does_not_kill_the_turn():
    """The field is named after a library whose figures do not serialise: an
    ordinary `px.line` holds numpy arrays, and the exception escaped the
    emitter past every handler."""
    import numpy as np

    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("make_chart", {}),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
    )
    deps.registry.add(
        _chart_tool(
            {
                "chart_ref": "c01",
                "table_ref": "t01_a",
                "plotly_json": {"data": [{"y": np.array([1.0, 2.0])}]},
            }
        )
    )

    result = turn(state, deps)

    import json

    assert result.answer == "Cevap."
    json.dumps(sink.last("chart").model_dump(mode="json"))


def test_a_chart_handed_back_through_the_shared_contract_is_not_lost():
    """`ToolResult.artifacts` is one of the eight types both lanes build
    against, and it was the output channel nothing read."""
    from pydantic import BaseModel

    from lakehouse_agent.contracts import Artifact, ToolResult
    from lakehouse_agent.tools.registry import ToolSpec

    class NoArgs(BaseModel):
        pass

    def run(args, ctx):
        return ToolResult(
            ok=True,
            summary="Grafik hazir.",
            artifacts=[Artifact(type="chart", ref="c01", title="Faiz")],
        )

    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call("make_chart", {}),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ],
        sink=sink,
    )
    deps.registry.add(
        ToolSpec(name="make_chart", description_tr="G.", args_model=NoArgs, fn=run)
    )

    turn(state, deps)

    assert [chart.ref for chart in state.charts] == ["c01"]
    assert sink.last("chart").payload["ref"] == "c01"


def test_fetching_into_the_active_table_refuses_to_rename_a_column_it_finds():
    """`merge` renames a colliding column to `_x`/`_y`, so the column the user
    was looking at silently changes name -- and the loop's "existing columns
    came back unchanged" check cannot see it, because the old name is no longer
    in the frame to compare against.
    """
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.tool_call(
                "get_series",
                {
                    "series_ids": ["evds:TP.KTF12"],
                    "start": "2021-01",
                    "end": "2025-12",
                    "into": "active",
                },
            ),
            FakeLLM.text("Tamam."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert list(state.tables["t01_get_series"].columns) == ["period", "TP_KTF12"]
    assert not any("_x" in str(column) for column in state.tables["t01_get_series"])
    assert "zaten var" in tool_messages(state)[-1]["content"]


# --------------------------------------------------------------------------
# The observing call's output cap
# --------------------------------------------------------------------------


def _chat_calls(deps, node):
    return [
        call
        for call in deps.llm.calls
        if call["kind"] == "chat" and str(call["tag"] or "").startswith(node)
    ]


def test_only_the_observing_call_is_capped():
    """Observing says what to do next in a sentence; it is not where the
    answer is written (planner rule 14). Planning and answering are not
    capped: a cap low enough to stop prose would cut a plan's tool call or
    an answer in half."""
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.text("Veriler hazir."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    assert [call["max_tokens"] for call in _chat_calls(deps, "observe")] == [
        deps.config.observe_max_tokens
    ]
    assert [call["max_tokens"] for call in _chat_calls(deps, "plan")] == [None]
    assert [call["max_tokens"] for call in _chat_calls(deps, "answer")] == [None]


def test_a_tool_call_cut_off_by_the_cap_is_asked_for_again_uncapped():
    """Trimming prose is the point; trimming an action is not. A capped
    observing call that came back mid-tool-call is redone with no cap, and
    the loop runs the tool it then asks for."""
    truncated = FakeLLM.tool_call(*FETCH_CREDIT, call_id="cut")
    truncated = truncated.model_copy(update={"finish_reason": "length"})
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            truncated,
            FakeLLM.tool_call(*FETCH_CREDIT, call_id="whole"),
            FakeLLM.text("Veriler hazir."),
            FakeLLM.text("Cevap."),
        ]
    )

    turn(state, deps)

    observing = _chat_calls(deps, "observe")
    assert [call["max_tokens"] for call in observing] == [
        deps.config.observe_max_tokens,
        None,  # asked again, uncapped
        deps.config.observe_max_tokens,
    ]
    assert [
        call["id"]
        for message in assistant_tool_calls(state)
        for call in message["tool_calls"]
    ] == ["call_1", "whole"]
    assert not orphan_call_ids(state)


def test_the_empty_answer_retry_still_has_time_late_in_a_turn():
    """`allowance` is recomputed from `active_s`, which the first attempt
    already increased, and `left()` subtracted that attempt a second time.
    Measured: at 125 s spent the retry was handed -15 s, which `llm._call`
    turns into `LlmTimeoutError` and `_answer` suppresses -- so main's
    empty-answer retry silently did not happen on the late turns it exists
    for. Both retry sites share one `_Spent`, so both were affected."""
    clock = _Clock()
    clock.now = 125.0
    llm = FakeLLM([FakeLLM.text("", duration_ms=45_000), FakeLLM.text("Cevap.")])
    deps = build([], clock=clock)
    deps.llm = llm
    state = initial_state("s1")
    state.active_s = 125.0
    spent = loop_module._Spent()

    async def two_calls():
        await loop_module._chat(
            state,
            deps,
            [{"role": "user", "content": "x"}],
            tools=None,
            thinking=True,
            tag="answer",
            spent=spent,
            last_chance=True,
        )
        clock.now += 45.0
        await loop_module._chat(
            state,
            deps,
            [{"role": "user", "content": "x"}],
            tools=None,
            thinking=False,
            tag="answer-retry",
            spent=spent,
            last_chance=True,
        )

    asyncio.run(two_calls())

    retry = llm.calls[-1]["timeout_s"]
    assert retry >= MIN_ATTEMPT_S, f"yeniden deneme {retry} sn aldi"
