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

import pandas as pd
import pytest

from lakehouse_agent.agent.events import ListSink
from lakehouse_agent.agent.loop import AgentConfig, AgentDeps, AgentError, run_turn
from lakehouse_agent.agent.state import ApprovalDecision, initial_state
from lakehouse_agent.llm import LlmServerError
from lakehouse_agent.testing.fake_llm import FakeLLM, ScriptExhausted
from lakehouse_agent.testing.fake_tools import make_fake_registry

SEARCH = ("search_catalog", {"query": "konut kredisi faiz orani"})
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
            FakeLLM.tool_call(*FETCH_RATE, call_id="c1"),
            FakeLLM.tool_call(
                "transform_table", {"op": "deflate", "column": "TP_KTF12"}, call_id="c2"
            ),
            FakeLLM.text("Hazir."),
            FakeLLM.text("Reel faiz hesaplandi."),
        ],
        sink=sink,
    )

    turn(state, deps)

    frame = state.tables["t01_get_series"]
    assert list(frame.columns) == ["period", "TP_KTF12", "reel_TP_KTF12"]
    assert state.active_table == "t01_get_series"
    assert sink.of_type("table_update")[-1].payload["new_columns"] == ["reel_TP_KTF12"]

    column = state.lineage["t01_get_series.reel_TP_KTF12"]
    assert column.op == "deflate"
    assert column.inputs == ["TP_KTF12"]
    assert column.source_series == ["evds:TP.KTF12", "evds:TP.TUKFIY2025.GENEL"]


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

    calls = [(call["tag"], call["thinking"]) for call in deps.llm.calls]
    assert calls == [
        ("plan@planner@1", True),
        ("observe@planner@1", False),
        ("answer@answer@1", True),
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
    sink = ListSink()
    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*SEARCH, call_id="c1"),
            FakeLLM.text("Secimi aldim."),
            FakeLLM.text("Akim serisiyle devam ettim."),
        ],
        sink=sink,
        ambiguous_search=True,
    )

    result = turn(state, deps)

    assert result.approvals == 1
    assert sink.last("approval_request").payload["kind"] == "series_choice"
    assert any(
        m.get("role") == "user" and "[Onay] Seri secimi" in m.get("content", "")
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


def test_a_tool_warning_reaches_the_answer_prompt_in_its_own_turn():
    """The route CLAUDE.md #3 depends on: a conversion the answer must repeat."""
    from lakehouse_agent.agent.loop import _collected_warnings

    state = initial_state("s1")
    deps = build(
        [
            FakeLLM.tool_call(*FETCH_RATE),
            FakeLLM.tool_call(
                "transform_table",
                {"table_ref": "t01_get_series", "column": "TP_KTF12", "op": "deflate"},
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

    deps.verifier.verify = explode

    with pytest.raises(RuntimeError):
        turn(state, deps)

    assert sink.of_type("error")
    assert sink.last("done").payload["truncated_reason"] == "llm_error"


def test_a_model_that_returns_no_answer_is_not_a_finished_turn():
    """The failure `AgentConfig` names in its own docstring."""
    state = initial_state("s1")
    result = turn(state, build([FakeLLM.text("Plan."), FakeLLM.text("")]))

    assert result.truncated_reason == "llm_error"
    assert "cevap uretmedi" in result.answer


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
