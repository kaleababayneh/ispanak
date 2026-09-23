"""What the model is shown of earlier turns (agent/history.py).

Earlier turns shrink to their question and final answer; the current turn is
shown whole; nothing is removed from the session itself.
"""

from __future__ import annotations

import asyncio

from lakehouse_agent.agent.history import UNANSWERED, model_messages
from lakehouse_agent.agent.loop import AgentDeps, run_turn
from lakehouse_agent.agent.state import initial_state
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.testing.fake_tools import make_fake_registry

FETCH = (
    "get_series",
    {"series_ids": ["evds:TP.KTF12"], "start": "2021-01", "end": "2025-12"},
)


def _tool_turn(state, question, call_id, answer):
    """One finished turn as the loop leaves it in `messages`."""
    state.turn_starts.append(len(state.messages))
    state.messages += [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "Getiriyorum.",
            "tool_calls": [{"id": call_id, "type": "function", "function": {}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "60 satir ..."},
        {"role": "assistant", "content": "Veriler hazir."},
    ]
    if answer is not None:
        state.messages.append({"role": "assistant", "content": answer})
        state.answer_indices.append(len(state.messages) - 1)


def test_a_single_turn_is_shown_whole():
    state = initial_state("s1")
    _tool_turn(state, "Soru 1", "c1", "Cevap 1")

    assert model_messages(state) == state.messages


def test_an_earlier_turn_is_reduced_to_its_question_and_answer():
    state = initial_state("s1")
    _tool_turn(state, "Soru 1", "c1", "Cevap 1")
    _tool_turn(state, "Soru 2", "c2", None)

    view = model_messages(state)

    assert view[:2] == [
        {"role": "user", "content": "Soru 1"},
        {"role": "assistant", "content": "Cevap 1"},
    ]
    # The current turn is untouched, its tool exchange included.
    assert view[2:] == state.messages[state.turn_starts[-1] :]


def test_no_tool_call_of_an_earlier_turn_is_left_unanswered():
    """Dropping a tool message but keeping the call asking for it would make
    the next request a protocol error; whole turns go, or nothing does."""
    state = initial_state("s1")
    _tool_turn(state, "Soru 1", "c1", "Cevap 1")
    _tool_turn(state, "Soru 2", "c2", "Cevap 2")
    _tool_turn(state, "Soru 3", "c3", None)

    view = model_messages(state)
    asked = {c["id"] for m in view for c in m.get("tool_calls") or []}
    answered = {m["tool_call_id"] for m in view if m.get("role") == "tool"}

    assert asked == answered == {"c3"}


def test_an_earlier_turn_without_an_answer_says_so():
    state = initial_state("s1")
    _tool_turn(state, "Soru 1", "c1", None)
    _tool_turn(state, "Soru 2", "c2", None)

    assert model_messages(state)[1] == {"role": "assistant", "content": UNANSWERED}


def test_the_session_keeps_every_message():
    state = initial_state("s1")
    _tool_turn(state, "Soru 1", "c1", "Cevap 1")
    _tool_turn(state, "Soru 2", "c2", None)
    before = [dict(m) for m in state.messages]

    model_messages(state)

    assert state.messages == before


def test_the_second_turn_does_not_resend_the_first_turns_tool_output():
    """End to end through the loop: what the model receives on turn two."""
    deps = AgentDeps(
        llm=FakeLLM(
            [
                FakeLLM.tool_call(*FETCH, call_id="c1"),
                FakeLLM.text("Hazir."),
                FakeLLM.text("Cevap 1."),
                FakeLLM.text("Tablo zaten var."),
                FakeLLM.text("Cevap 2."),
            ]
        ),
        registry=make_fake_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
    )
    state = initial_state("s1")
    asyncio.run(run_turn(state, deps, "Faizi getir."))
    asyncio.run(run_turn(state, deps, "Peki son ay?"))

    second_turn_plan = deps.llm.calls[3]["messages"]
    roles = [m["role"] for m in second_turn_plan]
    assert roles == ["system", "user", "assistant", "user"]
    assert second_turn_plan[2]["content"] == "Cevap 1."
    # The table the first turn built is still described to the model.
    assert "t01_get_series" in second_turn_plan[0]["content"]


def test_a_reply_whose_call_is_out_of_view_is_not_sent():
    """The converse of the rule above: a tool message answering a call the
    request does not contain is the same protocol error from the other side."""
    state = initial_state("s1")
    _tool_turn(state, "Soru 1", "c1", None)
    _tool_turn(state, "Soru 2", "c2", None)
    # A reply to c1 that ended up in the current turn's stretch.
    state.messages.append({"role": "tool", "tool_call_id": "c1", "content": "gec"})

    view = model_messages(state)

    assert [m.get("tool_call_id") for m in view if m.get("role") == "tool"] == ["c2"]


def test_a_turn_that_died_mid_call_is_repaired_inside_that_turn():
    """Turn 1 stopped between the model asking for a tool and the tool
    answering. Turn 2 must neither send the call's reply without the call nor
    tell its own answer that one of *its* tools did not run."""
    from lakehouse_agent.agent.loop import _collected_warnings

    state = initial_state("s1")
    state.turn_no = 1
    state.turn_starts.append(0)
    state.messages += [
        {"role": "user", "content": "Soru 1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "ingest"}}
            ],
        },
    ]
    deps = AgentDeps(
        llm=FakeLLM([FakeLLM.text("Arac gerekmiyor."), FakeLLM.text("Cevap 2.")]),
        registry=make_fake_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
    )

    asyncio.run(run_turn(state, deps, "Soru 2"))

    reply = next(
        i for i, m in enumerate(state.messages) if m.get("tool_call_id") == "c1"
    )
    assert reply < state.turn_starts[-1]  # inside turn 1, not turn 2
    for call in deps.llm.calls:
        asked = {c["id"] for m in call["messages"] for c in m.get("tool_calls") or []}
        answered = {
            m["tool_call_id"] for m in call["messages"] if m.get("role") == "tool"
        }
        assert answered <= asked
    repair = next(step for step in state.steps if step.tool == "ingest")
    assert repair.turn_no == 1
    assert not any("ingest" in warning for warning in _collected_warnings(state))
