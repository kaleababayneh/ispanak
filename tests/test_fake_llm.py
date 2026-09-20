"""Tests for the scripted model stand-in.

`FakeLLM` is test infrastructure, not production code, which is exactly why it
needs its own tests: every loop test asserts what the loop did *in response to
a scripted reply*, so a reply that is shaped wrong makes those tests agree with
each other about something untrue. The replies built here are checked against
the same `parse_tool_calls` / `as_message` code that runs against real MIA.
"""

import json

import pytest

from lakehouse_agent.contracts import Usage
from lakehouse_agent.testing import FakeLLM, ScriptExhausted

# --- the script ------------------------------------------------------------


def test_replies_come_back_in_script_order():
    llm = FakeLLM([FakeLLM.text("bir"), FakeLLM.text("iki")])
    assert llm.chat([{"role": "user", "content": "?"}]).content == "bir"
    assert llm.chat([{"role": "user", "content": "?"}]).content == "iki"


def test_running_off_the_end_raises_instead_of_defaulting():
    """The loop calling the model once more than a test scripted is the bug we
    most want to catch, so it must fail loudly rather than get a default."""
    llm = FakeLLM([FakeLLM.text("tek")])
    llm.chat([{"role": "user", "content": "?"}])
    with pytest.raises(ScriptExhausted):
        llm.chat([{"role": "user", "content": "?"}])


def test_empty_script_raises_on_the_first_call():
    with pytest.raises(ScriptExhausted):
        FakeLLM().chat([{"role": "user", "content": "?"}])


def test_a_callable_script_item_sees_what_the_loop_sent():
    seen = {}

    def reply(messages, tools, thinking):
        seen["messages"] = messages
        seen["tools"] = tools
        seen["thinking"] = thinking
        return FakeLLM.text("tamam")

    llm = FakeLLM([reply])
    result = llm.chat(
        [{"role": "user", "content": "soru"}], tools=[{"name": "t"}], thinking=False
    )
    assert result.content == "tamam"
    assert seen["messages"][0]["content"] == "soru"
    assert seen["tools"] == [{"name": "t"}]
    assert seen["thinking"] is False


# --- the call log ----------------------------------------------------------


def test_every_call_is_recorded_with_its_arguments():
    llm = FakeLLM([FakeLLM.text("x")])
    llm.chat(
        [{"role": "user", "content": "q"}],
        thinking=False,
        tag="plan",
        session_id="s1",
    )
    call = llm.calls[-1]
    assert call["kind"] == "chat"
    assert call["thinking"] is False
    assert call["tag"] == "plan"
    assert call["session_id"] == "s1"


def test_recorded_messages_are_copies_not_live_references():
    """A test asserting on `calls` after the loop mutated its own message list
    would otherwise read the mutated version and quietly agree with itself."""
    llm = FakeLLM([FakeLLM.text("x")])
    messages = [{"role": "user", "content": "ilk"}]
    llm.chat(messages)
    messages[0]["content"] = "sonradan degisti"
    assert llm.calls[-1]["messages"][0]["content"] == "ilk"


# --- reply constructors ----------------------------------------------------


def test_text_has_no_tool_calls_and_carries_usage():
    result = FakeLLM.text("cevap", prompt_tokens=7, completion_tokens=3)
    assert result.content == "cevap"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage == Usage(
        prompt_tokens=7, completion_tokens=3, calls=1, duration_ms=5
    )


def test_tool_call_is_shaped_like_a_real_one():
    result = FakeLLM.tool_call("get_series", {"series_ids": ["evds:TP.KTF12"]})
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "get_series"
    assert call.arguments == {"series_ids": ["evds:TP.KTF12"]}
    assert call.parse_error is None
    assert result.finish_reason == "tool_calls"
    # The raw message must stay OpenAI-shaped: the loop echoes it back verbatim.
    raw = result.raw_message["tool_calls"][0]
    assert raw["type"] == "function"
    assert json.loads(raw["function"]["arguments"]) == {"series_ids": ["evds:TP.KTF12"]}


def test_several_tool_calls_are_trimmed_in_both_places():
    """The same rule `parse_tool_calls` applies to MIA: one call survives, and
    the echoed message is trimmed to match. An assistant message carrying a
    tool_call with no matching tool message is a protocol error next turn."""
    result = FakeLLM.tool_calls([("a", {}), ("b", {})])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "a"
    assert len(result.raw_message["tool_calls"]) == 1


def test_bad_json_tool_call_reports_instead_of_raising():
    result = FakeLLM.bad_json_tool_call("get_series", "{bozuk")
    call = result.tool_calls[0]
    assert call.arguments is None
    assert call.parse_error
    assert call.arguments_raw == "{bozuk"


def test_structured_fills_parsed_and_content():
    result = FakeLLM.structured({"durum": "tamam"})
    assert result.parsed == {"durum": "tamam"}
    assert json.loads(result.content) == {"durum": "tamam"}


def test_as_message_round_trips_for_every_constructor():
    """`as_message()` is what the loop puts back into `messages`, so each
    constructor has to survive it with the role and the tool calls intact."""
    assert FakeLLM.text("x").as_message() == {"role": "assistant", "content": "x"}

    message = FakeLLM.tool_call("t", {"k": 1}).as_message()
    assert message["role"] == "assistant"
    assert len(message["tool_calls"]) == 1

    assert FakeLLM.structured({"a": 1}).as_message()["content"] == '{"a": 1}'


# --- embed / ocr -----------------------------------------------------------


def test_embed_is_deterministic_and_text_dependent():
    llm = FakeLLM()
    assert llm.embed(["konut"]) == llm.embed(["konut"])
    assert llm.embed(["konut"]) != llm.embed(["kredi"])


def test_embed_returns_one_unit_vector_per_text():
    llm = FakeLLM(embed_dim=8)
    vectors = llm.embed(["a", "b", "c"])
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == 8
        assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_embed_of_nothing_is_empty():
    assert FakeLLM().embed([]) == []


def test_ocr_returns_the_configured_text():
    llm = FakeLLM(ocr_text="belge metni")
    result = llm.ocr([b"png"])
    assert result.text == "belge metni"
    assert llm.calls[-1] == {"kind": "ocr", "images": 1, "prompt": None, "tag": None}
