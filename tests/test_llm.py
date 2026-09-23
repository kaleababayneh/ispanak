"""Tests for the MIA client.

No network: a fake OpenAI client captures the kwargs we send and returns
canned responses. What is asserted here is the request *shape* we promise MIA
and the behaviour the agent loop depends on.
"""

import json
from types import SimpleNamespace

import openai
import pytest

from lakehouse_agent.config import AgentSettings
from lakehouse_agent.llm import (
    MAX_IMAGES_PER_CHAT,
    OCR_EXTRA_BODY_MULTI,
    OCR_EXTRA_BODY_SINGLE,
    OCR_PROMPT,
    ChatResult,
    LlmBadRequestError,
    LlmConfigError,
    LlmInputError,
    LlmRateLimitError,
    LlmServerError,
    LlmTimeoutError,
    MiaLLM,
    count_images,
    image_part,
)

# --- fakes ----------------------------------------------------------------


def _message(content=None, tool_calls=None):
    """A stand-in for the SDK's message object: needs .model_dump and attributes."""
    payload = {"role": "assistant", "content": content}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return SimpleNamespace(
        content=content,
        model_dump=lambda exclude_none=False: {
            k: v for k, v in payload.items() if not exclude_none or v is not None
        },
    )


def _completion(
    content=None, tool_calls=None, finish_reason="stop", prompt=10, completion=5
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=_message(content, tool_calls), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        model="kkbhackathon2026/Qwen3.8-27B",
    )


# The SDK raises a specific subclass per status code, and our error mapping and
# usage log both record the class name, so the fakes must do the same.
_STATUS_CLASSES: dict[int, type[openai.APIStatusError]] = {
    400: openai.BadRequestError,
    401: openai.AuthenticationError,
    403: openai.PermissionDeniedError,
    404: openai.NotFoundError,
    429: openai.RateLimitError,
}


def _status_error(code: int) -> openai.APIStatusError:
    # A stub is enough: the SDK only reads status_code off the response here.
    # Importing openai's HTTP client directly would tie this file to an
    # undeclared transitive dependency, and the day openai swaps that client
    # every test here would vanish with an import error instead of failing.
    response = SimpleNamespace(status_code=code, headers={}, request=None)
    cls = _STATUS_CLASSES.get(code)
    if cls is None:
        cls = openai.InternalServerError if code >= 500 else openai.APIStatusError
    return cls("boom", response=response, body=None)


class FakeClient:
    """Records every call and replays a scripted sequence of outcomes."""

    def __init__(self, outcomes=None, embed_outcome=None):
        self.calls: list[dict] = []
        self._outcomes = list(outcomes or [])
        self._embed_outcome = embed_outcome
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                result = (
                    outer._outcomes.pop(0) if outer._outcomes else _completion("ok")
                )
                if isinstance(result, Exception):
                    raise result
                return result

        class _Embeddings:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if isinstance(outer._embed_outcome, Exception):
                    raise outer._embed_outcome
                return outer._embed_outcome

        self.chat = SimpleNamespace(completions=_Completions())
        self.embeddings = _Embeddings()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MIA_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return AgentSettings()


def make_llm(settings, outcomes=None, embed_outcome=None, sleeps=None):
    client = FakeClient(outcomes, embed_outcome)
    llm = MiaLLM(
        settings,
        client=client,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )
    return llm, client


# --- request shape --------------------------------------------------------


def test_chat_sends_model_and_messages(settings):
    llm, client = make_llm(settings, [_completion("merhaba")])
    result = llm.chat([{"role": "user", "content": "selam"}])

    sent = client.calls[0]
    assert sent["model"] == settings.mia_chat_model
    assert sent["messages"] == [{"role": "user", "content": "selam"}]
    assert "tools" not in sent  # no tools requested, none sent
    assert result.content == "merhaba"
    assert result.usage.calls == 1


@pytest.mark.parametrize("thinking", [True, False])
def test_thinking_flag_travels_in_extra_body(settings, thinking):
    """The one MIA-specific switch; the measured quality/latency trade-off."""
    llm, client = make_llm(settings, [_completion("x")])
    llm.chat([{"role": "user", "content": "x"}], thinking=thinking)
    assert (
        client.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        is thinking
    )


def test_tools_are_sent_with_tool_choice(settings):
    tools = [{"type": "function", "function": {"name": "get_series", "parameters": {}}}]
    llm, client = make_llm(settings, [_completion("x")])
    llm.chat([{"role": "user", "content": "x"}], tools=tools)
    assert client.calls[0]["tools"] == tools
    assert client.calls[0]["tool_choice"] == "auto"


def test_schema_becomes_json_schema_response_format(settings):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    llm, client = make_llm(settings, [_completion('{"a": "b"}')])
    result = llm.chat([{"role": "user", "content": "x"}], schema=schema)

    fmt = client.calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == schema
    assert "strict" not in fmt["json_schema"]  # unverified on vLLM, so never sent
    assert result.parsed == {"a": "b"}


def test_unparseable_schema_output_sets_parsed_none_without_raising(settings):
    """The loop decides what to do; the client does not crash the turn."""
    llm, _ = make_llm(settings, [_completion("bu JSON degil")])
    result = llm.chat([{"role": "user", "content": "x"}], schema={"type": "object"})
    assert result.parsed is None
    assert result.content == "bu JSON degil"


def test_tools_and_schema_together_are_rejected_locally(settings):
    llm, client = make_llm(settings)
    with pytest.raises(LlmInputError):
        llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}], schema={"b": 2})
    assert client.calls == []  # never left the process


def test_too_many_images_are_rejected_locally(settings):
    llm, client = make_llm(settings)
    parts = [image_part(b"png") for _ in range(MAX_IMAGES_PER_CHAT + 1)]
    with pytest.raises(LlmInputError):
        llm.chat([{"role": "user", "content": [*parts, {"type": "text", "text": "x"}]}])
    assert client.calls == []


def test_count_images_ignores_malformed_content_parts():
    """A content list is not guaranteed to hold only dicts.

    A bare string or a None among the parts used to raise AttributeError from
    inside `chat()`, before the request was even built. That breaks this
    module's contract that every failure it raises is an LlmError, so the
    counter now skips anything that is not a dict.
    """
    messy = [{"role": "user", "content": [{"type": "text"}, "duz metin", None]}]
    assert count_images(messy) == 0

    mixed = [{"role": "user", "content": ["duz metin", image_part(b"x"), None]}]
    assert count_images(mixed) == 1

    # The ordinary case -- content as a plain string -- is untouched.
    assert count_images([{"role": "user", "content": "duz metin"}]) == 0


def test_empty_messages_are_rejected_locally(settings):
    """An empty turn is our bug, so it never costs a round trip."""
    llm, client = make_llm(settings)
    with pytest.raises(LlmInputError):
        llm.chat([])
    assert client.calls == []


# --- tool call handling ---------------------------------------------------


def _tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_tool_call_is_parsed(settings):
    llm, _ = make_llm(
        settings,
        [
            _completion(
                None,
                [_tool_call("c1", "get_series", '{"start": "2024-01"}')],
                "tool_calls",
            )
        ],
    )
    result = llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_series"
    assert result.tool_calls[0].arguments == {"start": "2024-01"}
    assert result.tool_calls[0].parse_error is None


def test_multiple_tool_calls_are_trimmed_in_both_places(settings):
    """An assistant message keeping a tool_call with no matching tool message is
    a protocol error on the next request, so raw_message is trimmed too."""
    calls = [_tool_call("c1", "a", "{}"), _tool_call("c2", "b", "{}")]
    llm, _ = make_llm(settings, [_completion(None, calls, "tool_calls")])
    result = llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "c1"
    assert len(result.raw_message["tool_calls"]) == 1
    assert len(result.as_message()["tool_calls"]) == 1


def test_a_call_with_no_id_is_answered_under_the_id_we_gave_it(settings):
    """A provider that sends no id gets `call_0`, and the tool message is
    written against that. Leaving the assistant message's own call without an
    id made the reply an answer to nothing, which the history view then drops
    -- a request carrying a tool call with no reply, the protocol error the
    trimming exists to prevent."""
    call = {"type": "function", "function": {"name": "a", "arguments": "{}"}}
    llm, _ = make_llm(settings, [_completion(None, [call], "tool_calls")])

    result = llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])

    assert result.tool_calls[0].id == "call_0"
    assert result.raw_message["tool_calls"][0]["id"] == "call_0"
    assert result.as_message()["tool_calls"][0]["id"] == "call_0"


def test_malformed_arguments_do_not_raise(settings):
    llm, _ = make_llm(
        settings, [_completion(None, [_tool_call("c1", "a", "{bozuk")], "tool_calls")]
    )
    result = llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])
    assert result.tool_calls[0].arguments is None
    assert "JSON" in (result.tool_calls[0].parse_error or "")
    assert result.tool_calls[0].arguments_raw == "{bozuk"


def test_as_message_drops_provider_specific_fields():
    """Measured against MIA: with thinking on the message carries a `reasoning`
    field. Echoing it back is not part of any contract, so it is stripped."""
    result = ChatResult(
        content="x",
        raw_message={
            "role": "assistant",
            "content": "x",
            "reasoning": "gizli dusunce zinciri",
            "reasoning_content": "baska bir saglayicinin adlandirmasi",
            "tool_calls": [{"id": "c1"}],
        },
    )
    projected = result.as_message()
    assert set(projected) == {"role", "content", "tool_calls"}
    assert "gizli dusunce zinciri" not in str(projected)


# --- retry and error mapping ---------------------------------------------


def test_retries_on_429_then_succeeds(settings):
    sleeps: list[float] = []
    llm, client = make_llm(
        settings,
        [_status_error(429), _status_error(503), _completion("sonunda")],
        sleeps=sleeps,
    )
    result = llm.chat([{"role": "user", "content": "x"}])

    assert result.content == "sonunda"
    assert len(client.calls) == 3
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] < 1.3  # exponential with a little jitter
    assert 2.0 <= sleeps[1] < 2.3


def test_gives_up_after_max_attempts(settings):
    llm, client = make_llm(settings, [_status_error(503)] * 3)
    with pytest.raises(LlmServerError):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(client.calls) == 3


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (429, LlmRateLimitError),
        (500, LlmServerError),
        (401, LlmConfigError),
        (403, LlmConfigError),
        (400, LlmBadRequestError),
        (404, LlmBadRequestError),
    ],
)
def test_error_mapping(settings, code, expected):
    llm, _ = make_llm(settings, [_status_error(code)] * 3)
    with pytest.raises(expected):
        llm.chat([{"role": "user", "content": "x"}])


def test_client_side_errors_are_not_retried(settings):
    """Retrying a 400 just burns quota and demo time."""
    llm, client = make_llm(settings, [_status_error(400)] * 3)
    with pytest.raises(LlmBadRequestError):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(client.calls) == 1


def test_timeout_is_mapped(settings):
    timeout = openai.APITimeoutError(request=SimpleNamespace(url="https://mia.test/v1"))
    llm, _ = make_llm(settings, [timeout] * 3)
    with pytest.raises(LlmTimeoutError):
        llm.chat([{"role": "user", "content": "x"}])


def test_a_timeout_is_not_retried(settings):
    """The identical request against the same server load takes as long again.
    Measured 22 Sep 2026: three 45 s timeouts in a row, 138 s, turn lost."""
    timeout = openai.APITimeoutError(request=SimpleNamespace(url="https://mia.test/v1"))
    llm, client = make_llm(settings, [timeout] * 3)
    with pytest.raises(LlmTimeoutError):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(client.calls) == 1


def test_a_dropped_connection_is_still_retried(settings):
    """A connection that never got going says nothing about how long the
    answer takes, so trying again is still reasonable."""
    dropped = openai.APIConnectionError(
        request=SimpleNamespace(url="https://mia.test/v1")
    )
    llm, client = make_llm(settings, [dropped, _completion("ok")])
    assert llm.chat([{"role": "user", "content": "x"}]).content == "ok"
    assert len(client.calls) == 2


def test_without_a_budget_the_request_carries_no_timeout(settings):
    llm, client = make_llm(settings, [_completion("x")])
    llm.chat([{"role": "user", "content": "x"}])
    assert "timeout" not in client.calls[0]


def test_a_budget_caps_the_attempt_timeout(settings):
    llm, client = make_llm(settings, [_completion("x")])
    llm.chat([{"role": "user", "content": "x"}], timeout_s=20.0)
    assert client.calls[0]["timeout"] == pytest.approx(20.0, abs=0.5)


def test_a_budget_is_obeyed_even_past_the_client_timeout(settings):
    """The caller chose the number. The agent loop gives a last-chance answer
    all the time it has; capping that at the client timeout is how the third
    22 Sep run lost an answer with a minute of its budget unused."""
    llm, client = make_llm(settings, [_completion("x")])
    llm.chat([{"role": "user", "content": "x"}], timeout_s=120.0)
    assert settings.llm_timeout_s < 120.0
    assert client.calls[0]["timeout"] == pytest.approx(120.0, abs=0.5)


def test_retries_stop_when_the_budget_runs_out(settings):
    """A retry that cannot finish inside the budget is not started, and the
    call is reported as out of time: the clock stopped it, not the server, so
    the caller's "answer with what there is" handling is what applies."""
    now = [0.0]
    client = FakeClient([_status_error(503)] * 3)

    def sleep(seconds):
        now[0] += seconds

    llm = MiaLLM(settings, client=client, sleep=sleep, clock=lambda: now[0])

    with pytest.raises(LlmTimeoutError, match="son hata: InternalServerError") as err:
        llm.chat([{"role": "user", "content": "x"}], timeout_s=7.0)

    # Attempt 1 at t=0; after ~1 s of backoff ~6 s are left, so attempt 2 runs
    # with that as its timeout; a ~2 s backoff would leave under 5 s: stop.
    assert len(client.calls) == 2
    assert 5.5 <= client.calls[1]["timeout"] <= 6.0
    assert isinstance(err.value.__cause__, openai.InternalServerError)
    assert now[0] <= 7.0  # the budget is a total, sleeps included
    assert _log_lines(settings)[-1]["error"] == "budget_after_InternalServerError"


def test_no_backoff_is_slept_before_a_retry_that_cannot_start(settings):
    """Sleeping first and then finding the time gone only carries the call
    past its budget. Measured shape: a 503 near the end of an observe call."""
    sleeps: list[float] = []
    llm, client = make_llm(settings, [_status_error(503)] * 3, sleeps=sleeps)

    with pytest.raises(LlmTimeoutError):
        llm.chat([{"role": "user", "content": "x"}], timeout_s=5.5)

    assert len(client.calls) == 1
    assert sleeps == []


def test_a_server_error_on_every_attempt_is_still_a_server_error(settings):
    """Only a retry the budget stopped becomes a timeout. Three 503s with
    time to spare mean MIA is down, and the caller must hear exactly that."""
    llm, client = make_llm(settings, [_status_error(503)] * 3)
    with pytest.raises(LlmServerError):
        llm.chat([{"role": "user", "content": "x"}], timeout_s=100.0)
    assert len(client.calls) == 3


def test_a_budget_too_small_to_try_raises_without_calling(settings):
    llm, client = make_llm(settings)
    with pytest.raises(LlmTimeoutError, match="sure kalmadi"):
        llm.chat([{"role": "user", "content": "x"}], timeout_s=1.0)
    assert client.calls == []
    assert _log_lines(settings)[0]["status"] == "error"


# --- usage log ------------------------------------------------------------


def _log_lines(settings):
    path = settings.usage_log_path
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_usage_log_writes_one_line_per_logical_call(settings):
    llm, _ = make_llm(settings, [_status_error(503), _completion("ok")])
    llm.chat([{"role": "user", "content": "x"}], tag="plan@planner@1", session_id="s1")

    lines = _log_lines(settings)
    assert len(lines) == 1  # one logical call, even though it took two attempts
    entry = lines[0]
    assert entry["kind"] == "chat"
    assert entry["status"] == "ok"
    assert entry["attempts"] == 2
    assert entry["tag"] == "plan@planner@1"
    assert entry["session_id"] == "s1"
    assert entry["prompt_tokens"] == 10


def test_usage_log_never_contains_message_content_or_the_key(settings):
    """This file is read during the demo and lives next to the repo."""
    secret = settings.mia_auth_key()
    llm, _ = make_llm(settings, [_completion("cevap metni")])
    llm.chat([{"role": "user", "content": "cok gizli soru metni"}])

    raw = settings.usage_log_path.read_text(encoding="utf-8")
    assert "cok gizli soru metni" not in raw
    assert "cevap metni" not in raw
    assert secret not in raw


def test_failures_are_logged_as_errors(settings):
    llm, _ = make_llm(settings, [_status_error(400)])
    with pytest.raises(LlmBadRequestError):
        llm.chat([{"role": "user", "content": "x"}])
    entry = _log_lines(settings)[0]
    assert entry["status"] == "error"
    assert entry["error"] == "BadRequestError"


# --- embeddings and ocr ---------------------------------------------------


def test_embed_preserves_input_order(settings):
    """The protocol does not guarantee order, so we sort by index."""
    out_of_order = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.2]),
            SimpleNamespace(index=0, embedding=[0.1]),
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=0),
    )
    llm, _ = make_llm(settings, embed_outcome=out_of_order)
    assert llm.embed(["a", "b"]) == [[0.1], [0.2]]


def test_embed_of_nothing_makes_no_call(settings):
    llm, client = make_llm(settings)
    assert llm.embed([]) == []
    assert client.calls == []


def test_ocr_uses_the_guide_prompt_and_extra_body(settings):
    llm, client = make_llm(settings, [_completion("cikarilan metin")])
    result = llm.ocr([b"png-bytes"])

    sent = client.calls[0]
    assert sent["model"] == settings.mia_ocr_model
    assert sent["extra_body"] == OCR_EXTRA_BODY_SINGLE
    assert sent["temperature"] == 0.0
    assert sent["messages"][0]["content"][-1]["text"] == OCR_PROMPT
    assert result.text == "cikarilan metin"


def test_ocr_switches_window_size_for_multiple_images(settings):
    """The guide is explicit: window_size becomes 1024 past one image."""
    llm, client = make_llm(settings, [_completion("x")])
    llm.ocr([b"a", b"b"])
    assert client.calls[0]["extra_body"] == OCR_EXTRA_BODY_MULTI


def test_ocr_rejects_too_many_images(settings):
    llm, client = make_llm(settings)
    with pytest.raises(LlmInputError):
        llm.ocr([b"a", b"b", b"c", b"d"])
    assert client.calls == []


# --- live (skipped by default) -------------------------------------------


@pytest.mark.live
def test_live_ping_roundtrip():
    """Run manually: uv run pytest -m live"""
    from dotenv import load_dotenv

    from lakehouse_agent.config import get_agent_settings

    load_dotenv()
    llm = MiaLLM(get_agent_settings())
    result = llm.chat(
        [{"role": "user", "content": "Tek kelime: tamam"}], thinking=False
    )
    assert result.content
    assert result.usage.prompt_tokens > 0


# --- regressions found in review -----------------------------------------


@pytest.mark.parametrize("arguments", [{"already": "decoded"}, 123, ["a", "b"]])
def test_non_string_tool_arguments_do_not_crash_the_turn(settings, arguments):
    """json.loads of a non-string raises TypeError, not JSONDecodeError.

    A provider that sends `arguments` already decoded would otherwise blow up
    out of chat() instead of being handed back to the model as a fixable
    mistake. The loop can recover from a bad argument; it cannot recover from
    an exception thrown past it.
    """
    broken = _completion(
        None,
        [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "a", "arguments": arguments},
            }
        ],
        "tool_calls",
    )
    llm, _ = make_llm(settings, [broken])
    result = llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])
    assert result.tool_calls[0].arguments is None
    assert result.tool_calls[0].parse_error


def test_null_tool_arguments_are_treated_as_empty(settings):
    """`arguments: null` is a legitimate 'no arguments', not an error."""
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "a", "arguments": None},
    }
    llm, _ = make_llm(settings, [_completion(None, [call], "tool_calls")])
    result = llm.chat([{"role": "user", "content": "x"}], tools=[{"a": 1}])
    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].parse_error is None


def test_embed_splits_large_inputs_into_batches(settings):
    """Measured: 1024 texts take 22s against a 45s timeout, and the catalog has
    tens of thousands of series. One request would never come back."""
    from lakehouse_agent.llm import EMBED_BATCH_SIZE

    sent: list[int] = []

    class _BatchingClient(FakeClient):
        def __init__(self):
            super().__init__()

            class _Embeddings:
                def create(self, **kwargs):
                    batch = kwargs["input"]
                    sent.append(len(batch))
                    return SimpleNamespace(
                        data=[
                            SimpleNamespace(index=i, embedding=[float(i)])
                            for i in range(len(batch))
                        ],
                        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
                    )

            self.embeddings = _Embeddings()

    llm = MiaLLM(settings, client=_BatchingClient(), sleep=lambda _s: None)
    total = EMBED_BATCH_SIZE * 2 + 5
    vectors = llm.embed([f"metin {i}" for i in range(total)])

    assert len(vectors) == total
    assert sent == [EMBED_BATCH_SIZE, EMBED_BATCH_SIZE, 5]


def test_embed_keeps_global_order_across_batches(settings):
    """Batch boundaries must not reorder the caller's list."""
    from lakehouse_agent.llm import EMBED_BATCH_SIZE

    class _OrderedClient(FakeClient):
        def __init__(self):
            super().__init__()

            class _Embeddings:
                def create(self, **kwargs):
                    batch = kwargs["input"]
                    # Answer out of order on purpose; the client must re-sort.
                    return SimpleNamespace(
                        data=[
                            SimpleNamespace(index=i, embedding=[float(hash(t) % 97)])
                            for i, t in reversed(list(enumerate(batch)))
                        ],
                        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
                    )

            self.embeddings = _Embeddings()

    texts = [f"metin {i}" for i in range(EMBED_BATCH_SIZE + 3)]
    llm = MiaLLM(settings, client=_OrderedClient(), sleep=lambda _s: None)
    vectors = llm.embed(texts)
    assert vectors == [[float(hash(t) % 97)] for t in texts]
