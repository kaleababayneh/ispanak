"""A scripted, offline stand-in for `MiaLLM`.

Every test of the agent loop runs against this. The script is a list of replies
handed out in order, so a test states exactly what the model does and then
asserts what the loop did about it.

Two deliberate choices:

* Running off the end of the script raises `ScriptExhausted` instead of
  returning a default. A loop that calls the model more times than the test
  expects is the bug we most want to catch, and a silent default would hide it.
* Replies are built as real OpenAI-shaped messages, so `ChatResult.as_message()`
  behaves here exactly as it does against MIA.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from lakehouse_agent.contracts import Usage
from lakehouse_agent.llm import ChatResult, OcrResult, parse_tool_calls

ScriptItem = (
    ChatResult
    | Callable[[list[dict[str, Any]], list[dict[str, Any]] | None, bool], ChatResult]
)


class ScriptExhausted(AssertionError):
    """The loop asked for one more reply than the script provides."""


class FakeLLM:
    """Satisfies the `LLM` protocol without a network.

    Args:
        script: replies in order. An entry may be a `ChatResult` or a callable
            taking `(messages, tools, thinking)` so a test can assert on what
            the loop sent before deciding what to reply.
        embed_dim: width of the fake embedding vectors.
        ocr_text: what `ocr()` returns.
    """

    def __init__(
        self,
        script: list[ScriptItem] | None = None,
        *,
        embed_dim: int = 8,
        ocr_text: str = "",
    ) -> None:
        self._script: list[ScriptItem] = list(script or [])
        self._embed_dim = embed_dim
        self._ocr_text = ocr_text
        self.calls: list[dict[str, Any]] = []

    def extend(self, items: list[ScriptItem]) -> None:
        """Append more replies to the script.

        The constructor copies the list it is given, which is right for a
        test that hands over a literal. The chat CLI needs the other thing:
        it cannot know how many questions will be asked, so it pushes one
        scripted turn per question as the person types it. Without this,
        `--fake --chat` raised `ScriptExhausted` on the second question and
        the interactive loop could not be exercised offline at all.
        """
        self._script.extend(items)

    # -- LLM protocol ------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking: bool = True,
        schema: dict[str, Any] | None = None,
        *,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        tag: str | None = None,
        session_id: str | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "kind": "chat",
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "thinking": thinking,
                "schema": schema,
                "tag": tag,
                "session_id": session_id,
                "max_tokens": max_tokens,
                "timeout_s": timeout_s,
            }
        )
        if not self._script:
            raise ScriptExhausted(
                f"FakeLLM script bitti ama {len(self.calls)}. chat cagrisi yapildi. "
                "Dongu beklenenden fazla adim atiyor olabilir."
            )
        item = self._script.pop(0)
        return item(messages, tools, thinking) if callable(item) else item

    def embed(self, texts: list[str], *, tag: str | None = None) -> list[list[float]]:
        self.calls.append({"kind": "embed", "inputs": list(texts), "tag": tag})
        return [self._vector(text) for text in texts]

    def ocr(
        self, images: list[bytes], *, prompt: str | None = None, tag: str | None = None
    ) -> OcrResult:
        self.calls.append(
            {"kind": "ocr", "images": len(images), "prompt": prompt, "tag": tag}
        )
        return OcrResult(text=self._ocr_text, usage=Usage(calls=1))

    # -- helpers -----------------------------------------------------------

    def _vector(self, text: str) -> list[float]:
        """Deterministic unit vector seeded from the text.

        Same text always gives the same vector and different texts give
        different ones, which is all a catalog-search test needs.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._embed_dim)]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    # -- reply constructors ------------------------------------------------

    @staticmethod
    def text(
        content: str,
        *,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        duration_ms: int = 5,
    ) -> ChatResult:
        """A plain answer with no tool call.

        `duration_ms` is how long the call claims to have taken. The loop adds
        it to the turn's working time, so a test can make the model slow
        without making the clock lie.
        """
        return ChatResult(
            content=content,
            raw_message={"role": "assistant", "content": content},
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                calls=1,
                duration_ms=duration_ms,
            ),
            duration_ms=duration_ms,
            model="fake",
        )

    @staticmethod
    def tool_call(
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str = "call_1",
        content: str | None = None,
        duration_ms: int = 5,
    ) -> ChatResult:
        """A reply asking for one tool."""
        return FakeLLM.tool_calls(
            [(name, arguments)],
            ids=[call_id],
            content=content,
            duration_ms=duration_ms,
        )

    @staticmethod
    def tool_calls(
        calls: list[tuple[str, dict[str, Any]]],
        *,
        ids: list[str] | None = None,
        content: str | None = None,
        duration_ms: int = 5,
    ) -> ChatResult:
        """A reply asking for several tools, to exercise the trimming rule."""
        ids = ids or [f"call_{i + 1}" for i in range(len(calls))]
        raw_calls = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for call_id, (name, args) in zip(ids, calls, strict=True)
        ]
        raw_message = {"role": "assistant", "content": content, "tool_calls": raw_calls}
        parsed_calls, trimmed = parse_tool_calls(raw_message)
        return ChatResult(
            content=content,
            tool_calls=parsed_calls,
            raw_message=trimmed,
            finish_reason="tool_calls",
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                calls=1,
                duration_ms=duration_ms,
            ),
            duration_ms=duration_ms,
            model="fake",
        )

    @staticmethod
    def bad_json_tool_call(
        name: str, raw: str, *, call_id: str = "call_1"
    ) -> ChatResult:
        """A tool call whose arguments are not valid JSON."""
        raw_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": raw},
                }
            ],
        }
        parsed_calls, trimmed = parse_tool_calls(raw_message)
        return ChatResult(
            tool_calls=parsed_calls,
            raw_message=trimmed,
            finish_reason="tool_calls",
            usage=Usage(calls=1),
            model="fake",
        )

    @staticmethod
    def structured(obj: dict[str, Any]) -> ChatResult:
        """A reply to a `schema=` request."""
        content = json.dumps(obj, ensure_ascii=False)
        return ChatResult(
            content=content,
            raw_message={"role": "assistant", "content": content},
            parsed=obj,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, calls=1),
            model="fake",
        )
