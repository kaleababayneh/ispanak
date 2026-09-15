"""The single door to MIA (Kloudeks), the only LLM provider this project may use.

Nothing else in the codebase imports `openai`. Everything goes through `MiaLLM`
so that retries, usage accounting and the thinking-mode switch exist in exactly
one place, and so the agent can be driven by `FakeLLM` in tests without a network.

Verified against openai 3.14.0 / httpx2 2.13.0 on 15 Sep 2026:
  - `OpenAI(api_key=..., base_url=..., timeout=float, max_retries=int)` works;
    `timeout` accepts a plain float.
  - `chat.completions.create` still takes tools / tool_choice / response_format /
    extra_body / stream.
  - Exception classes are still at `openai.*` and `APIStatusError.status_code`
    is set in `__init__`.
`max_retries=0` is passed on purpose: the SDK's own retry is silent, ours is
logged and measured.
"""

from __future__ import annotations

import base64
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import openai
from pydantic import BaseModel, Field

from lakehouse_agent.config import AgentSettings
from lakehouse_agent.contracts import Usage

# Hard limits from the MIA guide. Exceeding them is our bug, not a server error,
# so it is rejected before the request leaves the process.
MAX_IMAGES_PER_CHAT = 5
MAX_IMAGES_PER_OCR = 3

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0  # sleeps are 1s, 2s (+ jitter) between three attempts
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Measured against MIA: 256 texts take ~5.6s, 1024 take ~22.4s. 256 keeps every
# batch comfortably inside the 45s client timeout while still being one request
# per few hundred series rather than per series.
EMBED_BATCH_SIZE = 256

# Copied verbatim from the MIA guide; changing either breaks OCR output.
OCR_PROMPT = "<image>\ndocument parsing"
OCR_EXTRA_BODY_SINGLE: dict[str, Any] = {
    "skip_special_tokens": False,
    "vllm_xargs": {"ngram_size": 35, "window_size": 128},
}
OCR_EXTRA_BODY_MULTI: dict[str, Any] = {
    "skip_special_tokens": False,
    "vllm_xargs": {"ngram_size": 35, "window_size": 1024},
}
OCR_MAX_TOKENS = 8192


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class LlmError(RuntimeError):
    """Base class for every failure raised by this module."""


class LlmConfigError(LlmError):
    """Settings are wrong or missing (bad base url, missing key)."""


class LlmInputError(LlmError):
    """We built an invalid request. Never retried; fix the caller."""


class LlmRateLimitError(LlmError):
    """429 after all attempts were used."""


class LlmServerError(LlmError):
    """5xx after all attempts were used."""


class LlmTimeoutError(LlmError):
    """The request did not finish inside the configured timeout."""


class LlmBadRequestError(LlmError):
    """A 4xx other than 429. Retrying cannot help, so we never do."""


# --------------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------------


class ToolCall(BaseModel):
    """One tool call requested by the model.

    `arguments` is None when the model produced malformed JSON. That is not an
    exception here: the agent loop feeds `parse_error` back to the model once,
    which is far more useful than crashing the turn.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None = None
    arguments_raw: str = ""
    parse_error: str | None = None


class ChatResult(BaseModel):
    """One chat completion.

    `raw_message` is kept because the OpenAI protocol requires echoing the
    assistant message back into `messages` before the matching tool messages.
    It is projected through `as_message()` so provider-specific extras are never
    sent back. Measured 15 Sep 2026: with thinking on, MIA adds a `reasoning`
    field to the message, and echoing it back is not part of any contract.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_message: dict[str, Any] = Field(default_factory=dict)
    parsed: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    duration_ms: int = 0
    model: str = ""

    def as_message(self) -> dict[str, Any]:
        """Whitelist projection of the assistant message, safe to send back."""
        message: dict[str, Any] = {"role": "assistant"}
        message["content"] = self.raw_message.get("content", self.content)
        tool_calls = self.raw_message.get("tool_calls")
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message


class OcrResult(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    duration_ms: int = 0


class PingReply(BaseModel):
    """Tiny schema used by the --ping CLI to exercise structured output."""

    durum: str = Field(..., description="Tek kelimelik durum, ornek 'tamam'.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def image_part(data: bytes, mime: str = "image/png") -> dict[str, Any]:
    """Build a base64 image content block. MIA rejects external image URLs."""
    encoded = base64.b64encode(data).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def count_images(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            total += sum(1 for part in content if part.get("type") == "image_url")
    return total


def parse_tool_calls(
    raw_message: dict[str, Any],
) -> tuple[list[ToolCall], dict[str, Any]]:
    """Extract tool calls, keeping only the first one.

    MIA calls tools one at a time in practice, and the agent loop is built around
    one call per step so that each step is a separate, cancellable, approvable
    unit. If the model ever emits several, the extras are dropped *and* trimmed
    out of `raw_message` too: an assistant message carrying a tool_call with no
    matching tool message is a protocol error on the next request.
    """
    calls = raw_message.get("tool_calls") or []
    if not calls:
        return [], raw_message

    first = calls[0]
    function = first.get("function", {})
    raw_arguments = function.get("arguments", "") or ""
    arguments: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        if not isinstance(raw_arguments, str):
            # A provider that sends `arguments` already decoded. json.loads would
            # raise TypeError here, so say what happened instead of crashing.
            raise TypeError(f"beklenen metin, gelen {type(raw_arguments).__name__}")
        loaded = json.loads(raw_arguments) if raw_arguments else {}
        if isinstance(loaded, dict):
            arguments = loaded
        else:
            parse_error = f"Argumanlar bir nesne degil, {type(loaded).__name__} geldi."
    except (json.JSONDecodeError, TypeError) as exc:
        # TypeError as well as JSONDecodeError: a provider that sends `arguments`
        # as null or as an already-decoded object makes json.loads raise
        # TypeError, and letting that escape would crash the turn instead of
        # feeding the problem back to the model.
        parse_error = f"Argumanlar gecerli JSON degil: {exc}"

    tool_call = ToolCall(
        id=first.get("id") or "call_0",
        name=function.get("name", ""),
        # Always a string: the field exists so the failure can be shown back to
        # the model, and a non-string payload is exactly the case worth showing.
        arguments_raw=raw_arguments
        if isinstance(raw_arguments, str)
        else repr(raw_arguments),
        arguments=arguments,
        parse_error=parse_error,
    )
    trimmed = dict(raw_message)
    trimmed["tool_calls"] = [first]
    return [tool_call], trimmed


class LLM(Protocol):
    """What the agent depends on. `FakeLLM` satisfies this without a network."""

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
    ) -> ChatResult: ...

    def embed(
        self, texts: list[str], *, tag: str | None = None
    ) -> list[list[float]]: ...

    def ocr(
        self, images: list[bytes], *, prompt: str | None = None, tag: str | None = None
    ) -> OcrResult: ...


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class MiaLLM:
    """Synchronous MIA client.

    Synchronous on purpose: the agent loop is async and calls this through
    `asyncio.to_thread`, which keeps this module and every tool simple to read
    and simple to test, while still never blocking the UI event loop.
    """

    def __init__(
        self,
        settings: AgentSettings,
        *,
        client: Any | None = None,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        self._clock = clock
        self._client = client or openai.OpenAI(
            api_key=settings.mia_auth_key(),
            base_url=settings.mia_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # retrying is ours, so that it is logged
        )

    # -- public API --------------------------------------------------------

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
    ) -> ChatResult:
        if tools and schema:
            raise LlmInputError(
                "tools ve schema ayni istekte kullanilamaz; ya arac cagrisi ya sema."
            )
        images = count_images(messages)
        if images > MAX_IMAGES_PER_CHAT:
            raise LlmInputError(
                f"Tek istekte en fazla {MAX_IMAGES_PER_CHAT} gorsel gonderilebilir, "
                f"{images} gorsel verildi."
            )

        kwargs: dict[str, Any] = {
            "model": self.settings.mia_chat_model,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                # `strict` is deliberately not sent: it is an OpenAI-specific
                # flag and vLLM's handling of it is unverified.
                "json_schema": {"name": "output", "schema": schema},
            }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response, duration_ms, attempts = self._call(
            "chat",
            lambda: self._client.chat.completions.create(**kwargs),
            tag=tag,
            session_id=session_id,
            extra_log={
                "thinking": thinking,
                "tools": len(tools or []),
                "schema": bool(schema),
            },
        )

        choice = response.choices[0]
        raw_message = choice.message.model_dump(exclude_none=True)
        tool_calls, raw_message = parse_tool_calls(raw_message)

        parsed: dict[str, Any] | None = None
        if schema and choice.message.content:
            try:
                candidate = json.loads(choice.message.content)
                parsed = candidate if isinstance(candidate, dict) else None
            except json.JSONDecodeError:
                # Not an error here. The loop decides whether to ask again.
                parsed = None

        return ChatResult(
            content=choice.message.content,
            tool_calls=tool_calls,
            raw_message=raw_message,
            parsed=parsed,
            finish_reason=choice.finish_reason,
            usage=_usage_from(response, duration_ms),
            duration_ms=duration_ms,
            model=getattr(response, "model", self.settings.mia_chat_model),
        )

    def embed(self, texts: list[str], *, tag: str | None = None) -> list[list[float]]:
        """Embed texts, in batches, preserving input order.

        Batching is not optional. Measured against MIA on 15 Sep 2026: 1 text
        takes 0.4s, 256 take 5.6s and 1024 take 22.4s. The catalog has tens of
        thousands of series, so a single request would run for minutes and die
        on the 45s client timeout long before it returned.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            response, _duration_ms, _attempts = self._call(
                "embed",
                lambda batch=batch: self._client.embeddings.create(
                    model=self.settings.mia_embed_model,
                    input=batch,
                    encoding_format="float",
                ),
                tag=tag,
                session_id=None,
                extra_log={"inputs": len(batch), "batch_start": start},
            )
            # Order is not guaranteed by the protocol, so sort by index explicitly.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        if len(vectors) != len(texts):
            raise LlmError(
                f"Embedding sayisi girdiyle uyusmuyor: {len(texts)} metin gonderildi, "
                f"{len(vectors)} vektor dondu."
            )
        return vectors

    def ocr(
        self, images: list[bytes], *, prompt: str | None = None, tag: str | None = None
    ) -> OcrResult:
        """Read text out of document images.

        Prompt and `extra_body` come straight from the MIA guide. The guide is
        explicit that both must be kept as-is and that `window_size` changes to
        1024 as soon as more than one image is sent.
        """
        if not images:
            raise LlmInputError("OCR icin en az bir gorsel gerekli.")
        if len(images) > MAX_IMAGES_PER_OCR:
            raise LlmInputError(
                f"OCR istegi basina en fazla {MAX_IMAGES_PER_OCR} gorsel gonderilebilir, "
                f"{len(images)} gorsel verildi."
            )

        content: list[dict[str, Any]] = [image_part(image) for image in images]
        content.append({"type": "text", "text": prompt or OCR_PROMPT})
        extra_body = OCR_EXTRA_BODY_MULTI if len(images) > 1 else OCR_EXTRA_BODY_SINGLE

        response, duration_ms, _ = self._call(
            "ocr",
            lambda: self._client.chat.completions.create(
                model=self.settings.mia_ocr_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=OCR_MAX_TOKENS,
                temperature=0.0,
                extra_body=extra_body,
            ),
            tag=tag,
            session_id=None,
            extra_log={"images": len(images)},
        )
        return OcrResult(
            text=response.choices[0].message.content or "",
            usage=_usage_from(response, duration_ms),
            duration_ms=duration_ms,
        )

    # -- internals ---------------------------------------------------------

    def _call(
        self,
        kind: str,
        fn: Any,
        *,
        tag: str | None,
        session_id: str | None,
        extra_log: dict[str, Any] | None = None,
    ) -> tuple[Any, int, int]:
        """Run one logical call: retry, time it, log it, map the error."""
        started = self._clock()
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = fn()
            except Exception as exc:  # noqa: BLE001 - re-raised as LlmError below
                last_error = exc
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    duration_ms = int((self._clock() - started) * 1000)
                    self._log(
                        kind=kind,
                        tag=tag,
                        session_id=session_id,
                        duration_ms=duration_ms,
                        attempts=attempt,
                        status="error",
                        error=type(exc).__name__,
                        extra=extra_log,
                    )
                    raise _map_error(exc) from exc
                self._sleep(
                    BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                )
                continue

            duration_ms = int((self._clock() - started) * 1000)
            usage = getattr(response, "usage", None)
            self._log(
                kind=kind,
                tag=tag,
                session_id=session_id,
                duration_ms=duration_ms,
                attempts=attempt,
                status="ok",
                error=None,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                model=getattr(response, "model", None),
                extra=extra_log,
            )
            return response, duration_ms, attempt

        raise _map_error(last_error) if last_error else LlmError("bilinmeyen hata")

    def _log(self, **fields: Any) -> None:
        """Append one JSONL line per logical call.

        Counters only. Message content, arguments and the API key never appear
        here: the file is committed-adjacent, shared and read during the demo.
        """
        extra = fields.pop("extra", None) or {}
        record = {
            "ts": datetime.now(UTC).isoformat(),
            **{k: v for k, v in fields.items() if v is not None},
            **extra,
        }
        try:
            path: Path = self.settings.usage_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Telemetry must never break a turn.
            pass


def _usage_from(response: Any, duration_ms: int) -> Usage:
    usage = getattr(response, "usage", None)
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        calls=1,
        duration_ms=duration_ms,
    )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    return False


def _map_error(exc: Exception) -> LlmError:
    if isinstance(exc, LlmError):
        return exc
    if isinstance(exc, openai.APITimeoutError):
        return LlmTimeoutError(f"MIA zaman asimi: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return LlmServerError(f"MIA'ya baglanilamadi: {exc}")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 429:
            return LlmRateLimitError(f"MIA kota/hiz siniri (429): {exc}")
        if status in (401, 403):
            # Measured 15 Sep 2026: MIA answers an unknown model id with 403,
            # not 404, so this message must not blame the key alone.
            return LlmConfigError(
                f"MIA erisimi reddetti ({status}). Anahtar gecersiz olabilir ya da "
                "model kimligi yanlis/erisiminiz yok."
            )
        if status >= 500:
            return LlmServerError(f"MIA sunucu hatasi ({status}): {exc}")
        return LlmBadRequestError(f"MIA istegi reddetti ({status}): {exc}")
    return LlmError(f"Beklenmeyen hata: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    """`python -m lakehouse_agent.llm --ping`

    This is an entrypoint, so it is the one place allowed to load `.env`.
    It prints counters and shapes, never the key and never full message text.
    """
    import argparse

    parser = argparse.ArgumentParser(description="MIA baglanti kontrolu")
    parser.add_argument("--ping", action="store_true", help="Canli baglanti testi yap")
    args = parser.parse_args(argv)
    if not args.ping:
        parser.print_help()
        return 0

    from dotenv import load_dotenv

    load_dotenv()

    from lakehouse_agent.config import get_agent_settings

    try:
        settings = get_agent_settings()
    except Exception as exc:  # noqa: BLE001
        print(f"Ayarlar okunamadi: {exc}")
        print("MIA_API_KEY .env dosyasinda tanimli mi?")
        return 2

    llm = MiaLLM(settings)
    print(f"openai={openai.__version__}  model={settings.mia_chat_model}")

    try:
        result = llm.chat(
            [{"role": "user", "content": "Tek kelimeyle cevap ver: tamam"}],
            thinking=False,
            tag="cli@ping",
        )
    except LlmError as exc:
        print(f"chat BASARISIZ: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"chat     OK  {result.duration_ms:5d} ms  "
        f"prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens}"
    )

    try:
        vectors = llm.embed(["Konut kredisi faiz orani"], tag="cli@ping")
    except LlmError as exc:
        print(f"embed BASARISIZ: {type(exc).__name__}: {exc}")
        return 1
    print(f"embed    OK  vektor boyutu={len(vectors[0])}")

    print(f"kullanim logu: {settings.usage_log_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
