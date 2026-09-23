"""The turn loop: plan, call a tool, look at the result, repeat, then answer.

This is our own loop rather than a graph framework's. The mentors asked for the
agent engine to be our code, the single-process interface makes a checkpointer
pointless, and a `while` that fits on a screen is far easier to reason about at
two in the morning than a runtime whose control flow lives in a library.

Shape of one turn:

    plan  ->  (act -> observe)*  ->  answer  ->  verify

The loop is async so the interface never freezes. The model client and the
tools stay synchronous and are run with `asyncio.to_thread`, which keeps every
tool a plain function that a test can call directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from lakehouse_agent.agent import approvals as approval_rules
from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.agent.events import EventSink, emit
from lakehouse_agent.agent.formatting import (
    build_tables_summary,
    describe_table,
    format_sources,
    format_tool_message,
    json_safe,
    preview_rows,
)
from lakehouse_agent.agent.history import model_messages
from lakehouse_agent.agent.prompts import load_prompt, render
from lakehouse_agent.agent.state import (
    ApprovalDecision,
    ApprovalRequest,
    SessionState,
    Step,
    VerificationResult,
    chart_ref_of,
    initial_state,
)
from lakehouse_agent.agent.verifier import AnswerVerifier
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.contracts import ToolResult, Usage
from lakehouse_agent.llm import (
    LLM,
    MIN_ATTEMPT_S,
    ChatResult,
    LlmError,
    LlmTimeoutError,
    MiaLLM,
    ToolCall,
)
from lakehouse_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DATA_RANGE = "2021-01 – 2026-06"


class AgentError(RuntimeError):
    """The turn could not be completed at all."""


class AgentConfig(BaseModel):
    """The guards. Every value here exists because something can run away.

    Open-weights models loop: they re-call a tool that returned nothing,
    or keep planning without ever answering. None of these limits are
    performance tuning, they are the difference between a demo that ends and
    one that hangs in front of a jury.
    """

    #: A question needing more than eight tool calls is a question we got wrong.
    #: Tool steps in one turn. Raised from 8 to 12 on 22 Sep: over 45k series
    #: the model spends steps on searching. Measured live -- a turn that
    #: misread "60 aylık" as a loan maturity used six searches before finding
    #: its series and had none left to fetch the data. A cap, not a target:
    #: the scenario's turns still finish in 2-4 steps, so a good turn costs
    #: nothing extra.
    max_steps: int = 12

    #: One slow tool must not take the turn with it.
    step_timeout_s: float = 60.0

    #: Output cap for an observing call. Observing decides the next move in a
    #: sentence (planner rule 14); measured 22 Sep on the 45k pool, DeepSeek
    #: twice wrote the whole answer there instead (696 and 352 tokens) and
    #: then wrote it again in the answer. A call that hits the cap while
    #: making a tool call is redone uncapped -- a cut-off tool call would
    #: cost more than the prose saves.
    observe_max_tokens: int = 300

    #: Total working time. Measured on `active_s`, which excludes time spent
    #: waiting for a human: a jury thinking about an approval is not the agent
    #: being slow. Model calls are held to it too: a call's attempts all come
    #: out of what was left of it when the call began (see `_chat`), not a
    #: fresh client timeout per retry.
    total_budget_s: float = 180.0

    #: A planning or observing call is not started with less budget than this;
    #: the turn goes to its answer with what it has instead.
    min_call_s: float = 10.0

    #: The answer always gets at least this long in total, even past the
    #: budget. A turn that did its work and then ran out of time to say so is
    #: the worst way to end one, and it is how the 22 Sep live run ended.
    answer_min_s: float = 30.0

    #: Output cap for the thinking-off retry of a call that timed out. The
    #: thinking-on call itself is not capped: reasoning tokens count against
    #: `max_tokens`, and a cap low enough to matter can leave the answer empty.
    fallback_max_tokens: int = 1024

    #: A thinking-on answer is tried first only if the model's recent speed
    #: could produce this many tokens inside the attempt's window; otherwise
    #: the answer starts with thinking off and all the time there is. The one
    #: thinking-on answer we measured (15 Sep) used 750 completion tokens; on
    #: 22 Sep, at 7-10 tokens/s, every thinking-on answer timed out at 45 s,
    #: and in the third run its thinking-off retry did too.
    answer_thinking_tokens: int = 800

    #: A bad argument is fed back once. Insisting a second time means the model
    #: is not reading the error, and a third call would just burn quota.
    max_arg_retries: int = 1

    #: Calling the same tool with the same arguments a third time is a loop.
    repeat_call_limit: int = 2

    #: Measured: thinking on costs about four times the time and tokens. Worth
    #: it where judgement matters, wasteful where the model is just filling in
    #: arguments it already decided on.
    thinking: dict[str, bool] = Field(
        default_factory=lambda: {"plan": True, "observe": False, "answer": True}
    )

    ambiguity_ratio: float = approval_rules.DEFAULT_AMBIGUITY_RATIO
    quality_missing_pct: float = approval_rules.DEFAULT_MISSING_PCT


class Verifier(Protocol):
    """Checks a finished answer against what the tools actually returned."""

    def verify(self, answer: str, state: SessionState) -> VerificationResult: ...


class NullVerifier:
    """Passes everything. The default is `AnswerVerifier`; this is for a
    caller that wants the answer unchecked."""

    def verify(self, answer: str, state: SessionState) -> VerificationResult:
        return VerificationResult(passed=True)


ApproveFn = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


@dataclass
class AgentDeps:
    """Everything the loop needs, injected so a test can replace any of it."""

    llm: LLM
    registry: ToolRegistry
    settings: AgentSettings
    lake: Any = None
    sink: EventSink | None = None
    verifier: Verifier = field(default_factory=AnswerVerifier)
    config: AgentConfig = field(default_factory=AgentConfig)

    #: Async on purpose. In a chat interface, asking a human means awaiting a
    #: round trip; a synchronous callback would block the whole event loop and
    #: freeze the very screen that is supposed to show the question.
    approve: ApproveFn | None = None

    clock: Callable[[], float] = time.monotonic
    today: Callable[[], date] = date.today


class TurnResult(BaseModel):
    """What the caller gets back. The state itself carries the detail."""

    answer: str | None = None
    truncated_reason: str | None = None
    table_refs: list[str] = Field(default_factory=list)
    active_table: str | None = None
    usage: Usage = Field(default_factory=Usage)
    duration_ms: int = 0
    approvals: int = 0


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


async def run_turn(state: SessionState, deps: AgentDeps, question: str) -> TurnResult:
    """Run one user turn end to end, mutating `state` as it goes."""
    # A turn that died mid-step -- cancelled at an approval card, crashed
    # between the model asking for a tool and the tool answering -- can leave
    # a call with no tool message. Answered here, before this turn's first
    # message, the reply lands in the dead turn's own stretch of the history,
    # which the model sees collapsed to question and answer, call and reply
    # dropped together. Answered at the end of this turn instead, it would
    # sit in the part shown in full with its call out of view: a 400.
    _answer_unanswered_calls(state, "Onceki tur yarida kaldi.", "abandoned")
    state.turn_no += 1
    state.steps_this_turn = 0
    state.active_s = 0.0
    state.approval_wait_s = 0.0
    state.truncated_reason = None
    state.pending_calls = []
    state.arg_retries = {}
    state.call_signatures = {}
    state.warnings = []
    state.verification = None
    # Cleared with the rest of the per-turn bookkeeping: a turn that dies
    # before writing an answer used to leave the previous turn's answer here,
    # and anything reading the answer off the state (contracts.md §5) would
    # have shown it again under the new question.
    state.draft_answer = None

    started = deps.clock()
    approvals_before = len(state.approvals)
    usage_before = state.usage
    state.turn_starts.append(len(state.messages))
    state.messages.append({"role": "user", "content": question})

    try:
        await _llm_step(state, deps, node="plan", question=question)

        while state.pending_calls:
            reason = _guard(state, deps.config)
            if reason:
                state.truncated_reason = reason
                break

            call = state.pending_calls.pop(0)
            await _act(state, deps, call, question)
            if state.truncated_reason:
                break

            try:
                await _llm_step(state, deps, node="observe", question=question)
            except LlmTimeoutError:
                # Observing only decides the next move; every tool result is
                # already in the session. Measured 22 Sep: an observe call ran
                # past 45 s after the scenario's last tool had listed the
                # months, and the whole finished analysis was thrown away.
                # Answer with what there is, and say the turn was cut short.
                # A 503 whose retry the budget stopped arrives here too: the
                # client reports that as out of time (llm.py `out_of_time`).
                state.truncated_reason = "llm_error"
                state.pending_calls = []
                break

        _answer_unanswered_calls(
            state,
            _ABANDONED_DETAIL.get(state.truncated_reason or "", "Tur sonlandi."),
            state.truncated_reason or "abandoned",
        )
        answer = await _answer(state, deps, question)
        await _verify(state, deps, answer)
        answer = _append_flags(answer, state.verification)
        state.draft_answer = answer

    except LlmError as exc:
        state.truncated_reason = "llm_error"
        await emit(
            deps.sink,
            state.session_id,
            "error",
            message=str(exc),
            kind=type(exc).__name__,
        )
        raise AgentError(f"Model cagrisi basarisiz: {exc}") from exc
    except asyncio.CancelledError:
        # Stopping a turn is not an `Exception`, so without this branch a
        # cancelled turn fell through to the `finally` below and published
        # `done` with `truncated_reason: null` -- indistinguishable from a
        # turn that finished. The tool messages are already repaired inside
        # `_act`; what was missing was saying out loud that the turn stopped.
        state.truncated_reason = "cancelled"
        with contextlib.suppress(Exception):
            await emit(
                deps.sink,
                state.session_id,
                "error",
                message="Tur durduruldu.",
                kind="CancelledError",
            )
        raise
    except Exception as exc:
        # Anything else still has to leave the channel consistent. Only
        # `LlmError` was handled, so any other crash reached the `finally`
        # below and published `done` with `truncated_reason: null` -- an
        # interface reading that closes the turn as a success.
        state.truncated_reason = "llm_error"
        await emit(
            deps.sink,
            state.session_id,
            "error",
            message=str(exc),
            kind=type(exc).__name__,
        )
        raise
    finally:
        duration_ms = int((deps.clock() - started) * 1000)
        await emit(
            deps.sink,
            state.session_id,
            "done",
            truncated_reason=state.truncated_reason,
            steps=state.steps_this_turn,
            duration_ms=duration_ms,
            usage=state.usage.model_dump(),
        )

    return TurnResult(
        answer=state.draft_answer,
        truncated_reason=state.truncated_reason,
        table_refs=sorted(state.tables),
        active_table=state.active_table,
        # This turn's cost. `state.usage` is the session running total, and
        # reporting that as the turn's made every turn look more expensive
        # than the last -- the one number a jury would read off the screen.
        usage=state.usage - usage_before,
        duration_ms=int((deps.clock() - started) * 1000),
        approvals=len(state.approvals) - approvals_before,
    )


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


async def _llm_step(
    state: SessionState, deps: AgentDeps, *, node: str, question: str
) -> None:
    """Ask the model what to do next. May produce one tool call."""
    prompt = load_prompt("planner")
    system = render(
        prompt,
        today=deps.today().isoformat(),
        data_range=DATA_RANGE,
        tables_summary=build_tables_summary(state.tables, state.active_table),
        active_table=state.active_table or "yok",
        tool_names=", ".join(deps.registry.names()) or "yok",
    )
    thinking = deps.config.thinking.get(node, True)
    started_at = datetime.now(UTC)

    if deps.config.total_budget_s - state.active_s < deps.config.min_call_s:
        # Not enough budget left to plan the next move. The answer still runs,
        # with what has been gathered and an honest "time ran out" in it.
        state.truncated_reason = state.truncated_reason or "time_budget"
        state.pending_calls = []
        return

    # The model call is a step like any other. It was the only kind that never
    # published one, so an interface drawing step cards saw the tools and never
    # the thinking between them -- and these are precisely the steps that carry
    # a token count, which docs/contracts.md puts in the `step_end` payload.
    step_no = state.next_step_no()
    node_name = "plan" if node == "plan" else "observe"
    await emit(
        deps.sink,
        state.session_id,
        "step_start",
        step_no=step_no,
        node=node_name,
        tool=None,
        args={},
    )

    spent = _Spent(thinking=thinking)
    # Observing decides the next move, it does not write the answer: capped so
    # a model that starts drafting one there stops early (AgentConfig).
    cap_tokens = deps.config.observe_max_tokens if node == "observe" else None
    try:
        messages = [{"role": "system", "content": system}, *model_messages(state)]
        result = await _chat(
            state,
            deps,
            messages,
            tools=deps.registry.openai_tools() or None,
            thinking=thinking,
            tag=f"{node}@{prompt.tag}",
            spent=spent,
            max_tokens=cap_tokens,
        )
        if (
            cap_tokens is not None
            and result.finish_reason == "length"
            and result.tool_calls
        ):
            # The cap cut a tool call in half. Prose is worth trimming; an
            # action is not -- ask again with no cap.
            result = await _chat(
                state,
                deps,
                messages,
                tools=deps.registry.openai_tools() or None,
                thinking=thinking,
                tag=f"{node}@{prompt.tag}",
                spent=spent,
            )
    except BaseException as exc:
        # The card was opened above, so it is closed here whatever ended the
        # call -- a timeout, a server error, a stopped turn -- or an interface
        # drawing step cards shows it running forever; the turn may well carry
        # on (see `run_turn`). `spent` says how long the call really ran and
        # whether its last attempt had thinking on.
        summary = str(exc)[:200] or type(exc).__name__
        state.steps.append(
            Step(
                step_no=step_no,
                turn_no=state.turn_no,
                node=node_name,
                ok=False,
                summary=summary,
                error=type(exc).__name__,
                started_at=started_at,
                duration_ms=spent.ms,
                thinking=spent.thinking,
                prompt_version=prompt.tag,
            )
        )
        with contextlib.suppress(Exception):
            await emit(
                deps.sink,
                state.session_id,
                "step_end",
                step_no=step_no,
                node=node_name,
                ok=False,
                summary=summary,
                error=type(exc).__name__,
                duration_ms=spent.ms,
                tokens=None,
            )
        raise

    state.messages.append(result.as_message())
    state.pending_calls = list(result.tool_calls)
    state.usage = state.usage + result.usage

    state.steps.append(
        Step(
            step_no=step_no,
            turn_no=state.turn_no,
            node=node_name,
            summary=(result.content or "")[:200],
            ok=True,
            started_at=started_at,
            duration_ms=spent.ms,
            tokens=result.usage,
            thinking=spent.thinking,
            prompt_version=prompt.tag,
        )
    )
    # The plan text is this step's content, so it belongs between the card
    # opening and the card closing, not after it.
    if node == "plan" and result.content:
        await emit(deps.sink, state.session_id, "plan", text=result.content)

    await emit(
        deps.sink,
        state.session_id,
        "step_end",
        step_no=step_no,
        node=node_name,
        ok=True,
        summary=(result.content or "")[:200],
        error=None,
        duration_ms=spent.ms,
        tokens=result.usage.model_dump(),
    )


async def _act(
    state: SessionState, deps: AgentDeps, call: ToolCall, question: str
) -> None:
    """Run one tool call.

    Every path through this function writes exactly one tool message for
    `call.id`, and publishes exactly one `step_start` / `step_end` pair. An
    assistant message carrying a tool call with no matching tool message makes
    the next request a protocol error, so "the tool failed", "we refused to run
    it" and "nobody answered the approval card" all still have to answer. The
    step events matter for the same reason from the other side: bad arguments
    and a cancelled fetch are the two most likely things a jury will see, and
    without an end event the screen would simply stop moving.
    """
    step_no = state.next_step_no()
    started_at = datetime.now(UTC)
    t0 = deps.clock()
    waited_before = state.approval_wait_s

    def working_s() -> float:
        """Seconds this step actually worked.

        The fetch gate happens inside this step, and a person can sit on that
        card for a minute. Charging that to the turn budget would time out the
        very interaction the gate exists for, so it is subtracted here rather
        than anywhere else: this is the only place a step and an approval
        overlap.
        """
        return (deps.clock() - t0) - (state.approval_wait_s - waited_before)

    await emit(
        deps.sink,
        state.session_id,
        "step_start",
        step_no=step_no,
        node="act",
        tool=call.name,
        args=call.arguments or {},
    )

    async def finish(
        content: str,
        *,
        ok: bool,
        summary: str,
        error: str | None = None,
        table_ref: str | None = None,
        node: str = "act",
    ) -> None:
        elapsed = working_s()
        state.messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": content}
        )
        state.steps.append(
            Step(
                step_no=step_no,
                turn_no=state.turn_no,
                node=node,  # type: ignore[arg-type]
                tool=call.name,
                args=call.arguments or {},
                ok=ok,
                summary=summary,
                error=error,
                started_at=started_at,
                duration_ms=int(elapsed * 1000),
                table_ref=table_ref,
            )
        )
        state.steps_this_turn += 1
        state.active_s += elapsed
        await emit(
            deps.sink,
            state.session_id,
            "step_end",
            step_no=step_no,
            node=node,
            ok=ok,
            summary=summary,
            error=error,
            duration_ms=int(elapsed * 1000),
            tokens=None,  # a tool call spends no model tokens
        )

    # -- the tool has to exist ------------------------------------------
    spec = deps.registry.get(call.name)
    if spec is None:
        message = (
            f"[HATA] Bilinmeyen arac: {call.name!r}. "
            f"Kullanilabilir araclar: {', '.join(deps.registry.names()) or 'yok'}."
        )
        await finish(message, ok=False, summary=message, error="unknown_tool")
        return

    # -- repeated identical call = a loop --------------------------------
    signature = _call_signature(call)
    seen = state.call_signatures.get(signature, 0)
    if seen >= deps.config.repeat_call_limit:
        message = (
            f"[HATA] {call.name} ayni argumanlarla {seen} kez cagrildi. "
            "Ayni cagriyi tekrarlama; elindeki sonuclarla devam et."
        )
        state.truncated_reason = "repeat_loop"
        await finish(message, ok=False, summary=message, error="repeat_loop")
        return
    state.call_signatures[signature] = seen + 1

    # -- arguments have to validate --------------------------------------
    if call.parse_error:
        await _record_arg_failure(state, deps, call, call.parse_error, finish)
        return
    try:
        args = deps.registry.validate(call.name, call.arguments or {})
    except ValidationError as exc:
        await _record_arg_failure(
            state, deps, call, _short_validation_error(exc), finish
        )
        return

    # -- a human may have to approve it ----------------------------------
    ctx = ToolContext(
        session_id=state.session_id,
        step_no=step_no,
        llm=deps.llm,
        lake=deps.lake,
        settings=deps.settings,
        active_table=state.active_table,
        lineage=state.lineage,
        user_message=question,
        _tables=state.tables,
        retired_refs=frozenset(state.retired_tables),
    )

    request = approval_rules.needs_fetch_approval(spec, args, ctx, state)
    if request is not None:
        try:
            decision = await _ask(state, deps, request)
        except asyncio.CancelledError:
            # A turn stopped while the card was open. Cancellation is not an
            # `Exception`, so without this branch the call went unanswered.
            message = "[HATA] Onay beklenirken tur durduruldu; arac calistirilmadi."
            with contextlib.suppress(Exception):
                await finish(
                    message,
                    ok=False,
                    summary=message,
                    error="cancelled",
                    node="approval",
                )
            raise
        except Exception as exc:  # noqa: BLE001 - see below
            # A browser closed, a card timed out, the interface crashed. None
            # of that may leave the tool call unanswered: the session would
            # then fail on every later question with an unexplained 400.
            message = (
                f"[HATA] Onay alinamadi ({type(exc).__name__}: {exc}). "
                "Bu arac calistirilmadi."
            )
            await finish(
                message,
                ok=False,
                summary=message,
                error="approval_failed",
                node="approval",
            )
            return
        if decision.choice == approval_rules.CHOICE_CANCEL:
            message = "[HATA] Kullanici dis veri cekimini iptal etti."
            await finish(
                message, ok=False, summary=message, error="cancelled", node="approval"
            )
            return

    # -- run it ----------------------------------------------------------
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(deps.registry.execute, call.name, args, ctx),
            timeout=deps.config.step_timeout_s,
        )
    except TimeoutError:
        message = (
            f"[HATA] {call.name} {deps.config.step_timeout_s:g} saniye icinde bitmedi. "
            "Daha dar bir aralik ya da daha az seri dene."
        )
        # ctx is discarded, so whatever the abandoned thread writes lands in an
        # orphan and the session stays consistent.
        await finish(message, ok=False, summary=message, error="timeout")
        return
    except asyncio.CancelledError:
        # Cancellation is not an exception the `except` below sees: it derives
        # from BaseException. Without this branch a stopped turn left the
        # model's tool call unanswered, and every later request in the session
        # is rejected for it -- the session is dead, not just the turn.
        message = f"[HATA] {call.name} iptal edildi."
        with contextlib.suppress(Exception):
            await finish(message, ok=False, summary=message, error="cancelled")
        raise
    except Exception as exc:  # noqa: BLE001 - a tool bug must not end the turn
        message = f"[HATA] {call.name} hata verdi: {type(exc).__name__}: {exc}"
        await finish(message, ok=False, summary=message, error=type(exc).__name__)
        return

    # -- merge the writes back -------------------------------------------
    active_before = state.active_table
    warnings, added_sources = _merge_context(state, ctx, result)
    if warnings:
        result = result.model_copy(update={"warnings": [*result.warnings, *warnings]})
        state.warnings.extend(warnings)

    table_note = ""
    if result.table_ref and result.table_ref in state.tables:
        table_note = describe_table(result.table_ref, state.tables[result.table_ref])

    await finish(
        format_tool_message(result, table_note),
        ok=result.ok,
        summary=result.summary,
        table_ref=result.table_ref,
    )

    await _emit_result_events(state, deps, result, ctx)

    # -- approvals that depend on what came back --------------------------
    # Guarded for the same reason as the gate above, with one difference: the
    # tool message is already written, so a failure here costs the gate, not
    # the session.
    try:
        await _post_run_approvals(
            state,
            deps,
            spec,
            result,
            step_no,
            active_before=active_before,
            added_sources=added_sources,
        )
    except Exception as exc:  # noqa: BLE001
        state.warnings.append(f"Onay adimi atlandi ({type(exc).__name__}).")


async def _answer(state: SessionState, deps: AgentDeps, question: str) -> str:
    """Write the final answer. No tools; the work is done by now."""
    prompt = load_prompt("answer")
    system = render(
        prompt,
        # The question itself, restated. It is the first user message and by
        # now fifteen steps of tool output sit between it and the model:
        # measured 23 Sep, a three-part question came back answered with
        # "Simdi ek bir isaretleme yapiyorum" while the tool that answered it
        # had already printed "3 ay: 2021-05, 2023-11, 2024-01".
        question=question,
        tables_summary=build_tables_summary(state.tables, state.active_table),
        charts=", ".join(chart_ref_of(chart) for chart in state.charts) or "yok",
        sources=format_sources(state.sources),
        warnings=" | ".join(_collected_warnings(state)) or "yok",
        truncated_reason=_truncated_text(state.truncated_reason),
    )
    started_at = datetime.now(UTC)

    spent = _Spent()
    try:
        result = await _chat(
            state,
            deps,
            [{"role": "system", "content": system}, *_answer_messages(state)],
            tools=None,
            thinking=deps.config.thinking.get("answer", True),
            tag=f"answer@{prompt.tag}",
            spent=spent,
            last_chance=True,
        )
    except BaseException as exc:
        # Recorded, so the recap of a failed turn shows where it ended and
        # what the answer cost; the third 22 Sep run lost two answer attempts
        # (90 s) with no step saying so.
        state.steps.append(
            Step(
                step_no=state.next_step_no(),
                turn_no=state.turn_no,
                node="answer",
                ok=False,
                summary=str(exc)[:200] or type(exc).__name__,
                error=type(exc).__name__,
                started_at=started_at,
                duration_ms=spent.ms,
                thinking=spent.thinking,
                prompt_version=prompt.tag,
            )
        )
        raise

    answer = result.content or ""
    if not answer.strip() and spent.thinking:
        # Empty content with thinking on means the model spent the call on
        # reasoning and never wrote the answer: measured 23 Sep, 374
        # completion tokens and not one character of content, on a turn whose
        # tools had all succeeded. Asking again with thinking off is a
        # different request -- there is no reasoning channel for it to
        # disappear into -- and it is cheap, because the work is already
        # done and only the writing is missing. Once: a model that answers
        # nothing twice is not going to answer the third time, and the turn
        # still owes the person a reply.
        logger.warning("cevap bos dondu; dusunme kapali bir kez daha deneniyor")
        with contextlib.suppress(LlmError):
            retry = await _chat(
                state,
                deps,
                [{"role": "system", "content": system}, *_answer_messages(state)],
                tools=None,
                thinking=False,
                tag=f"answer-retry@{prompt.tag}",
                spent=spent,
                last_chance=True,
            )
            state.usage = state.usage + retry.usage
            answer = retry.content or ""
    if not answer.strip():
        # The failure `AgentConfig` names in its own docstring: an open-weights
        # model that plans forever and never answers, or returns only a
        # `reasoning` field. Reported as a truncation, because a turn that
        # produced no answer did not finish -- whatever the `done` event says.
        state.truncated_reason = state.truncated_reason or "llm_error"
        answer = "Model bu tur icin bir cevap uretmedi. Soruyu daraltip tekrar deneyin."
    state.messages.append({"role": "assistant", "content": answer})
    state.answer_indices.append(len(state.messages) - 1)
    state.usage = state.usage + result.usage
    state.steps.append(
        Step(
            step_no=state.next_step_no(),
            turn_no=state.turn_no,
            node="answer",
            ok=True,
            summary=answer[:200],
            started_at=started_at,
            duration_ms=spent.ms,
            tokens=result.usage,
            thinking=spent.thinking,
            prompt_version=prompt.tag,
        )
    )

    for chunk in _sentences(answer):
        await emit(deps.sink, state.session_id, "answer_delta", text=chunk)
    return answer


def _answer_messages(state: SessionState) -> list[dict[str, Any]]:
    """The conversation for the answer call, not ending on the model's own
    words.

    The loop's last message is usually the observe step's note ("Veriler
    hazir."). A request ending with an assistant message asks some chat
    templates to *continue* that message rather than answer: measured 22 Sep
    on DeepSeek-V4.1-Flash, the answer came back as 16 tokens of nonsense
    with no reasoning, while the same request ending on the tool result
    answered properly. The note carries nothing the tool messages do not.
    """
    messages = model_messages(state)
    while (
        messages
        and messages[-1].get("role") == "assistant"
        and not messages[-1].get("tool_calls")
    ):
        messages = messages[:-1]
    return messages


@dataclass
class _Spent:
    """What one model call used, however it ended. The step card reports it:
    a call that timed out still took its seconds, and after a retry the step
    is the whole of it, not only the attempt that answered."""

    seconds: float = 0.0
    thinking: bool = False

    @property
    def ms(self) -> int:
        return int(self.seconds * 1000)


async def _chat(
    state: SessionState,
    deps: AgentDeps,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    thinking: bool,
    tag: str,
    spent: _Spent,
    last_chance: bool = False,
    max_tokens: int | None = None,
) -> ChatResult:
    """One model call held to the turn's budget.

    The call's allowance is what is left of `total_budget_s`, worked out once
    -- for the answer (`last_chance`) never less than `answer_min_s` -- and
    every attempt comes out of that one allowance, so a retry does not get a
    fresh floor. A planning or observing attempt is also capped at the client
    timeout, which leaves the turn time for what comes after it. The answer's
    thinking-off attempt is not capped: nothing comes after it, and in the
    third 22 Sep run a 45 s cap ended it with a minute of the budget unused.

    A thinking-on attempt that times out is tried once more with thinking off
    and a capped output: a different request, which a blind retry of the same
    one is not -- thinking is where the time goes, about four times the tokens
    by our own measurement. The retry runs whenever the client could still
    start an attempt (`MIN_ATTEMPT_S`), and for the answer the thinking-on
    attempt is skipped outright when the model's recent speed could not
    produce `answer_thinking_tokens` inside it. Every second an attempt used
    is charged to the turn, and to `spent`, whether it answered or not.
    """
    config = deps.config
    cap = deps.settings.llm_timeout_s
    floor = config.answer_min_s if last_chance else 0.0
    allowance = max(config.total_budget_s - state.active_s, floor)

    # What this call has spent, not what the whole retry pair has. `spent` is
    # shared with the caller's earlier `_chat` (the empty-answer retry in
    # `_answer`, the uncapped retry in `_llm_step`), and `allowance` is
    # recomputed from `state.active_s`, which that earlier attempt already
    # increased. Subtracting it again charged it twice: measured, the answer
    # retry was handed -15 s once the turn had spent 125 s, which `llm._call`
    # turns into `LlmTimeoutError` and `_answer` then suppresses -- so the
    # retry silently did not happen on exactly the late turns it exists for.
    already = spent.seconds

    def left() -> float:
        return allowance - (spent.seconds - already)

    async def attempt(
        think: bool, max_tokens: int | None, timeout: float
    ) -> ChatResult:
        spent.thinking = think
        started = deps.clock()
        try:
            result = await asyncio.to_thread(
                partial(
                    deps.llm.chat,
                    messages,
                    tools=tools,
                    thinking=think,
                    max_tokens=max_tokens,
                    tag=tag,
                    session_id=state.session_id,
                    timeout_s=timeout,
                )
            )
        except BaseException:
            elapsed = deps.clock() - started
            spent.seconds += elapsed
            state.active_s += elapsed
            raise
        spent.seconds += result.duration_ms / 1000
        state.active_s += result.duration_ms / 1000
        return result

    window = min(cap, allowance - config.min_call_s)
    think_first = thinking and window >= config.min_call_s
    if think_first and last_chance:
        think_first = _fast_enough(state, config.answer_thinking_tokens, window)
    if think_first:
        try:
            return await attempt(True, max_tokens, window)
        except LlmTimeoutError:
            # What the fallback needs is what the client needs to start an
            # attempt at all, not a whole `min_call_s`. The thinking-on
            # attempt is capped at `allowance - min_call_s` and then overruns
            # that cap by however long the timeout itself takes to fire, so
            # `left()` lands just under `min_call_s` every time a real clock
            # is involved. Measured: with the answer's allowance at or below
            # `llm_timeout_s + min_call_s` -- a turn that has already spent
            # 125 s of its 180 -- the fallback never ran at all, which is
            # exactly the turn that needed it (22 Sep, third run).
            if left() < MIN_ATTEMPT_S:
                raise
    fallback = config.fallback_max_tokens if thinking else None
    if max_tokens is not None:
        fallback = max_tokens if fallback is None else min(fallback, max_tokens)
    return await attempt(
        False,
        fallback,
        left() if last_chance else min(cap, left()),
    )


#: How many of the session's latest model calls `_recent_rate` averages over.
RATE_SAMPLE = 4


def _recent_rate(state: SessionState) -> float | None:
    """Completion tokens per second over the session's latest model calls.

    Measured over each call's whole duration, prompt processing included: the
    rate that decides whether an answer fits in a window. Pooled rather than
    averaged per call, so a five-token reply that spent two seconds reading a
    long prompt does not pass for the model being slow. None until a model
    call has answered.
    """
    calls = [
        step.tokens
        for step in state.steps
        if step.node in ("plan", "observe", "answer")
        and step.ok
        and step.tokens is not None
        and step.tokens.duration_ms > 0
    ][-RATE_SAMPLE:]
    seconds = sum(usage.duration_ms for usage in calls) / 1000
    tokens = sum(usage.completion_tokens for usage in calls)
    if seconds <= 0 or tokens <= 0:
        return None
    return tokens / seconds


def _fast_enough(state: SessionState, tokens: int, window_s: float) -> bool:
    """Could the model, at its recent speed, write `tokens` in `window_s`?
    With nothing measured yet, assume it can: the configured default holds."""
    rate = _recent_rate(state)
    return rate is None or rate * window_s >= tokens


async def _verify(state: SessionState, deps: AgentDeps, answer: str) -> None:
    """Check the answer against the tool output, then publish the verdict.

    A verifier that raises must not cost the turn its answer. The answer is
    already written at this point, and checking it is the last thing that
    happens; a crash here -- a regex on text nobody anticipated, an injected
    verifier calling a model that is down -- would throw away finished work
    to report a doubt. So a failure is logged and read as "nothing to flag".
    """
    started_at = datetime.now(UTC)
    # Run off the event loop. `AnswerVerifier` is rule-based and quick, but a
    # verifier is injected, and one that calls a model would otherwise freeze
    # the interface for the length of that call.
    try:
        verification = await asyncio.to_thread(deps.verifier.verify, answer, state)
    except Exception:
        logger.exception("dogrulayici calismadi; cevap oldugu gibi birakildi")
        verification = VerificationResult(passed=True)
    state.verification = verification
    state.steps.append(
        Step(
            step_no=state.next_step_no(),
            turn_no=state.turn_no,
            node="verify",
            ok=verification.passed,
            summary=f"{len(verification.flags)} bayrak",
            started_at=started_at,
        )
    )
    await emit(
        deps.sink,
        state.session_id,
        "verification",
        passed=verification.passed,
        flags=[flag.model_dump() for flag in verification.flags],
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


#: What an unanswered call's tool message says, by why the turn stopped.
_ABANDONED_DETAIL = {
    "step_limit": "Adim sinirina ulasildi.",
    "time_budget": "Sure butcesi doldu.",
}


def _answer_unanswered_calls(state: SessionState, detail: str, error: str) -> None:
    """Write a tool message for every call in the history that has none.

    The step limit and the time budget both stop the loop while an assistant
    message asking for a tool is already in the history, and a turn that dies
    mid-step leaves one too. Leaving it unanswered makes the very next request
    a protocol error, so the turn would not merely be truncated, it would fail
    on the final answer and take the rest of the session with it. Scanning
    the history rather than just `pending_calls` makes this a repair for the
    invariant itself, whatever broke it. The repair steps are stamped with
    `state.turn_no`, which `run_turn` has not yet advanced when it repairs a
    dead turn: they belong to that turn.
    """
    answered = {
        message["tool_call_id"]
        for message in state.messages
        if message.get("role") == "tool"
    }
    for message in list(state.messages):
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            if not call_id or call_id in answered:
                continue
            text = f"[HATA] Bu arac calistirilmadi. {detail}"
            state.messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": text}
            )
            state.steps.append(
                Step(
                    step_no=state.next_step_no(),
                    turn_no=state.turn_no,
                    node="act",
                    tool=(call.get("function") or {}).get("name"),
                    ok=False,
                    summary=text,
                    error=error,
                    started_at=datetime.now(UTC),
                )
            )
            answered.add(call_id)
    state.pending_calls.clear()


def _guard(state: SessionState, config: AgentConfig) -> str | None:
    """Should the loop stop before doing more work?"""
    if state.steps_this_turn >= config.max_steps:
        return "step_limit"
    if state.active_s >= config.total_budget_s:
        return "time_budget"
    return None


def _call_signature(call: ToolCall) -> str:
    payload = json.dumps(call.arguments or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(f"{call.name}:{payload}".encode()).hexdigest()[:12]


def _short_validation_error(exc: ValidationError) -> str:
    """Only the first couple of problems: the model fixes one thing at a time."""
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error.get("loc", ()))
        parts.append(f"{location}: {error.get('msg')}")
    return "; ".join(parts)


async def _record_arg_failure(
    state: SessionState, deps: AgentDeps, call: ToolCall, detail: str, finish
) -> None:
    """Feed a bad argument back once, then stop asking.

    Counted per tool name. Every tool call the model makes carries a fresh id,
    so a counter keyed on the id would read zero on exactly the retry it is
    supposed to catch and the model could fail the same arguments forever.
    """
    tries = state.arg_retries.get(call.name, 0)
    state.arg_retries[call.name] = tries + 1
    if tries >= deps.config.max_arg_retries:
        message = (
            f"[HATA] {call.name} argumanlari yine gecersiz ({detail}). "
            "Bu araci bu argumanlarla tekrar deneme."
        )
        await finish(message, ok=False, summary=message, error="invalid_args")
        return
    message = f"[HATA] Arguman hatasi: {detail}. Duzeltip tekrar dene."
    await finish(message, ok=False, summary=message, error="invalid_args")


def _merge_context(
    state: SessionState, ctx: ToolContext, result: ToolResult
) -> tuple[list[str], set[str]]:
    """Move the tool's writes into the session, checking the promises it made.

    Returns the warnings, and the citation keys this step was the first to add
    -- the quality gate needs the second half to undo a rejected table without
    dropping a source an earlier step is still relying on.

    The most important check:
    when a tool adds a column to an existing table, the columns that were
    already there must come back byte for byte. "Add a column without breaking
    the table" is the scenario's second question, and a silent change to an
    existing column would be exactly the failure a jury would catch.
    """
    warnings: list[str] = []
    if not result.ok:
        # Same rule as a timed-out step: a tool that reports failure does not
        # get to leave half a table, a dangling lineage entry or a citation
        # for data nobody can see.
        return warnings, set()

    for ref, frame in ctx.written.items():
        previous = state.tables.get(ref)
        if previous is not None:
            shared = [column for column in previous.columns if column in frame.columns]
            try:
                pd.testing.assert_frame_equal(
                    previous[shared].reset_index(drop=True),
                    frame[shared].reset_index(drop=True),
                    check_dtype=False,
                )
            except AssertionError:
                warnings.append(
                    f"{ref} tablosunda mevcut sutunlar degisti; bu beklenmiyordu."
                )
        state.tables[ref] = frame

    if ctx.new_active and ctx.new_active in state.tables:
        state.active_table = ctx.new_active

    for column in result.columns:
        if result.table_ref:
            state.lineage[f"{result.table_ref}.{column.name}"] = column

    known_sources = {source.key() for source in state.sources}
    state.add_sources(list(result.sources))
    added_sources = {source.key() for source in state.sources} - known_sources

    for warning in result.warnings:
        if warning not in state.warnings:
            state.warnings.append(warning)

    for chart in ctx.written_charts:
        state.charts.append(chart)
    # `ToolResult.artifacts` is the other half of the same contract -- one of
    # the eight shared types both lanes build against -- and it was the half
    # nothing read. A tool handing a chart back that way lost it without a
    # word: not in the session, not on the channel, not in the answer.
    state.charts.extend(
        artifact for artifact in result.artifacts if artifact.type == "chart"
    )

    return warnings, added_sources


async def _emit_result_events(
    state: SessionState,
    deps: AgentDeps,
    result: ToolResult,
    ctx: ToolContext,
) -> None:
    """Tell the interface what the step produced.

    Nothing at all when the step failed. `_merge_context` already refuses to
    put a failed tool's writes in the session, and publishing its sources or
    its quality card anyway would leave the interface showing a citation for
    data the session does not have.
    """
    if not result.ok:
        return

    if result.table_ref and result.table_ref in state.tables:
        frame = state.tables[result.table_ref]
        await emit(
            deps.sink,
            state.session_id,
            "table_update",
            table_ref=result.table_ref,
            columns=[str(column) for column in frame.columns],
            rows=preview_rows(frame),
            new_columns=[column.name for column in result.columns],
        )

    if result.quality is not None:
        await emit(
            deps.sink,
            state.session_id,
            "quality_report",
            **result.quality.model_dump(mode="json"),
        )

    if result.sources:
        await emit(
            deps.sink,
            state.session_id,
            "sources",
            sources=[source.model_dump(mode="json") for source in result.sources],
        )

    for chart in [*ctx.written_charts, *result.artifacts]:
        if _chart_table_ref(chart) == "" and not isinstance(chart, dict):
            # An `Artifact` names its type; only the charts are drawn.
            if getattr(chart, "type", "chart") != "chart":
                continue
        await emit(deps.sink, state.session_id, "chart", **_chart_payload(chart))


async def _post_run_approvals(
    state: SessionState,
    deps: AgentDeps,
    spec: Any,
    result: ToolResult,
    step_no: int,
    *,
    active_before: str | None,
    added_sources: set[str],
) -> None:
    """The two gates that can only be judged after seeing what came back.

    Neither runs on a failed step. A quality card about a table that was never
    merged has nothing to cancel, and its `cancel` branch would go looking for
    the newest table it could find -- which is the previous, perfectly good one.
    """
    if not result.ok:
        return

    if spec.returns_candidates:
        request = approval_rules.needs_series_choice(
            result, state, step_no, ratio=deps.config.ambiguity_ratio
        )
        if request is not None:
            decision = await _ask(state, deps, request)
            message = approval_rules.decision_message(request, decision)
            if message:
                state.messages.append({"role": "user", "content": message})

    if result.quality is not None:
        request = approval_rules.needs_quality_gate(
            result, state, step_no, missing_pct=deps.config.quality_missing_pct
        )
        if request is not None:
            decision = await _ask(state, deps, request)
            if decision.choice == approval_rules.CHOICE_CANCEL and result.table_ref:
                await _discard_table(
                    state,
                    deps,
                    result.table_ref,
                    active_before=active_before,
                    added_sources=added_sources,
                )
                state.messages.append(
                    {
                        "role": "user",
                        "content": "[Onay] Kalite nedeniyle tablo iptal edildi.",
                    }
                )
            else:
                message = approval_rules.decision_message(request, decision)
                if message:
                    state.messages.append({"role": "user", "content": message})


async def _discard_table(
    state: SessionState,
    deps: AgentDeps,
    ref: str,
    *,
    active_before: str | None,
    added_sources: set[str],
) -> None:
    """Undo a table a person rejected, all of it.

    Popping the frame was only the visible part. What stayed behind mattered
    more: the lineage rows still described the rejected columns, and the
    citations still listed the series they came from -- and answer.md tells the
    model to cite from exactly that list, so the jury could read the series it
    had just rejected back in the answer.

    The reference is retired rather than released. The gate always rejects the
    newest table, which is precisely the number `new_table_ref` counts from, so
    reissuing it is the normal outcome and the traceability panel would then
    show one card's lineage under another card's name.
    """
    state.tables.pop(ref, None)
    state.retired_tables.add(ref)

    for key in [key for key in state.lineage if key.startswith(f"{ref}.")]:
        del state.lineage[key]

    state.charts = [chart for chart in state.charts if _chart_table_ref(chart) != ref]

    if added_sources:
        state.sources = [
            source for source in state.sources if source.key() not in added_sources
        ]

    if state.active_table == ref:
        # Back to what the person was working on, not to nothing. Clearing it
        # would make the model rebuild a table that is still right there.
        state.active_table = active_before if active_before in state.tables else None

    await emit(
        deps.sink,
        state.session_id,
        "table_update",
        table_ref=ref,
        removed=True,
        reason="quality_cancel",
    )


def _chart_payload(chart: Any) -> dict[str, Any]:
    """One chart, as something the event channel can carry.

    `ToolContext.add_chart` takes `Any` and the codebase documents three
    carriers, so reading `model_dump` off the object and shipping `{}` when it
    is missing turned a plain dict into an empty chart event. And the field is
    named after a library whose figures do not serialise: `plotly_json` from an
    ordinary `px.line` holds numpy arrays, which raised out of the emitter --
    past every handler, after `done` had already been sent.
    """
    if isinstance(chart, dict):
        payload = dict(chart)
    elif hasattr(chart, "model_dump"):
        payload = chart.model_dump(mode="json")
    else:  # pragma: no cover - not a carrier the codebase produces
        payload = {"chart_ref": chart_ref_of(chart)}
    return {str(key): json_safe(value) for key, value in payload.items()}


def _chart_table_ref(chart: Any) -> str:
    """The table a chart is drawn from, however the chart is carried."""
    if isinstance(chart, dict):
        return str(chart.get("table_ref") or "")
    return str(getattr(chart, "table_ref", "") or "")


async def _ask(
    state: SessionState, deps: AgentDeps, request: ApprovalRequest
) -> ApprovalDecision:
    """Publish the question, wait for the answer, record it.

    Time spent here does not count against the turn budget. A person reading an
    approval card is not the agent being slow, and charging them for it would
    time out the very interaction we built the gate for.
    """
    await emit(
        deps.sink,
        state.session_id,
        "approval_request",
        **request.model_dump(mode="json"),
    )
    started_at = datetime.now(UTC)

    waiting_from = deps.clock()
    try:
        if deps.approve is None:
            decision = await approval_rules.auto_default(request)
        else:
            decision = await deps.approve(request)
    finally:
        # In a `finally` because a card that times out is still a person who
        # was kept waiting. Charged to the turn budget instead, a minute at an
        # approval card ended the turn the gate exists to make possible.
        state.approval_wait_s += deps.clock() - waiting_from

    decision = approval_rules.settle(request, decision)
    state.approvals.append(decision)
    state.steps.append(
        Step(
            step_no=state.next_step_no(),
            turn_no=state.turn_no,
            node="approval",
            ok=True,
            summary=(
                f"{request.kind}: otomatik varsayilan '{decision.choice}'"
                if decision.automatic
                else f"{request.kind}: kullanici '{decision.choice}' secti"
            ),
            started_at=started_at,
        )
    )
    # The card opening is an event; the card closing has to be one too. The
    # JSONL sink exists so a session can be replayed, and a replay that shows
    # two gates opening and nothing ever answering them is not the session.
    await emit(
        deps.sink,
        state.session_id,
        "approval_decision",
        request_id=decision.request_id,
        kind=request.kind,
        choice=decision.choice,
        note=decision.note,
        automatic=decision.automatic,
    )
    return decision


#: How many of a turn's tool warnings go into the answer prompt by name.
MAX_ANSWER_WARNINGS = 10


def _collected_warnings(state: SessionState) -> list[str]:
    """What the answer has to mention: tool warnings first, then failures.

    A tool saying "weekly series averaged to monthly" changes what the numbers
    mean; an answer that does not repeat it is quietly wrong. Failed steps come
    after, so the answer can be honest about what did not run.

    The first warnings are kept, not the last: `get_series` puts its unit
    conversions first, and keeping the tail let one call's coverage notes and
    a later deflate push them out. What does not fit is counted, not dropped
    silently -- the full text is still in the turn's tool messages.
    """
    failures = [
        f"{step.tool or step.node}: {step.error}"
        for step in state.steps
        # This turn's failures only. `state.warnings` is cleared at the top of
        # every turn and `state.steps` is not, so without the turn check the
        # two halves of this list describe different windows -- and answer.md
        # tells the model to open with what went wrong, so turn 3 would apologise
        # for a tool that failed in turn 1.
        if step.error and step.turn_no == state.turn_no
    ]
    # Capped separately, so failed steps cannot push out the unit conversion
    # the answer is obliged to mention.
    warnings = list(dict.fromkeys(state.warnings))
    kept = warnings[:MAX_ANSWER_WARNINGS]
    if len(warnings) > len(kept):
        kept.append(
            f"(+{len(warnings) - len(kept)} uyari daha; tam metni arac mesajlarinda)"
        )
    return [*kept, *failures[-3:]]


def _truncated_text(reason: str | None) -> str:
    return {
        "step_limit": "Adim sinirina ulasildi; analiz tamamlanmadi.",
        "time_budget": "Sure butcesi doldu; analiz tamamlanmadi.",
        "repeat_loop": "Ayni arac cagrisi tekrarlandi; dongu durduruldu.",
        "llm_error": "Model cagrisi basarisiz oldu.",
        None: "yok",
    }.get(reason, "yok")


def _append_flags(answer: str, verification: VerificationResult | None) -> str:
    """Show unverifiable statements instead of quietly rewriting the answer.

    A rewrite costs another thinking-on call on the critical path and risks
    making a good answer worse. Showing the flag tells the jury the system
    audits itself, which is worth more than hiding the doubt.
    """
    if verification is None or verification.passed or not verification.flags:
        return answer
    lines = "\n".join(f"- {flag.text}" for flag in verification.flags)
    return f"{answer}\n\n**Dogrulanamayan ifadeler**\n{lines}"


def _sentences(text: str, chunk: int = 160) -> list[str]:
    """Split the answer for the delta stream.

    A pseudo-stream: the answer is already complete by the time it is split.
    Real token streaming is a later change and it does not alter this contract,
    because the interface consumes the same event either way.
    """
    if not text:
        return []
    out: list[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= chunk:
            out.append(text[start:])
            break
        # Cut at a space and keep it. Rebuilding each chunk with `strip()` ate
        # exactly one space per boundary, so a consumer joining the deltas got
        # two words glued together -- the stream and the answer disagreed.
        cut = text.rfind(" ", start, start + chunk)
        cut = cut + 1 if cut > start else start + chunk
        out.append(text[start:cut])
        start = cut
    return out


# --------------------------------------------------------------------------
# Headless entry point
# --------------------------------------------------------------------------


class JsonlSink:
    """Writes each event as one JSON line.

    The line-per-event shape is the point: a demo can tail it, a test can parse
    it, and the evaluation runner can replay a whole session from the file. It
    is also the cheapest proof that nothing unserialisable, a DataFrame above
    all, ever reaches an event payload.
    """

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    async def emit(self, event: Any) -> None:
        self.stream.write(
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
        )
        self.stream.flush()


#: The scripted turn `--fake` replays: search the catalog, pull the series,
#: derive a column, then answer. It walks the same path the scenario's first
#: question takes, which is what makes this flag a smoke test and not a toy.
def _fake_script(question: str) -> list[Any]:
    from lakehouse_agent.testing.fake_llm import FakeLLM

    return [
        FakeLLM.tool_call(
            "search_catalog",
            {"query": question[:60] or "konut kredisi faiz orani"},
            call_id="call_1",
            content="Once katalogda ilgili seriyi ariyorum.",
        ),
        FakeLLM.tool_call(
            "get_series",
            {
                "series_ids": ["evds:TP.KTF12"],
                "start": "2021-01",
                "end": "2025-12",
                "into": "new",
            },
            call_id="call_2",
        ),
        FakeLLM.tool_call(
            "transform_table",
            {"op": "diff", "column": "TP_KTF12"},
            call_id="call_3",
        ),
        FakeLLM.text("Veriler hazir, cevabi yazabilirim."),
        FakeLLM.text(
            "Faiz serisi 2021-01 ile 2025-12 arasinda aylik olarak getirildi ve "
            "aylik farki hesaplandi. Ayrintilar tabloda. Kaynak: evds:TP.KTF12."
        ),
    ]


def _fake_deps(questions: list[str]) -> AgentDeps:
    """Build a fully offline dependency set.

    The import is inside the function on purpose. `lakehouse_agent.testing`
    ships in the installed package so this flag, the eval runner and the Docker
    image can all reach it, but no production code path may depend on the
    fakes, and a module-level import here would make one.

    One scripted turn per question, in order, so several `--question`s replay
    as several turns of the same session.
    """
    from lakehouse_agent.testing.fake_llm import FakeLLM
    from lakehouse_agent.testing.fake_tools import make_fake_registry

    script = [item for question in questions for item in _fake_script(question)]
    return AgentDeps(
        llm=FakeLLM(script),
        registry=make_fake_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        sink=JsonlSink(),
    )


def _live_deps() -> AgentDeps:
    """Real model, real tools, synthetic data (`FakeLake`).

    The one thing the offline suite cannot tell us: whether the model actually
    picks the right tool, fills its arguments from the schema, and writes
    sensible Turkish. The tools are the real ones -- the fake registry's
    `transform_table` knows two operations and the prompt asks for eight --
    and the data is `FakeLake`'s, with planted answers, so a failure here is
    the model or the prompt, never the pool's data.
    """
    from dotenv import find_dotenv, load_dotenv

    from lakehouse_agent.testing.fake_lake import FakeLake
    from lakehouse_agent.tools.toolset import make_registry

    # An entrypoint, so it is allowed to load .env. `usecwd` because otherwise
    # dotenv searches next to this file instead of next to the project.
    load_dotenv(find_dotenv(usecwd=True))

    from lakehouse_agent.config import get_agent_settings

    settings = get_agent_settings()
    return AgentDeps(
        llm=MiaLLM(settings),
        registry=make_registry(),
        settings=settings,
        lake=FakeLake(),
        sink=JsonlSink(),
    )


def _real_deps() -> AgentDeps:
    """Real model, real tools, real data: what the demo runs.

    No fakes are imported here, so this mode cannot put a synthetic number on
    screen. The lake reads the committed gold data and, on the first search,
    embeds the pool catalog once (cached on disk afterwards).
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    from lakehouse_agent.config import get_agent_settings
    from lakehouse_agent.tools.lake_adapter import LakeAdapter
    from lakehouse_agent.tools.toolset import make_registry

    settings = get_agent_settings()
    return AgentDeps(
        llm=MiaLLM(settings),
        registry=make_registry(),
        settings=settings,
        lake=LakeAdapter(),
        sink=JsonlSink(),
    )


def _summarise(state: SessionState, result: TurnResult | None) -> None:
    """A human-readable recap on stderr, so stdout stays pure JSONL.

    Also printed when the turn failed (`result` None): the steps that did run
    and the tables they built are still the session's, and a bare error line
    hid them -- the 22 Sep live run lost a finished analysis that way.
    """
    print("", file=sys.stderr)
    for step in state.steps:
        if step.turn_no != state.turn_no:
            continue
        mark = "ok" if step.ok is not False else "HATA"
        detail = f" [{step.tool}]" if step.tool else ""
        print(
            f"  {step.step_no:>2}. {step.node:<8}{detail:<18} {mark:<4} "
            f"{step.duration_ms:>6} ms  {step.summary[:60]}",
            file=sys.stderr,
        )
    for ref, frame in state.tables.items():
        marker = " (aktif)" if ref == state.active_table else ""
        print(f"\n  TABLO {describe_table(ref, frame)}{marker}", file=sys.stderr)
    _print_sources(state)
    if result is None:
        print(
            f"\n  kesinti {state.truncated_reason or 'yok'} | CEVAP YOK\n",
            file=sys.stderr,
        )
        return
    print(
        f"\n  sure {result.duration_ms} ms | token "
        f"{result.usage.prompt_tokens}+{result.usage.completion_tokens} | "
        f"onay {result.approvals} | kesinti {result.truncated_reason or 'yok'}",
        file=sys.stderr,
    )
    print(f"\n  CEVAP: {result.answer}\n", file=sys.stderr)


#: Sources listed under a recap. Past this the list stops being read, and a
#: search alone can add five.
RECAP_SOURCES = 12


def _print_sources(state: SessionState) -> None:
    """Every source the session looked at, in full, on the screen.

    The addresses in an answer are whatever the model wrote, and a model
    shortens a long one ("tcmb.gov.tr/.../duy2023-25"): readable, not
    clickable. The full list is already carried by the `sources` event -- the
    traceability panel will draw it -- but until there is an interface the
    only place a person sees it is here. Printed beside the answer rather
    than inside it: the answer is the model's text, and appending to it would
    also hand the verifier its own citations back.

    "Bakilan", not "kullanilan": a search contributes every hit it returned,
    and the answer may rest on two of them.
    """
    if not state.sources:
        return
    print("\n  KAYNAKLAR (bakilan)", file=sys.stderr)
    for source in state.sources[:RECAP_SOURCES]:
        name = f"  {source.name}" if getattr(source, "name", None) else ""
        print(f"  - {source.series_id or source.url}{name}", file=sys.stderr)
    more = len(state.sources) - RECAP_SOURCES
    if more > 0:
        print(f"  ... ve {more} kaynak daha", file=sys.stderr)


DEFAULT_QUESTION = "Konut kredisi faizi nasil seyretti?"


def _cli(argv: list[str] | None = None) -> int:
    """`python -m lakehouse_agent.agent.loop --real --question "..." [--question ...]`

    Three modes, and no default, because they differ in what is real. `--fake`
    is offline and deterministic: the smoke test the suite relies on. `--live`
    is the real model and tools over synthetic data, to judge the model alone.
    `--real` is the real model over the real tools and data: the demo.

    Each `--question` is one turn of the same session, so a follow-up such as
    "tabloyu bozmadan ..." sees the table the previous question built.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Agent dongusunu calistirir")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--fake",
        action="store_true",
        help="Sahte model ve sahte araclar; ag yok, sonuc her zaman ayni",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Gercek MIA modeli ve araclar, sentetik veri (FakeLake); MIA_API_KEY gerekli",
    )
    mode.add_argument(
        "--real",
        action="store_true",
        help="Gercek MIA modeli, gercek araclar ve veri; MIA_API_KEY gerekli",
    )
    parser.add_argument(
        "--question",
        action="append",
        help="Sorulacak soru; birden cok verilirse ayni oturumda sirayla sorulur",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Etkilesimli oturum: soru ve onay klavyeden alinir, ekran okunur",
    )
    parser.add_argument(
        "--events",
        metavar="DOSYA",
        help="JSONL olay akisini bu dosyaya yazar (sohbet modunda varsayilan)",
    )
    args = parser.parse_args(argv)
    questions = args.question or ([] if args.chat else [DEFAULT_QUESTION])

    if args.chat and not sys.stdin.isatty():
        print(
            "--chat bir terminal gerektiriyor; stdin baska yere bagli.",
            file=sys.stderr,
        )
        print("Basli olmayan kosum icin --question kullanin.", file=sys.stderr)
        return 2

    state = initial_state("cli")
    try:
        if args.real:
            deps = _real_deps()
        elif args.live:
            deps = _live_deps()
        else:
            deps = _fake_deps(questions)
    except Exception as exc:  # noqa: BLE001 - a missing key is a normal outcome
        print(f"Ayarlar okunamadi: {exc}", file=sys.stderr)
        print("MIA_API_KEY .env dosyasinda tanimli mi?", file=sys.stderr)
        return 2

    terminal = None
    stack = contextlib.ExitStack()
    with stack:
        if args.events or args.chat:
            path = Path(args.events) if args.events else _session_log(deps)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("w", encoding="utf-8"))
            deps.sink = JsonlSink(handle)
            if args.chat:
                print(f"olaylar: {path}", file=sys.stderr)
        if args.chat:
            # Imported here and nowhere else: a headless run must not load
            # the interface, and the engine must not depend on it.
            from lakehouse_agent.agent.console import Terminal

            terminal = Terminal(tables=state.tables.get)
            deps.sink = _Both(deps.sink, terminal)
            deps.approve = terminal.approve
        return asyncio.run(_run_cli(state, deps, questions, terminal, args.fake))


def _session_log(deps: AgentDeps) -> Path:
    """Where a chat session's JSONL goes when nobody named a file."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return deps.settings.adhoc_dir / "sessions" / f"{stamp}.jsonl"


class _Both:
    """Two sinks as one: the file keeps the record, the screen shows it."""

    def __init__(self, *sinks: Any) -> None:
        self._sinks = [sink for sink in sinks if sink is not None]

    async def emit(self, event: Any) -> None:
        for sink in self._sinks:
            await sink.emit(event)


async def _questions(given: list[str], terminal: Any) -> AsyncIterator[str]:
    """The questions given on the command line, then whatever is typed."""
    for question in given:
        yield question
    while terminal is not None:
        question = await terminal.ask()
        if question is None:
            return
        yield question


async def _run_cli(
    state: SessionState,
    deps: AgentDeps,
    given: list[str],
    terminal: Any,
    fake: bool,
) -> int:
    """One event loop for every question.

    `asyncio.run` per question waited at each exit for any tool thread a step
    timeout had abandoned, which held the next question back for as long as
    that tool kept running.
    """
    code = 0
    async for question in _questions(given, terminal):
        if fake:
            # The scripted model cannot know how many questions will come, so
            # a turn's worth of replies is pushed as each one arrives.
            deps.llm.extend(_fake_script(question))
        try:
            result = await run_turn(state, deps, question)
        except AgentError as exc:
            print(
                json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False),
                file=sys.stderr,
            )
            _summarise(state, None)
            return 1
        if terminal is None:
            _summarise(state, result)
        if not result.answer:
            code = 1
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
