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
import hashlib
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import partial
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
    preview_rows,
)
from lakehouse_agent.agent.prompts import load_prompt, render
from lakehouse_agent.agent.state import (
    ApprovalDecision,
    ApprovalRequest,
    SessionState,
    Step,
    VerificationResult,
    initial_state,
)
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.contracts import ToolResult, Usage
from lakehouse_agent.llm import LLM, LlmError, MiaLLM, ToolCall
from lakehouse_agent.tools.registry import ToolRegistry

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
    max_steps: int = 8

    #: One slow tool must not take the turn with it.
    step_timeout_s: float = 60.0

    #: Total working time. Measured on `active_s`, which excludes time spent
    #: waiting for a human: a jury thinking about an approval is not the agent
    #: being slow.
    total_budget_s: float = 180.0

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
    """Passes everything. The real one arrives with the validation stage."""

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
    verifier: Verifier = field(default_factory=NullVerifier)
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

    started = deps.clock()
    approvals_before = len(state.approvals)
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

            await _llm_step(state, deps, node="observe", question=question)

        await _answer_abandoned_calls(state)
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
        usage=state.usage,
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

    result = await asyncio.to_thread(
        partial(
            deps.llm.chat,
            [{"role": "system", "content": system}, *state.messages],
            tools=deps.registry.openai_tools() or None,
            thinking=thinking,
            tag=f"{node}@{prompt.tag}",
            session_id=state.session_id,
        )
    )

    state.messages.append(result.as_message())
    state.pending_calls = list(result.tool_calls)
    state.usage = state.usage + result.usage
    state.active_s += result.duration_ms / 1000

    state.steps.append(
        Step(
            step_no=state.next_step_no(),
            turn_no=state.turn_no,
            node="plan" if node == "plan" else "observe",
            summary=(result.content or "")[:200],
            ok=True,
            started_at=started_at,
            duration_ms=result.duration_ms,
            tokens=result.usage,
            thinking=thinking,
            prompt_version=prompt.tag,
        )
    )

    if node == "plan" and result.content:
        await emit(deps.sink, state.session_id, "plan", text=result.content)


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
            ok=ok,
            summary=summary,
            error=error,
            duration_ms=int(elapsed * 1000),
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
    )

    request = approval_rules.needs_fetch_approval(spec, args, ctx, state)
    if request is not None:
        try:
            decision = await _ask(state, deps, request)
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
    except Exception as exc:  # noqa: BLE001 - a tool bug must not end the turn
        message = f"[HATA] {call.name} hata verdi: {type(exc).__name__}: {exc}"
        await finish(message, ok=False, summary=message, error=type(exc).__name__)
        return

    # -- merge the writes back -------------------------------------------
    warnings = _merge_context(state, ctx, result, step_no)
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

    await _emit_result_events(state, deps, result, step_no, ctx)

    # -- approvals that depend on what came back --------------------------
    # Guarded for the same reason as the gate above, with one difference: the
    # tool message is already written, so a failure here costs the gate, not
    # the session.
    try:
        await _post_run_approvals(state, deps, spec, result, step_no)
    except Exception as exc:  # noqa: BLE001
        state.warnings.append(f"Onay adimi atlandi ({type(exc).__name__}).")


async def _answer(state: SessionState, deps: AgentDeps, question: str) -> str:
    """Write the final answer. No tools; the work is done by now."""
    prompt = load_prompt("answer")
    flags = state.verification.flags if state.verification else []
    system = render(
        prompt,
        tables_summary=build_tables_summary(state.tables, state.active_table),
        charts=", ".join(chart.chart_ref for chart in state.charts) or "yok",
        sources=format_sources(state.sources),
        warnings=" | ".join(_collected_warnings(state)) or "yok",
        truncated_reason=_truncated_text(state.truncated_reason),
        verification_flags=", ".join(flag.text for flag in flags) or "yok",
    )
    started_at = datetime.now(UTC)

    result = await asyncio.to_thread(
        partial(
            deps.llm.chat,
            [{"role": "system", "content": system}, *state.messages],
            thinking=deps.config.thinking.get("answer", True),
            tag=f"answer@{prompt.tag}",
            session_id=state.session_id,
        )
    )

    answer = result.content or ""
    state.messages.append({"role": "assistant", "content": answer})
    state.usage = state.usage + result.usage
    state.active_s += result.duration_ms / 1000
    state.steps.append(
        Step(
            step_no=state.next_step_no(),
            turn_no=state.turn_no,
            node="answer",
            ok=True,
            summary=answer[:200],
            started_at=started_at,
            duration_ms=result.duration_ms,
            tokens=result.usage,
            thinking=deps.config.thinking.get("answer", True),
            prompt_version=prompt.tag,
        )
    )

    for chunk in _sentences(answer):
        await emit(deps.sink, state.session_id, "answer_delta", text=chunk)
    return answer


async def _verify(state: SessionState, deps: AgentDeps, answer: str) -> None:
    """Check the answer against the tool output, then publish the verdict."""
    started_at = datetime.now(UTC)
    # Run off the event loop: the real verifier makes a model call, and doing
    # that inline would freeze the interface for the length of it.
    verification = await asyncio.to_thread(deps.verifier.verify, answer, state)
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


async def _answer_abandoned_calls(state: SessionState) -> None:
    """Write a tool message for every call the loop never got to run.

    The step limit and the time budget both stop the loop while an assistant
    message asking for a tool is already in the history. Leaving it unanswered
    makes the very next request a protocol error, so the turn would not merely
    be truncated, it would fail on the final answer and take the rest of the
    session with it. Scanning the history rather than just `pending_calls`
    makes this a repair for the invariant itself, whatever broke it.
    """
    answered = {
        message["tool_call_id"]
        for message in state.messages
        if message.get("role") == "tool"
    }
    detail = {
        "step_limit": "Adim sinirina ulasildi.",
        "time_budget": "Sure butcesi doldu.",
    }.get(state.truncated_reason or "", "Tur sonlandi.")

    for message in state.messages:
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
                    error=state.truncated_reason or "abandoned",
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
    state: SessionState, ctx: ToolContext, result: ToolResult, step_no: int
) -> list[str]:
    """Move the tool's writes into the session, checking the promises it made.

    Returns warnings for anything that looked wrong. The most important check:
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
        return warnings

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

    state.add_sources(list(result.sources))
    for warning in result.warnings:
        if warning not in state.warnings:
            state.warnings.append(warning)

    for chart in ctx.written_charts:
        state.charts.append(chart)

    return warnings


async def _emit_result_events(
    state: SessionState,
    deps: AgentDeps,
    result: ToolResult,
    step_no: int,
    ctx: ToolContext,
) -> None:
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

    for chart in ctx.written_charts:
        await emit(
            deps.sink,
            state.session_id,
            "chart",
            **(chart.model_dump(mode="json") if hasattr(chart, "model_dump") else {}),
        )


async def _post_run_approvals(
    state: SessionState, deps: AgentDeps, spec: Any, result: ToolResult, step_no: int
) -> None:
    """The two gates that can only be judged after seeing what came back."""
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
                state.tables.pop(result.table_ref, None)
                if state.active_table == result.table_ref:
                    state.active_table = None
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
    if deps.approve is None:
        decision = await approval_rules.auto_default(request)
    else:
        decision = await deps.approve(request)
    state.approval_wait_s += deps.clock() - waiting_from

    state.approvals.append(decision)
    state.steps.append(
        Step(
            step_no=state.next_step_no(),
            turn_no=state.turn_no,
            node="approval",
            ok=True,
            summary=f"{request.kind}: kullanici '{decision.choice}' secti",
            started_at=started_at,
        )
    )
    return decision


def _collected_warnings(state: SessionState) -> list[str]:
    """What the answer has to mention: tool warnings first, then failures.

    A tool saying "weekly series averaged to monthly" changes what the numbers
    mean; an answer that does not repeat it is quietly wrong. Failed steps come
    after, so the answer can be honest about what did not run.
    """
    failures = [
        f"{step.tool or step.node}: {step.error}" for step in state.steps if step.error
    ]
    # Capped separately. A single tail slice would let four failed steps push
    # out the unit conversion the answer is obliged to mention.
    return [*state.warnings[-4:], *failures[-3:]]


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
    buffer = ""
    for part in text.split(" "):
        buffer = f"{buffer} {part}".strip()
        if len(buffer) >= chunk:
            out.append(buffer)
            buffer = ""
    if buffer:
        out.append(buffer)
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


def _fake_deps(question: str) -> AgentDeps:
    """Build a fully offline dependency set.

    The import is inside the function on purpose. `lakehouse_agent.testing`
    ships in the installed package so this flag, the eval runner and the Docker
    image can all reach it, but no production code path may depend on the
    fakes, and a module-level import here would make one.
    """
    from lakehouse_agent.testing.fake_llm import FakeLLM
    from lakehouse_agent.testing.fake_tools import make_fake_registry

    return AgentDeps(
        llm=FakeLLM(_fake_script(question)),
        registry=make_fake_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        sink=JsonlSink(),
    )


def _live_deps() -> AgentDeps:
    """Real model, fake tools.

    The one thing the offline suite cannot tell us: whether the model actually
    picks the right tool, fills its arguments from the schema, and writes
    sensible Turkish. The tools stay fake so a failure here is unambiguous -- it
    is the model or the prompt, never the data layer.
    """
    from dotenv import find_dotenv, load_dotenv

    from lakehouse_agent.testing.fake_tools import make_fake_registry

    # An entrypoint, so it is allowed to load .env. `usecwd` because otherwise
    # dotenv searches next to this file instead of next to the project.
    load_dotenv(find_dotenv(usecwd=True))

    from lakehouse_agent.config import get_agent_settings

    settings = get_agent_settings()
    return AgentDeps(
        llm=MiaLLM(settings),
        registry=make_fake_registry(),
        settings=settings,
        sink=JsonlSink(),
    )


def _summarise(state: SessionState, result: TurnResult) -> None:
    """A human-readable recap on stderr, so stdout stays pure JSONL."""
    print("", file=sys.stderr)
    for step in state.steps:
        mark = "ok" if step.ok is not False else "HATA"
        detail = f" [{step.tool}]" if step.tool else ""
        print(
            f"  {step.step_no:>2}. {step.node:<8}{detail:<18} {mark:<4} "
            f"{step.duration_ms:>6} ms  {step.summary[:60]}",
            file=sys.stderr,
        )
    print(
        f"\n  sure {result.duration_ms} ms | token "
        f"{result.usage.prompt_tokens}+{result.usage.completion_tokens} | "
        f"onay {result.approvals} | kesinti {result.truncated_reason or 'yok'}",
        file=sys.stderr,
    )
    print(f"\n  CEVAP: {result.answer}\n", file=sys.stderr)


def _cli(argv: list[str] | None = None) -> int:
    """`python -m lakehouse_agent.agent.loop --fake --question "..."`

    Two modes. `--fake` is offline and deterministic: the smoke test the suite
    relies on. `--live` swaps in the real model but keeps the fake tools, which
    is the only way to see whether MIA chooses correctly before the real tools
    exist.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Agent dongusunu tek tur calistirir")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--fake",
        action="store_true",
        help="Sahte model ve sahte araclar; ag yok, sonuc her zaman ayni",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Gercek MIA modeli, sahte araclar; MIA_API_KEY gerekli",
    )
    parser.add_argument("--question", default="Konut kredisi faizi nasil seyretti?")
    args = parser.parse_args(argv)

    state = initial_state("cli")
    try:
        deps = _live_deps() if args.live else _fake_deps(args.question)
    except Exception as exc:  # noqa: BLE001 - a missing key is a normal outcome
        print(f"Ayarlar okunamadi: {exc}", file=sys.stderr)
        print("MIA_API_KEY .env dosyasinda tanimli mi?", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(run_turn(state, deps, args.question))
    except AgentError as exc:
        print(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
        return 1

    _summarise(state, result)
    return 0 if result.answer else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
