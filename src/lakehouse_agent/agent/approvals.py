"""The three places the agent stops and asks a person.

KKB said data quality and human approval are where teams differentiate. The
trap is over-correcting: an agent that asks at every step is unusable in a live
demo. So there are exactly three gates, each tied to a decision a person can
actually make better than the model.

The loop owns all three. Tools declare that they need approval; they never ask.
Keeping the decision in one place means "when do we interrupt" is answerable by
reading one file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.agent.formatting import json_safe
from lakehouse_agent.agent.state import (
    ApprovalDecision,
    ApprovalOption,
    ApprovalRequest,
    SessionState,
)
from lakehouse_agent.contracts import ToolResult

#: Two candidates this close in score are not distinguishable by score alone.
#: A ratio, not an absolute gap: hybrid search fuses ranks into values that are
#: only comparable inside one query, so "within 0.05" means nothing.
DEFAULT_AMBIGUITY_RATIO = 0.9

#: Above this share of missing data, a person should see the table first.
DEFAULT_MISSING_PCT = 10.0

CHOICE_APPROVE = "approve"
CHOICE_CANCEL = "cancel"
CHOICE_CONTINUE = "continue"
CHOICE_CHANGE_ASSUMPTION = "change_assumption"


def next_request_id(state: SessionState) -> str:
    return f"{state.session_id}-{state.turn_no}-{len(state.approvals) + 1}"


def needs_fetch_approval(
    spec: Any, args: BaseModel, ctx: ToolContext, state: SessionState
) -> ApprovalRequest | None:
    """Ask before pulling data that is not in the pool.

    Fetching is the one tool action with a cost outside our process: it hits a
    public service, takes real time and may fail mid-demo. Showing what is
    about to happen is cheap and it is the moment a jury sees the system
    deferring to a human.
    """
    if spec.approval != "fetch":
        return None

    context: dict[str, Any] = {}
    if spec.approval_preview is not None:
        try:
            preview = spec.approval_preview(args, ctx)
        except Exception:  # noqa: BLE001 - a broken preview must not block the ask
            preview = None
        # Shape-checked, not just call-checked. The `except` above covers a
        # preview that raises; one that simply returns a list took the gate
        # down on the next `.get`, which is the same lost approval by a
        # different route.
        context = preview if isinstance(preview, dict) else {}

    source = context.get("source", "dis kaynak")
    detail = context.get("detail", "")
    question = f"{source} uzerinden veri cekilsin mi?"
    if detail:
        question = f"{question} ({detail})"

    return ApprovalRequest(
        request_id=next_request_id(state),
        kind="fetch",
        question=question,
        options=[
            ApprovalOption(id=CHOICE_APPROVE, label="Onayla", detail=detail or None),
            ApprovalOption(id=CHOICE_CANCEL, label="Iptal"),
        ],
        default=CHOICE_APPROVE,
        context=_card_context(context),
        step_no=ctx.step_no,
        created_at=datetime.now(UTC),
    )


def needs_series_choice(
    result: ToolResult,
    state: SessionState,
    step_no: int,
    *,
    ratio: float = DEFAULT_AMBIGUITY_RATIO,
    top_k: int = 4,
) -> ApprovalRequest | None:
    """Ask when the catalog cannot tell two candidates apart.

    The scenario's own trap: "housing loan interest rate" matches both the flow
    series and the stock series. They answer different questions and picking
    wrong produces a confident, wrong analysis. A person settles it in one
    click; the model would just guess.

    Does not fire when there is one clear winner, which is why the scenario
    itself runs without interruptions.
    """
    candidates = result.preview or []
    if len(candidates) < 2:
        return None

    top, second = candidates[0], candidates[1]
    top_score = float(top.get("score") or 0)
    second_score = float(second.get("score") or 0)

    close_scores = top_score > 0 and second_score >= top_score * ratio
    different_unit = top.get("unit") != second.get("unit")
    different_kind = top.get("kind") != second.get("kind")
    if not (close_scores or different_unit or different_kind):
        return None

    options = [
        ApprovalOption(
            id=str(candidate.get("series_id")),
            label=str(candidate.get("name_tr") or candidate.get("series_id")),
            detail=", ".join(
                str(candidate.get(key))
                for key in ("unit", "freq", "kind")
                if candidate.get(key)
            )
            or None,
        )
        for candidate in candidates[:top_k]
    ]
    return ApprovalRequest(
        request_id=next_request_id(state),
        kind="series_choice",
        question="Birden fazla seri sorguya uyuyor. Hangisiyle devam edelim?",
        options=options,
        default=options[0].id,
        context=_card_context({"candidates": candidates[:top_k]}),
        step_no=step_no,
        created_at=datetime.now(UTC),
    )


def needs_quality_gate(
    result: ToolResult,
    state: SessionState,
    step_no: int,
    *,
    missing_pct: float = DEFAULT_MISSING_PCT,
) -> ApprovalRequest | None:
    """Ask when a new table arrives in questionable shape.

    Gaps, duplicate periods and an unknown unit each change what the numbers
    mean. Continuing silently would produce an answer that looks the same as a
    good one, which is the failure mode the trust layer exists to prevent.
    """
    quality = result.quality
    if quality is None:
        return None

    problems: list[str] = []
    # The larger of the two things the report can tell us. `missing_pct` is the
    # richer measure -- it counts NaN cells inside present rows, which gaps
    # alone cannot see -- but the two profilers behind it disagree about its
    # scale: a quarter missing reads 25.0 from the fake and 0.25 from the real
    # one, and no fraction ever exceeds a threshold of ten. Counting periods is
    # coarser but unambiguous, so neither implementation's omission can hide a
    # gap behind the other's.
    total = quality.rows + quality.missing_periods
    gap_pct = (quality.missing_periods / total * 100) if total else 0.0
    share = max(quality.missing_pct, gap_pct)
    if share > missing_pct:
        problems.append(f"eksik veri %{share:.1f}")
    if quality.duplicates:
        problems.append(f"{quality.duplicates} tekrarli donem")
    if quality.unit_guess is None:
        problems.append("birim belirlenemedi")
    if not problems:
        return None

    return ApprovalRequest(
        request_id=next_request_id(state),
        kind="quality",
        question="Yeni tabloda kalite sorunlari var: "
        + ", ".join(problems)
        + ". Devam edelim mi?",
        options=[
            ApprovalOption(id=CHOICE_CONTINUE, label="Devam et"),
            ApprovalOption(
                id=CHOICE_CHANGE_ASSUMPTION,
                label="Varsayimi degistir",
                detail="Analize devam etmeden once varsayimi yaziniz.",
            ),
            ApprovalOption(id=CHOICE_CANCEL, label="Iptal et"),
        ],
        default=CHOICE_CONTINUE,
        context=_card_context(
            {"quality": quality.model_dump(mode="json"), "problems": problems}
        ),
        step_no=step_no,
        created_at=datetime.now(UTC),
    )


def _card_context(context: dict[str, Any]) -> dict[str, Any]:
    """The card's context, as something the event channel can carry.

    An approval card goes out as an `Event`, whose payload must serialise. A
    catalog hit carrying a numpy score or a NaT date would raise inside `emit`
    -- and the loop treats a failed gate as "skip the gate", so an
    unserialisable value did not surface as an error, it silently removed the
    human from the decision.
    """
    return {str(key): json_safe(value) for key, value in context.items()}


def settle(request: ApprovalRequest, decision: ApprovalDecision) -> ApprovalDecision:
    """The answer, checked against the card it answers.

    The decision arrives from outside the engine -- an interface, an eval
    runner, a test. An id that was never on the card is not a choice the person
    could have made, and acting on it would mean doing something nobody
    approved. The default is what the card itself offered, so falling back to
    it is the conservative reading rather than an invention.
    """
    allowed = {option.id for option in request.options}
    fallback = request.default or (
        request.options[0].id if request.options else CHOICE_CANCEL
    )
    note = decision.note
    choice = decision.choice

    if decision.request_id != request.request_id:
        note = f"[uyari] farkli karta ait cevap ({decision.request_id!r}); " + (
            note or "varsayilan uygulandi"
        )
        choice = fallback
    elif choice not in allowed:
        note = f"[uyari] kartta olmayan secenek ({choice!r}); " + (
            note or "varsayilan uygulandi"
        )
        choice = fallback

    if choice == decision.choice and note == decision.note:
        return decision
    return ApprovalDecision(request_id=request.request_id, choice=choice, note=note)


async def auto_default(request: ApprovalRequest) -> ApprovalDecision:
    """Pick the default without asking. Used by tests and the eval runner.

    Never used in a demo: the point of the gate is that a person answers it.
    """
    return ApprovalDecision(
        request_id=request.request_id,
        choice=request.default
        or (request.options[0].id if request.options else CHOICE_CANCEL),
        note="otomatik onay (etkilesimsiz mod)",
    )


def decision_message(
    request: ApprovalRequest, decision: ApprovalDecision
) -> str | None:
    """Turn a human's answer into a line the model can act on.

    Written as a user message rather than a tool result because it is a person
    speaking, and the model should treat it with that weight.
    """
    if request.kind == "series_choice":
        return (
            f"[Onay] Seri secimi: {decision.choice}. "
            "Bu seriyle devam et, tekrar katalog aramasi yapma."
        )
    if request.kind == "quality":
        if decision.choice == CHOICE_CONTINUE:
            return (
                "[Onay] Kalite karti gorulup devam edildi. Cevapta bu sinirlari belirt."
            )
        if decision.choice == CHOICE_CHANGE_ASSUMPTION:
            return f"[Onay] Kullanici varsayimi degistirdi: {decision.note or '-'}"
    return None
