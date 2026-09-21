"""Everything one chat session carries.

`SessionState` is a plain dataclass, not a TypedDict with reducers. The loop is
our own `while`, single process and single threaded per session, so there is no
graph runtime merging concurrent updates and nothing to reduce. A dataclass says
that plainly and is far easier to read in a debugger.

Field names follow docs/contracts.md exactly; the interface renders them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from lakehouse_agent.contracts import (
    CHART_REF_PATTERN,
    TABLE_REF_PATTERN,
    ColumnInfo,
    Source,
    Usage,
)
from lakehouse_agent.llm import ToolCall

NodeName = Literal["plan", "act", "observe", "answer", "verify", "approval"]

TruncatedReason = Literal["step_limit", "time_budget", "repeat_loop", "llm_error"]


class Step(BaseModel):
    """One entry in the step log.

    This is what the interface draws as a live step card and what the
    traceability panel reads afterwards, so it carries both the mechanics
    (which tool, which arguments) and the cost (duration, tokens).
    """

    step_no: int
    turn_no: int
    node: NodeName
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool | None = None
    summary: str = ""
    error: str | None = None
    started_at: datetime
    duration_ms: int = 0
    tokens: Usage | None = None
    thinking: bool | None = None
    table_ref: str | None = None
    prompt_version: str | None = None


class ChartSeries(BaseModel):
    column: str
    axis: Literal["left", "right"] = "left"
    kind: Literal["line", "bar"] = "line"


class ChartSpec(BaseModel):
    """A chart the model asked for. The model describes it; it never draws it."""

    chart_ref: str = Field(..., pattern=CHART_REF_PATTERN)
    table_ref: str = Field(..., pattern=TABLE_REF_PATTERN)
    x: str = "period"
    series: list[ChartSeries] = Field(default_factory=list)
    title: str | None = None
    plotly_json: dict[str, Any] | None = None
    step_no: int | None = None


class ApprovalOption(BaseModel):
    """One button on an approval card.

    `id` is English because code compares it; `label` is Turkish because a
    person reads it. Turkish identifiers invite casefold bugs on the dotted
    and dotless i.
    """

    id: str = Field(
        ...,
        description="approve | cancel | continue | change_assumption veya bir series_id.",
    )
    label: str = Field(..., description="Arayuzde gorunen Turkce etiket.")
    detail: str | None = Field(None, description="Secenegin altinda gorunen aciklama.")


class ApprovalRequest(BaseModel):
    """A question the loop stops to ask.

    Only three kinds exist on purpose. Asking at every step would drown the
    demo; never asking would drop the one thing KKB called a differentiator.
    """

    request_id: str
    kind: Literal["series_choice", "fetch", "quality"]
    question: str
    options: list[ApprovalOption]
    default: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    step_no: int = 0
    created_at: datetime


class ApprovalDecision(BaseModel):
    request_id: str
    choice: str
    note: str | None = None


class VerificationFlag(BaseModel):
    kind: Literal[
        "number_not_found",
        "source_not_listed",
        "date_out_of_range",
        "unit_mix",
        "model_contradiction",
    ]
    text: str
    detail: str | None = None


class VerificationResult(BaseModel):
    passed: bool = True
    flags: list[VerificationFlag] = Field(default_factory=list)


@dataclass
class SessionState:
    """Mutable state for one chat session, carried across turns.

    The active table living here is what makes "add a column without breaking
    the table" possible: follow-up questions operate on the frame already in
    this store instead of rebuilding it from scratch.
    """

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    active_table: str | None = None
    lineage: dict[str, ColumnInfo] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    charts: list[ChartSpec] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    # --- per-turn bookkeeping --------------------------------------------
    turn_no: int = 0
    step_no: int = 0
    steps_this_turn: int = 0
    #: Seconds actually spent working. Waiting for a human is not spending.
    active_s: float = 0.0
    #: Seconds spent waiting for a person. Subtracted from `active_s`, because
    #: a jury reading an approval card is not the agent being slow.
    approval_wait_s: float = 0.0
    pending_calls: list[ToolCall] = field(default_factory=list)
    #: Keyed by tool name, not by call id: every tool call carries a fresh id,
    #: so a per-id counter would reset on the retry it is meant to count.
    arg_retries: dict[str, int] = field(default_factory=dict)
    call_signatures: dict[str, int] = field(default_factory=dict)
    #: Tool warnings collected across the turn, for the answer prompt. A unit
    #: conversion the answer does not mention is a wrong sentence.
    warnings: list[str] = field(default_factory=list)
    truncated_reason: TruncatedReason | None = None
    draft_answer: str | None = None
    verification: VerificationResult | None = None
    approvals: list[ApprovalDecision] = field(default_factory=list)

    def next_step_no(self) -> int:
        self.step_no += 1
        return self.step_no

    def add_sources(self, sources: list[Source]) -> None:
        """Append, keeping one entry per underlying source.

        The same series fetched in two steps is still one source; a citation
        list that repeats it reads as sloppy to a jury looking for provenance.
        """
        seen = {existing.key() for existing in self.sources}
        for source in sources:
            if source.key() not in seen:
                self.sources.append(source)
                seen.add(source.key())


def initial_state(session_id: str) -> SessionState:
    return SessionState(session_id=session_id)


def slug(text: str) -> str:
    """Lowercase ascii-ish slug, trimmed to fit the table_ref pattern."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", text.casefold()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned) or "table"
    if not cleaned[0].isalpha():
        cleaned = f"t_{cleaned}"
    return cleaned[:24].rstrip("_") or "table"


def new_table_ref(tables: dict[str, Any], suffix: str) -> str:
    """Generate the next table reference: t01_get_series, t02_deflate, ...

    Generated, never taken from the model. The narrow pattern in contracts.py
    is the other half of that rule: an invented name fails validation before it
    can reach a tool.

    Numbered from the highest reference seen, not from how many tables exist.
    A table can be removed mid-session, the quality gate does exactly that when
    a person rejects one, and counting would then reissue a number that is
    still in use and silently overwrite the table behind it.
    """
    used = [
        int(match.group(1))
        for key in tables
        if (match := re.match(r"t(\d+)_", str(key)))
    ]
    ref = f"t{max(used, default=0) + 1:02d}_{slug(suffix)}"
    if not re.fullmatch(TABLE_REF_PATTERN, ref):  # pragma: no cover - defensive
        raise ValueError(f"Uretilen tablo anahtari desene uymuyor: {ref!r}")
    return ref


def new_chart_ref(charts: list[Any]) -> str:
    """Generate the next chart reference: c01, c02, ...

    Numbered from the highest reference seen, for the same reason tables are:
    counting hands out a number that is still in use the moment anything is
    removed from the list. Charts had kept the counting version even after
    tables were fixed, and they get the pattern check too -- without it an
    out-of-range reference is built quietly here and only fails later, inside
    `ChartSpec` validation, where nothing points back to this function.
    """
    refs = [_chart_ref_of(chart) for chart in charts]
    used = [
        int(match.group(1)) for ref in refs if (match := re.fullmatch(r"c(\d+)", ref))
    ]
    ref = f"c{max(used, default=0) + 1:02d}"
    if not re.fullmatch(CHART_REF_PATTERN, ref):  # pragma: no cover - defensive
        raise ValueError(f"Uretilen grafik anahtari desene uymuyor: {ref!r}")
    return ref


def _chart_ref_of(chart: Any) -> str:
    """The reference of a chart, however it is being carried.

    `ChartSpec` calls the field `chart_ref`; `contracts.Artifact` -- the shared
    contract, and what `add_chart` is annotated to take -- calls it `ref`.
    Reading only one of them is worse than counting was: an unrecognised
    carrier contributes no number at all, so the scan sees an empty list and
    hands out `c01` again for every chart in the session.
    """
    if isinstance(chart, dict):
        value = chart.get("chart_ref") or chart.get("ref")
    else:
        value = getattr(chart, "chart_ref", None) or getattr(chart, "ref", None)
    return str(value or "")
