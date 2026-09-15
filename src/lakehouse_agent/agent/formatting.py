"""Turning results and state into the short text the model actually reads.

The rule everything here serves: the model never sees a table, it sees a
description of one. Two reasons. Context stays small enough that the model
keeps paying attention to the question, and a model that never holds the digits
cannot copy them into its answer with one slipped decimal place.
"""

from __future__ import annotations

import pandas as pd

from lakehouse_agent.contracts import ToolResult

#: Tool messages are capped so one chatty tool cannot crowd out the
#: conversation. Long output is a sign the tool should summarise better.
TOOL_MESSAGE_MAX_CHARS = 2000

PREVIEW_ROWS = 5


def describe_table(ref: str, frame: pd.DataFrame) -> str:
    """One line: what this table is, in the terms the model reasons about."""
    columns = [str(column) for column in frame.columns]
    period_note = ""
    if "period" in frame.columns and len(frame):
        try:
            periods = pd.to_datetime(frame["period"])
            period_note = (
                f", {periods.min().strftime('%Y-%m')}"
                f" - {periods.max().strftime('%Y-%m')}"
            )
        except (ValueError, TypeError):
            # A period column we cannot read is a data problem, not a reason to
            # lose a step that already produced a table.
            period_note = ", donem araligi okunamadi"
    return f"{ref}: {len(frame)} satir{period_note}, sutunlar: {', '.join(columns)}"


def build_tables_summary(tables: dict[str, pd.DataFrame], active: str | None) -> str:
    """What tables exist right now, for the planner prompt.

    Without this the model has no way to know a follow-up question can build on
    the table it made two turns ago, and it starts over from the catalog.
    """
    if not tables:
        return "Su an oturumda tablo yok."
    lines = []
    for ref, frame in tables.items():
        marker = "  (aktif)" if ref == active else ""
        lines.append(f"- {describe_table(ref, frame)}{marker}")
    return "\n".join(lines)


def preview_rows(frame: pd.DataFrame, rows: int = PREVIEW_ROWS) -> list[dict]:
    """First and last few rows, JSON-safe.

    Both ends, not just the head: the model is usually asked about a range and
    the last period is as informative as the first.
    """
    if len(frame) <= rows * 2:
        chosen = frame
    else:
        chosen = pd.concat([frame.head(rows), frame.tail(rows)])
    safe = chosen.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%d")
    return safe.to_dict(orient="records")


def format_tool_message(result: ToolResult, table_note: str = "") -> str:
    """The text handed back to the model after a tool ran.

    Deliberately short and structured: what happened, what the table looks
    like, what to watch out for. Warnings are included because a unit
    conversion the model does not know about becomes a wrong sentence in the
    answer.
    """
    parts: list[str] = []
    if not result.ok:
        parts.append(f"[HATA] {result.summary}")
    else:
        parts.append(result.summary)

    if table_note:
        parts.append(table_note)

    if result.warnings:
        parts.append("Uyarilar: " + " | ".join(result.warnings))

    if result.quality is not None:
        quality = result.quality
        bits = [f"{quality.rows} satir"]
        if quality.missing_periods:
            bits.append(f"{quality.missing_periods} eksik donem")
        if quality.duplicates:
            bits.append(f"{quality.duplicates} tekrarli donem")
        if quality.unit_guess is None:
            bits.append("birim bilinmiyor")
        parts.append("Kalite: " + ", ".join(bits))

    if result.preview:
        parts.append(f"Ornek satirlar: {result.preview}")

    text = "\n".join(parts)
    if len(text) > TOOL_MESSAGE_MAX_CHARS:
        text = text[: TOOL_MESSAGE_MAX_CHARS - 20] + "\n... (kisaltildi)"
    return text


def format_sources(sources: list) -> str:
    """Source ids for the answer prompt, so citations are copied not invented."""
    if not sources:
        return "Kaynak yok."
    return "\n".join(
        f"- {source.key()}"
        + (f" ({source.name})" if getattr(source, "name", None) else "")
        for source in sources
    )
