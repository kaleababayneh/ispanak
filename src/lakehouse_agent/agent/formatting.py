"""Turning results and state into the short text the model actually reads.

The rule everything here serves: the model never sees a table, it sees a
description of one. Two reasons. Context stays small enough that the model
keeps paying attention to the question, and a model that never holds the digits
cannot copy them into its answer with one slipped decimal place.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import numbers
from decimal import Decimal
from typing import Any

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
        column = frame["period"]
        if _holds_numbers(column):
            # `to_datetime` reads a plain integer as nanoseconds since the
            # epoch, so a column of 1, 2, 3 would be described as a range in
            # January 1970. Saying nothing beats saying that confidently.
            period_note = ", donem araligi okunamadi"
        else:
            try:
                periods = pd.to_datetime(column)
                period_note = (
                    f", {periods.min().strftime('%Y-%m')}"
                    f" - {periods.max().strftime('%Y-%m')}"
                )
            except (ValueError, TypeError):
                # A period column we cannot read is a data problem, not a
                # reason to lose a step that already produced a table.
                period_note = ", donem araligi okunamadi"
    return f"{ref}: {len(frame)} satir{period_note}, sutunlar: {', '.join(columns)}"


def _holds_numbers(column: pd.Series) -> bool:
    """Whether a column is plain numbers rather than anything date-shaped.

    Asked of the values, not of the dtype, for the same reason `json_safe`
    converts per value: an object column of integers has a dtype that says
    nothing, and it is exactly the column `to_datetime` would misread. Cheap in
    the ordinary case -- the first date ends the scan.
    """
    if pd.api.types.is_numeric_dtype(column):
        return True
    # Any, not all. One integer among the dates is enough for `to_datetime` to
    # place that row in 1970 and drag the reported range with it.
    return any(
        isinstance(value, numbers.Number) and not isinstance(value, bool)
        for value in column.dropna()
    )


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
    # Built by position, not through `to_dict`. `to_dict` collapses duplicate
    # column labels down to the last one and reports it as a warning on stderr,
    # which nothing on the model or event path ever reads -- silent data loss
    # in the one function whose job is to show what the table holds.
    labels = [str(column) for column in chosen.columns]
    records = []
    for row in chosen.itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for label, cell in zip(labels, row, strict=True):
            key, suffix = label, 2
            while key in record:
                key, suffix = f"{label}__{suffix}", suffix + 1
            record[key] = json_safe(cell)
        records.append(record)
    return records


def json_safe(value: Any) -> Any:
    """One cell, as something `json.dumps` will accept.

    Converting per value rather than per dtype, because dtype is not what
    decides. Gold stores `period` as `datetime.date` in an object column: no
    dtype check sees it, and it reaches `json.dumps` as the unserialisable
    thing it is. The failure would land on the event channel in the middle of a
    turn, far from here and after the work was already done.

    `NaN` is converted too. `json.dumps` writes it as a bare `NaN`, which is
    not JSON and which the interface's `JSON.parse` rejects outright. Missing
    stays missing -- it becomes `null`, it is not filled in.
    """
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, bool):  # before int: bool is a subclass of it
        return value
    if isinstance(value, dt.datetime):  # before date: datetime is a subclass
        stamp = pd.Timestamp(value)
        if stamp.time() == dt.time(0, 0) and stamp.tz is None:
            return stamp.strftime("%Y-%m-%d")
        return stamp.isoformat()
    if isinstance(value, dt.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, Decimal):
        # `is_finite` and not `is_nan`: `Decimal("Infinity")` would otherwise
        # reach json.dumps and be written as a bare `Infinity`, which has the
        # same problem NaN does.
        return float(value) if value.is_finite() else None
    if isinstance(value, float):
        # `inf` has the same problem as `NaN`: json.dumps emits `Infinity`.
        return value if math.isfinite(value) else None
    if isinstance(value, int | str):
        return value
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if kind in ("M", "m"):
        # `np.datetime64` at nanosecond resolution has no lossless Python
        # equivalent, so `.item()` below would hand back a bare 19-digit
        # integer and the model would read a timestamp as a number. Day
        # resolution converts cleanly, which is why this stayed invisible.
        return json_safe(pd.Timestamp(value) if kind == "M" else pd.Timedelta(value))
    if hasattr(value, "item"):  # numpy and pyarrow scalars
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            # Reached by anything with an `item` that is not a 0-d scalar --
            # an ndarray in a cell, which a groupby/apply result produces.
            return str(value)
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


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
        # Rendered as JSON, not as a Python repr. This is the only place the
        # model reads preview rows, and it is reached by every tool -- not just
        # the ones that built their preview with `preview_rows`. Left as a repr
        # the model is handed `datetime.date(2020, 9, 1)` and `nan` where it
        # should see a date and a null.
        rows = [
            {str(key): json_safe(cell) for key, cell in row.items()}
            for row in result.preview
        ]
        spent = len("\n".join(parts)) + 1
        parts.append(_preview_line(rows, TOOL_MESSAGE_MAX_CHARS - spent))

    text = "\n".join(parts)
    if len(text) > TOOL_MESSAGE_MAX_CHARS:
        text = text[: TOOL_MESSAGE_MAX_CHARS - 20] + "\n... (kisaltildi)"
    return text


def _preview_line(rows: list[dict[str, Any]], budget: int) -> str:
    """The preview, fitted to what is left of the message.

    Dropped from the middle, never from the end. The preview is built last, so
    slicing the finished message would always delete the newest rows first --
    and `preview_rows` keeps both ends precisely because the last period is as
    informative as the first. Losing it silently, under a `(kisaltildi)` that
    does not say what went, is the worst of the three options.
    """
    for keep in range(len(rows), 0, -1):
        if keep == len(rows):
            chosen, label = rows, "Ornek satirlar"
        else:
            head = (keep + 1) // 2
            chosen = rows[:head] + rows[len(rows) - (keep - head) :][: keep - head]
            label = f"Ornek satirlar (ortadan {len(rows) - keep} satir atlandi)"
        line = f"{label}: {json.dumps(chosen, ensure_ascii=False)}"
        if len(line) <= budget:
            return line
    return (
        f"Ornek satirlar mesaja sigmadi: {len(rows)} satir, "
        f"sutunlar: {', '.join(rows[0])}"
    )


def format_sources(sources: list) -> str:
    """Source ids for the answer prompt, so citations are copied not invented."""
    if not sources:
        return "Kaynak yok."
    return "\n".join(
        f"- {source.key()}"
        + (f" ({source.name})" if getattr(source, "name", None) else "")
        for source in sources
    )
