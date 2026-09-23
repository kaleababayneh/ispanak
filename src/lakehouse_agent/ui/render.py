"""Event payload -> content, for the Chainlit adapter to draw.

Pure functions only: no `chainlit` import, no I/O, no settings reads
(`tests/test_render.py` covers this file without a browser or a network).
Every function treats a missing or malformed field as empty rather than
raising -- an event list that grows must not crash an interface that has not
caught up with it yet (the same rule `agent/console.py` follows for the event
types it does not draw).
"""

from __future__ import annotations

from typing import Any

#: What Enter-key labels the plan/observe steps get, since `step_start`
#: carries no tool name for them (`docs/contracts.md` section 6).
_NODE_LABELS = {"plan": "Planlama", "observe": "Degerlendirme"}

#: Short, non-technical text for a known error kind. Anything else -- and the
#: raw `message` -- stays server-side; a person must not read a stack trace.
_ERROR_MESSAGES = {
    "LlmError": "Model ile iletisim kurulamadi. Lutfen tekrar deneyin.",
    "LlmTimeoutError": "Model yaniti zaman asimina ugradi. Lutfen tekrar deneyin.",
    "CancelledError": "Tur durduruldu.",
}


def render_step_start(payload: dict[str, Any] | None) -> dict[str, str]:
    """A step card's opening line: what is about to run."""
    payload = payload or {}
    tool = payload.get("tool")
    if tool:
        args = ", ".join(f"{k}={v}" for k, v in (payload.get("args") or {}).items())
        return {"label": str(tool), "input_text": args}
    node = str(payload.get("node") or "")
    return {"label": _NODE_LABELS.get(node, node or "Adim"), "input_text": ""}


def render_step_end(payload: dict[str, Any] | None) -> dict[str, Any]:
    """A step card's closing line: what happened."""
    payload = payload or {}
    ok = payload.get("ok") is not False
    summary = str(payload.get("summary") or "").splitlines()[:1]
    return {
        "output_text": summary[0] if summary else "",
        "duration_ms": int(payload.get("duration_ms") or 0),
        "is_error": not ok,
        "error_text": str(payload.get("error") or "") if not ok else "",
    }


def render_table_note(payload: dict[str, Any] | None) -> str:
    """One line describing a table, drawn once alongside it.

    Takes the same shape `table_update` carries -- `table_ref` and
    `new_columns` -- but the caller need not call it per event: a turn's
    columns are accumulated across several `table_update`s and this is
    called once, when the table itself is drawn (`docs/decisions.md`, "Ekran
    tabloyu tam gosterir").
    """
    payload = payload or {}
    ref = str(payload.get("table_ref") or "")
    if payload.get("removed"):
        reason = payload.get("reason")
        return f"tablo {ref} kaldirildi" + (f" ({reason})" if reason else "")
    fresh = [str(name) for name in payload.get("new_columns") or []]
    if fresh:
        return f"tablo {ref}: {', '.join(fresh)} eklendi"
    return f"tablo {ref} kuruldu"


def render_sources(payload: dict[str, Any] | None) -> str | None:
    """The sources block under an answer, or `None` when there is nothing to cite."""
    payload = payload or {}
    sources = payload.get("sources") or []
    if not sources:
        return None
    lines = ["**Kaynaklar**"]
    for source in sources:
        if not isinstance(source, dict):
            continue
        ident = source.get("series_id") or source.get("url") or ""
        bits = [f"- `{ident}`"] if ident else ["-"]
        if source.get("name"):
            bits.append(str(source["name"]))
        if source.get("fetched_at"):
            bits.append(f"(cekim: {source['fetched_at']})")
        line = " ".join(bits)
        if source.get("metadata_url"):
            line += f" — [metaveri]({source['metadata_url']})"
        if source.get("note"):
            line += f" — {source['note']}"
        lines.append(line)
    return "\n".join(lines)


def render_verification(payload: dict[str, Any] | None) -> str | None:
    """The verification warning block, or `None` when nothing was flagged.

    Shown only when there is a flag: a clean verification is not worth a
    person's attention, same rule as `agent/console.py`'s `_draw_verification`.
    """
    payload = payload or {}
    flags = payload.get("flags") or []
    if not flags:
        return None
    lines = ["**Dogrulanamayan ifadeler**"]
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        text = str(flag.get("text") or "")
        line = f"- {text}"
        if flag.get("detail"):
            line += f" ({flag['detail']})"
        lines.append(line)
    return "\n".join(lines)


def render_chart(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """A chart the model asked for, or `None` when the payload is not a
    Plotly figure the adapter can draw (`chart` -> `cl.Plotly`, else skipped).
    """
    payload = payload or {}
    plotly_json = payload.get("plotly_json")
    if not isinstance(plotly_json, dict) or not isinstance(
        plotly_json.get("data"), list
    ):
        return None
    if not plotly_json["data"]:
        return None
    return {
        "chart_ref": str(payload.get("chart_ref") or ""),
        "title": payload.get("title"),
        "plotly_json": plotly_json,
    }


def render_approval_card(payload: dict[str, Any] | None) -> dict[str, Any]:
    """An `ApprovalRequest`, as something `cl.AskActionMessage` can be built from."""
    payload = payload or {}
    default = payload.get("default")
    options = [
        {
            "id": str(option.get("id") or ""),
            "label": str(option.get("label") or option.get("id") or ""),
            "detail": option.get("detail"),
            "is_default": option.get("id") == default,
        }
        for option in payload.get("options") or []
        if isinstance(option, dict)
    ]
    return {
        "question": str(payload.get("question") or ""),
        "kind": payload.get("kind"),
        "options": options,
        "default": default,
    }


def render_approval_decision(payload: dict[str, Any] | None) -> str:
    """One line confirming how an approval card was settled."""
    payload = payload or {}
    how = "otomatik" if payload.get("automatic") else "secildi"
    return f"Onay [{payload.get('kind')}]: {payload.get('choice')} ({how})"


def render_error(payload: dict[str, Any] | None) -> dict[str, str]:
    """A short Turkish message for the person, and the raw detail for the
    server log -- never the other way around."""
    payload = payload or {}
    kind = str(payload.get("kind") or "")
    user_message = _ERROR_MESSAGES.get(kind, "Bir hata olustu. Lutfen tekrar deneyin.")
    log_message = f"{kind}: {payload.get('message') or ''}".strip(": ")
    return {"user_message": user_message, "log_message": log_message}
