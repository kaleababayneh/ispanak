"""The browser interface, minus the browser.

`app.py` at the repository root is the Chainlit shell. It cannot be imported
by a test -- pytest does not put the root on `sys.path` -- so nothing that
makes a decision lives there. Everything that does lives here, imports neither
`chainlit` nor `plotly`, and is tested like any other module.

The sibling of `agent/console.py`: one draws the terminal, this one prepares
what the browser draws. Both read the event contract in `docs/contracts.md`
section 6 and neither is known to the loop.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from lakehouse_agent.agent.labels import column_label, column_units, table_labels
from lakehouse_agent.agent.state import (
    ApprovalDecision,
    ApprovalRequest,
    ChartSeries,
    ChartSpec,
    new_chart_ref,
)
from lakehouse_agent.tools.charts import figure_json, without_totals

#: How many sources are listed before the rest are counted. Same as the
#: terminal's, so the two interfaces do not disagree about what is shown.
SOURCES_SHOWN = 12

#: Columns that are an axis, not a measurement. A default chart draws
#: everything else.
AXIS_COLUMNS = ("period", "obs_date", "date", "tarih")

#: Decimals shown in the browser table. The session's frame is not rounded;
#: the eval reads it and a ratio of 19.998 must stay 19.998. Four is what
#: the terminal shows too (`console.DECIMALS`), so the two screens agree.
DECIMALS = 4

#: A series id, `evds:TP.KTF12` or `bddk:aylik:tuketici:konut:TL:sektor`.
#: Fenced in backticks before it reaches Chainlit's markdown: the shipped
#: frontend bundles `micromark-extension-directive` (its `directiveContainer*`
#: tokenizer states are in `frontend/dist/assets/index-*.js`, checked 23
#: Sep), which reads a bare `:name` outside a code span as a directive. So
#: `evds:TP.KTF12` rendered as "evds", a break, ".KTF12". The lookaround
#: leaves an id a caller already fenced alone. Found and fixed on Burak's
#: `feat/chainlit-ui` (`ui/render.py`); carried over here unchanged in effect.
_SEGMENT = r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
_SERIES_ID = re.compile(rf"(?<!`)\b(?:evds|bddk):{_SEGMENT}(?::{_SEGMENT})*(?!`)")

#: Short, non-technical text for a known error kind. The raw message goes to
#: the server log; a person must not be shown a stack trace. Keyed by the
#: `kind` the `error` event carries (`loop.py`: `type(exc).__name__`).
ERROR_MESSAGES = {
    "LlmError": "Model ile iletişim kurulamadı. Lütfen tekrar deneyin.",
    "LlmTimeoutError": "Model yanıtı zaman aşımına uğradı. Lütfen tekrar deneyin.",
    "CancelledError": "Tur durduruldu.",
}
GENERIC_ERROR = "Bir hata oluştu. Lütfen tekrar deneyin."

#: What each kind of table is, by the suffix `new_table_ref` gave it
#: (`tools/*.py`). A reader met `t02_detect_changes` with no idea whose
#: break it held (seen 23 Sep); the tool that built it and the column it
#: analysed are both in the lineage, so the title is derived, not typed.
TOOL_TITLES = {
    "get_series": "Seri tablosu",
    "detect_changes": "Kırılma analizi",
    "detect_anomalies": "Aykırı dönem analizi",
    "causality": "Granger öncüllük testi",
    "lag_corr": "Gecikmeli korelasyon",
    "read_url": "Adresten okunan tablo",
}

#: Most series a default chart draws. Beyond this the picture is a hairball
#: and the model should have asked for the columns it meant.
DEFAULT_SERIES_MAX = 4


def action_specs(request: ApprovalRequest) -> list[dict[str, Any]]:
    """The approval card's options as Chainlit action arguments.

    Plain dicts, so `app.py` only has to call `cl.Action(**spec)`. The choice
    travels in `payload["value"]`, which is where `cl.AskActionMessage` puts
    it in the response.
    """
    specs = []
    for option in request.options:
        mark = " (varsayilan)" if option.id == request.default else ""
        label = f"{option.label}{mark}"
        specs.append(
            {
                "name": option.id,
                "payload": {"value": option.id},
                "label": label,
                "tooltip": option.detail or label,
            }
        )
    return specs


def decision(request: ApprovalRequest, value: Any) -> ApprovalDecision:
    """Turn what the card came back with into a decision.

    Anything unrecognised -- a timeout, a closed tab, an option that is not on
    this card -- falls to the default and is marked `automatic`. The flag is
    not cosmetic: `approvals.decision_message` tells the model a different
    thing when a person actually chose, so passing a default off as a choice
    would put a sentence in the answer that nobody said.
    """
    chosen = str(value) if value is not None else ""
    if chosen not in {option.id for option in request.options}:
        return ApprovalDecision(
            request_id=request.request_id,
            choice=request.default or "",
            note="karta cevap gelmedi, varsayilan uygulandi",
            automatic=True,
        )
    return ApprovalDecision(
        request_id=request.request_id,
        choice=chosen,
        note="arayuzden secildi",
    )


def protect_series_ids(text: str) -> str:
    """Fence every bare series id in `text` with backticks (see `_SERIES_ID`).

    For prose the model or a person wrote. `source_lines` fences its own ids
    by hand and the lookaround leaves those alone either way.
    """
    if not text:
        return text
    return _SERIES_ID.sub(lambda m: f"`{m.group(0)}`", text)


def error_message(payload: Mapping[str, Any] | None) -> tuple[str, str]:
    """`(what the person reads, what the log gets)` for an `error` event."""
    payload = payload or {}
    kind = str(payload.get("kind") or "")
    user = ERROR_MESSAGES.get(kind, GENERIC_ERROR)
    log = f"{kind}: {payload.get('message') or ''}".strip(": ")
    return user, log


def removed_note(payload: Mapping[str, Any] | None) -> str | None:
    """One line when the quality gate took a table away, else `None`.

    The loop fires `table_update{removed, reason}` when a person rejects a
    low-quality table (`loop.py`, `quality_cancel`). Silently dropping that
    event left the interface showing a table the session no longer has.
    """
    payload = payload or {}
    if not payload.get("removed"):
        return None
    ref = str(payload.get("table_ref") or "")
    reason = payload.get("reason")
    return f"Tablo {ref} kaldırıldı" + (f" ({reason})" if reason else "") + "."


def note_table(known: set[str], ref: str, fresh: Iterable[str]) -> tuple[bool, bool]:
    """Whether a `table_update` is worth drawing, and whether the table is new.

    Returns `(is_new, changed)` and records the table in `known`. A turn
    that only read a table -- "the three highest months" is a `top`, which
    adds no column -- fires an update with no new columns; drawing sixty
    rows again under that answer was noise (seen 23 Sep). A table is drawn
    the first time it is seen and whenever a column is added, and not
    otherwise.
    """
    is_new = ref not in known
    known.add(ref)
    return is_new, is_new or bool(list(fresh))


def fallback_table(
    active_table: str | None, new_tables: Iterable[str], charts_added: int
) -> str | None:
    """Which table, if any, the interface should chart on the model's behalf.

    Only a table this turn **created**, and only when the turn drew nothing
    itself. The fallback exists for one case: the model spent its steps on
    the analysis and ended table-only. It is not for follow-ups. The active
    table survives across turns, so a question that produced no table (a
    page read, a refusal) would otherwise get the previous question's chart
    under its answer -- seen 23 Sep, a TCMB summary with housing-sales
    lines beneath it -- and a follow-up that only read the table ("the
    three highest months") got the same chart a second time.
    """
    if charts_added or not active_table:
        return None
    return active_table if active_table in set(new_tables) else None


def default_chart(
    frame: pd.DataFrame,
    table_ref: str,
    charts: list[Any],
    title: str | None = None,
    lineage: Mapping[str, Any] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> ChartSpec | None:
    """A line chart of whatever the table holds, or `None` if it holds nothing.

    A fallback, not a feature. `make_chart` is a step and a turn has twelve
    (`loop.AgentConfig.max_steps`), so the model will sometimes spend them all
    on the analysis and never draw. Rather than show a table-only answer, the
    interface draws the obvious chart itself. It is built here so the choice
    of columns is testable, and it is marked in its title so nobody reads it
    as something the agent asked for.
    """
    x = next((name for name in AXIS_COLUMNS if name in frame.columns), None)
    if x is None or len(frame) < 2:
        # One row is a number, not a line. Seen 23 Sep: a single-month fetch
        # drew an empty chart with a time axis of milliseconds around it.
        return None
    numeric = [
        str(name)
        for name in frame.columns
        if str(name) != x and pd.api.types.is_numeric_dtype(frame[name])
    ]
    numeric = [name for name in numeric if frame[name].notna().any()]
    if not numeric:
        return None

    label = title or f"{table_ref} (arayuz cizdi)"
    frame, _ = without_totals(frame, x)
    series = _by_unit(numeric[:DEFAULT_SERIES_MAX], table_ref, lineage or {})
    labels = table_labels(table_ref, frame.columns, lineage or {}, overrides)
    units = column_units(table_ref, frame.columns, lineage or {})
    return ChartSpec(
        chart_ref=new_chart_ref(charts),
        table_ref=table_ref,
        x=x,
        series=series,
        title=label,
        plotly_json=figure_json(frame, x, series, label, labels, units),
    )


def chart_labels(charts: Iterable[Any], table_ref: str) -> dict[str, str]:
    """The names the model gave this table's columns when it drew them.

    Read off the session's charts so the table header says what the legend
    says: a reader who has just seen "Konut kredisi faiz oranı" on the
    chart should not meet "Konut Kredisi (TL, Akım, %)" in the table below
    it. The latest chart of the table wins; a series left unlabelled
    contributes nothing.
    """
    found: dict[str, str] = {}
    for chart in charts:
        if str(_attr(chart, "table_ref") or "") != table_ref:
            continue
        for one in _attr(chart, "series") or []:
            label = str(_attr(one, "label") or "").strip()
            if label:
                found[str(_attr(one, "column"))] = label
    return found


def _attr(thing: Any, name: str) -> Any:
    if thing is None:
        return None
    return thing.get(name) if isinstance(thing, Mapping) else getattr(thing, name, None)


def _by_unit(
    columns: list[str], table_ref: str, lineage: Mapping[str, Any]
) -> list[ChartSeries]:
    """One axis per unit: the first unit seen goes left, the second right.

    A share in percent drawn on an axis of counts is a flat line at zero,
    which is what the first fallback chart did. Columns of a third unit are
    left out rather than drawn on the wrong scale; a chart that says less
    beats one that misleads. Unit comes from lineage; a column with none
    forms its own group.
    """
    axes: dict[str, str] = {}
    series: list[ChartSeries] = []
    for name in columns:
        info = lineage.get(f"{table_ref}.{name}")
        unit = str(getattr(info, "unit", None) or "").strip().casefold()
        if unit not in axes:
            if len(axes) == 2:
                continue
            axes[unit] = "left" if not axes else "right"
        series.append(ChartSeries(column=name, axis=axes[unit]))  # type: ignore[arg-type]
    return series


def display_frame(
    frame: pd.DataFrame,
    table_ref: str,
    lineage: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """The table as a person reads it: labelled headers, same values.

    A renamed copy, never the session's frame. The session keeps the names
    the model works with; only what is drawn changes. Two columns that
    label the same (a fixture, or a catalog collision) are left under their
    own names rather than collapsed into one header. `overrides` are the
    model's own words for some columns (`chart_labels`); they win over the
    catalog's.
    """
    labels = table_labels(table_ref, frame.columns, lineage, overrides)
    seen = list(labels.values())
    unique = {name: label for name, label in labels.items() if seen.count(label) == 1}
    shown = frame.rename(columns=unique)
    for name in shown.columns:
        if pd.api.types.is_datetime64_any_dtype(shown[name]):
            # `period` is a date by contract (the first day, no time), but
            # the frame holds it as datetime64 and the browser table then
            # printed `2021-01-01T00:00:00.000`. The time was never data.
            shown[name] = shown[name].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(shown[name]):
            # 1716481.7634920634 is not more true than 1716481.7635; it is
            # just harder to read. Display only; the frame keeps every digit.
            shown[name] = shown[name].round(DECIMALS)
    return shown


def step_line(payload: dict[str, Any], tool: str | None = None) -> str:
    """The one-line label of a finished step.

    `step_end` carries no tool name (`docs/contracts.md` section 6), so the
    caller passes the one it saw at `step_start`; without it the node
    ("act") is all there is, and a screen of "act · 1707 ms" lines told a
    person nothing about which tool ran (seen 23 Sep).
    """
    ok = payload.get("ok") is not False
    mark = "✓" if ok else "✗"
    who = tool or str(payload.get("node") or "")
    return f"{mark} {who} · {payload.get('duration_ms', 0)} ms"


def table_title(
    ref: str,
    columns: Iterable[Any],
    lineage: Mapping[str, Any],
    charts: Iterable[Any] = (),
) -> str:
    """What a table is, in words: the analysis and the series it was run on.

    `t02_detect_changes` → "Kırılma analizi — Konut kredisi faiz oranı".
    The subject is the analysed column named in the result columns' lineage
    (`inputs`), labelled the way the rest of the screen labels it: the
    model's chart label if it gave one anywhere in the session, else the
    catalog name from whichever table fetched it. A series table is titled
    by its own columns. Falls back to the suffix when the lineage is silent.
    """
    suffix = ref.split("_", 1)[1] if "_" in ref else ref
    kind = TOOL_TITLES.get(suffix, suffix.replace("_", " "))
    names = [
        str(name) for name in columns if str(name) not in ("period", "tarih", "dönem")
    ]
    entries = [lineage.get(f"{ref}.{name}") for name in names]
    entries = [entry for entry in entries if entry is not None]

    if suffix == "get_series":
        # The series that were fetched, not the columns derived from them
        # since: a title listing nominal, real, rebased and percent change
        # of one balance ran to three lines (seen 23 Sep). A fetched column
        # is the one whose lineage carries the catalog's name.
        fetched = [
            name for name in names if _attr(lineage.get(f"{ref}.{name}"), "label_tr")
        ] or names
        subject = ", ".join(
            _label_anywhere(name, lineage, charts) for name in fetched[:3]
        )
        if len(fetched) > 3:
            subject += f" … (+{len(fetched) - 3})"
    elif suffix == "read_url":
        urls = {str(_attr(e, "source_url") or "") for e in entries} - {""}
        subject = _short_url(next(iter(urls), ""))
    else:
        inputs: list[str] = []
        for entry in entries:
            for name in _attr(entry, "inputs") or []:
                if str(name) not in inputs:
                    inputs.append(str(name))
        subject = " ve ".join(_label_anywhere(name, lineage, charts) for name in inputs)
    return f"{kind} — {subject}" if subject else kind


def _short_url(url: str) -> str:
    """An address as a title: the file's name when it has one, else the host
    and path. `kmp_au.pdf` says more in a heading than sixty characters of
    path do."""
    from urllib.parse import urlparse

    if not url:
        return ""
    parsed = urlparse(url)
    last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if "." in last:
        return last
    text = f"{parsed.netloc}{parsed.path}".rstrip("/")
    return text if len(text) <= 60 else text[:57] + "…"


def _label_anywhere(
    column: str, lineage: Mapping[str, Any], charts: Iterable[Any]
) -> str:
    """A column's label regardless of which table it sits in.

    The analysed column lives in the *source* table, so its lineage key is
    not under the result table's ref: search every entry by name. The
    model's own word for it, from any chart in the session, wins.
    """
    for chart in charts:
        for one in _attr(chart, "series") or []:
            if (
                str(_attr(one, "column")) == column
                and str(_attr(one, "label") or "").strip()
            ):
                return str(_attr(one, "label")).strip()
    for key in lineage:
        table_ref, _, name = str(key).rpartition(".")
        if name == column and table_ref:
            # No unit: a title is a name; the unit is on the column itself.
            return column_label(table_ref, column, lineage, with_unit=False)
    return column


def flag_lines(payload: dict[str, Any]) -> list[str]:
    """One bullet per statement the verifier could not confirm."""
    return [f"- {flag.get('text', '')}" for flag in (payload.get("flags") or [])]


def merge_sources(payloads: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every source the turn cited, once each, in the order first seen.

    The loop publishes `sources` per tool step, before the answer streams
    (`loop.py`, in the step merge), so a turn of three fetches used to put
    three "Kaynaklar" blocks above the answer. They are gathered here and
    drawn once, at the end. Keyed on `series_id`, else `url`.
    """
    seen: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for source in (payload or {}).get("sources") or []:
            if not isinstance(source, Mapping):
                continue
            key = str(source.get("series_id") or source.get("url") or "")
            if key and key not in seen:
                seen[key] = dict(source)
    return list(seen.values())


def source_lines(payload: dict[str, Any]) -> list[str]:
    """The sources list, capped, with the remainder counted rather than cut."""
    sources = payload.get("sources") or []
    lines = []
    for source in sources[:SOURCES_SHOWN]:
        who = source.get("series_id") or source.get("url") or ""
        name = source.get("name") or ""
        lines.append(f"- `{who}` {name}".rstrip())
    more = len(sources) - SOURCES_SHOWN
    if more > 0:
        lines.append(f"- … ve {more} kaynak daha")
    return lines
