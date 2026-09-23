"""Measure the agent end to end against eval/questions.yaml.

The sibling of `lake/search_eval.py`, one layer up: that one asks whether the
pool search can find a series, this one asks whether the whole loop -- plan,
tools, approvals, answer, verifier -- turns a question into a defensible
answer. Real models and real data, one MIA call per step, so it is a
measurement and not a unit test.

Each question is scored on four things, and all four have to hold:

* **the series it used** (`state.sources`) covers the expected ones. Search
  hits do not count: `search_catalog` attaches no sources, so this is the
  series that actually reached a table or a citation;
* **the tools it called** cover the expected ones. A superset passes -- the
  agent may reasonably search before fetching, or pull a deflator nobody
  listed -- an omission does not;
* **the checks** in the question's own `checks:` block;
* **the turn finished**: a `truncated_reason` fails the row whatever else
  came out, because a truncated answer is not the answer we asked about.

Verifier flags are reported per question but do **not** fail the row. The
verifier is itself under measurement here, and making it a pass condition
would reward quietly loosening it.

Answers vary between runs. Read a red row before believing it: the question's
expectation is as likely to be wrong as the agent.

    uv run python -m lakehouse_agent.agent.answer_eval
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lakehouse_agent.agent.loop import AgentDeps, run_turn
from lakehouse_agent.agent.state import SessionState, initial_state
from lakehouse_agent.llm import MiaLLM

QUESTIONS_PATH = Path("eval/questions.yaml")
RESULTS_ROOT = Path("eval/results")

logger = logging.getLogger(__name__)


def _deps() -> AgentDeps:
    """Real models, real pool, no event stream.

    Deliberately not `loop._real_deps`: that one wires a `JsonlSink` so the
    CLI can stream events to stdout, and a measurement wants the summary at
    the end rather than a hundred thousand lines of JSON. A `None` sink is
    supported on purpose (`agent/events.py`).

    `.env` is loaded here rather than at import: only an entry point may read
    it (`config.py`).
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
        sink=None,
    )


# --- reading the session ---------------------------------------------------


def _used_series(state: SessionState) -> set[str]:
    """Series the answer actually rests on.

    `search_catalog` returns candidates as a preview and attaches no sources,
    so this set is what `get_series`, `transform_table` and the web tools put
    on the record -- not everything the agent looked at.
    """
    return {source.series_id for source in state.sources if source.series_id}


def _called_tools(state: SessionState) -> set[str]:
    """Every tool the turn attempted.

    Filtered on `tool`, not on `node == "act"`: a call cancelled at the fetch
    gate is filed under `node="approval"` and would otherwise vanish from the
    ledger. Filtered by turn because `state.steps` is never cleared.
    """
    return {
        step.tool for step in state.steps if step.turn_no == state.turn_no and step.tool
    }


def _table(state: SessionState, suffix: str) -> pd.DataFrame | None:
    """The table a given tool produced, by the suffix in its reference.

    References are generated as `t01_get_series`, `t02_causality`; the suffix
    is what the tool asked for, which is not always its own name
    (`test_causality` writes `causality` and `lag_corr`).
    """
    for ref in sorted(state.tables):
        if ref.split("_", 1)[-1] == suffix:
            return state.tables[ref]
    return None


# --- checks ----------------------------------------------------------------


def _same(cell: Any, expected: Any, tolerance_pct: float) -> bool:
    """Does one table cell match what the question expects?

    Booleans are tested first, because in Python `isinstance(True, int)`
    holds and a verdict would otherwise be scored as arithmetic. With a
    tolerance that is wrong rather than merely odd: `beklenen: false` with
    `tolerans_yuzde: 1` becomes "within 1 of zero", which `True` satisfies,
    so a question would pass on the opposite verdict.

    Numbers use the percentage tolerance, read as an absolute margin when
    the expected value is zero (a percentage of zero admits nothing).
    Everything else (dates, labels) is compared as text, with timestamps
    narrowed to their date.
    """
    if isinstance(expected, bool):
        return bool(cell) is expected
    if isinstance(expected, int | float):
        if pd.isna(cell):
            return False
        margin = abs(expected) * tolerance_pct / 100 if expected else tolerance_pct
        return abs(float(cell) - float(expected)) <= margin
    text = cell.strftime("%Y-%m-%d") if isinstance(cell, pd.Timestamp) else str(cell)
    return text == str(expected)


def _check_deger(spec: dict, state: SessionState) -> tuple[bool, str]:
    """One named cell of one table equals what was measured by hand."""
    suffix = spec.get("tablo")
    if suffix:
        frame = _table(state, suffix)
        if frame is None:
            return False, f"'{suffix}' tablosu uretilmedi"
    else:
        frame = next(
            (
                state.tables[ref]
                for ref in sorted(state.tables)
                if spec["sutun"] in state.tables[ref]
            ),
            None,
        )
        if frame is None:
            return False, f"'{spec['sutun']}' sutunu hicbir tabloda yok"
    if spec["sutun"] not in frame:
        return False, f"'{spec['sutun']}' sutunu tabloda yok: {list(frame.columns)}"

    if donem := spec.get("donem"):
        if "period" not in frame:
            return False, "tabloda period sutunu yok"
        rows = frame.loc[frame["period"] == pd.Timestamp(f"{donem}-01")]
        if rows.empty:
            return False, f"{donem} donemi tabloda yok"
    else:
        rows = frame

    # Without a period, every row has to match. That is what makes a verdict
    # table checkable without knowing its row order: `test_causality` writes
    # one row per direction, and which one comes first depends on which column
    # the model passed as `column` rather than on anything we decide here.
    if rows.empty:
        return False, "tablo bos"
    tolerance = float(spec.get("tolerans_yuzde", 0))
    cells = list(rows[spec["sutun"]])
    ok = all(_same(cell, spec["beklenen"], tolerance) for cell in cells)
    seen = cells[0] if len(cells) == 1 else cells
    return ok, f"{seen!r} (beklenen {spec['beklenen']!r})"


def _check_donem_yok(spec: dict, state: SessionState) -> tuple[bool, str]:
    """One column must carry no value before a date.

    The machine-readable half of the refusal category: an agent that says
    "the pool does not go back that far" and then tabulates it anyway is
    caught here. One that simply refuses never creates the column and passes.

    Scoped to a column and to non-null values on purpose. A period alone
    proves nothing -- EVDS hands back a cell for every period in the window
    and leaves it empty before the series was issued, so `get_series` over
    2021-2026 legitimately returns empty 2021 rows. And the agent may
    reasonably pull a second series (a deflator) that does have 2021 values.
    """
    column, edge = spec["sutun"], pd.Timestamp(f"{spec['once']}-01")
    for ref in sorted(state.tables):
        frame = state.tables[ref]
        if column not in frame or "period" not in frame:
            continue
        early = frame.loc[(frame["period"] < edge) & frame[column].notna(), "period"]
        if not early.empty:
            return False, f"{ref}.{column} {early.min():%Y-%m} icin deger tasiyor"
    return True, f"{spec['once']} oncesi '{column}' degeri yok"


def _run_check(spec: dict, state: SessionState, answer: str) -> dict:
    """Score one entry of a question's `checks:` list."""
    (kind, value), *_ = spec.items()
    if kind == "cevapta_gecmeli":
        ok = str(value).casefold() in answer.casefold()
        detail = "gecti" if ok else f"'{value}' cevapta yok"
    elif kind == "deger":
        ok, detail = _check_deger(value, state)
    elif kind == "tablo_var":
        ok = _table(state, str(value)) is not None
        detail = (
            "gecti" if ok else f"'{value}' tablosu uretilmedi ({list(state.tables)})"
        )
    elif kind == "kaynak_url_var":
        found = [source.url for source in state.sources if source.url]
        ok = bool(found) is bool(value)
        detail = f"{len(found)} adres kaynagi"
    elif kind == "tabloda_donem_yok":
        ok, detail = _check_donem_yok(value, state)
    else:
        ok, detail = False, f"bilinmeyen kontrol turu: {kind}"
    return {"tip": kind, "gecti": ok, "detay": detail}


# --- one question ----------------------------------------------------------


def _one(entry: dict, deps: AgentDeps) -> dict:
    """Run one question on a session of its own and score what came back.

    A fresh `SessionState` per question, unlike the CLI which shares one so a
    follow-up can build on the previous table. Eval items are independent;
    sharing would let one question's table satisfy the next question's check.
    """
    state = initial_state(f"eval-{entry['id']}")
    error = None
    try:
        result = asyncio.run(run_turn(state, deps, entry["question"]))
        answer = result.answer or ""
        usage, duration = result.usage, result.duration_ms
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        logger.warning("%s dustu: %s", entry["id"], exc)
        error = f"{type(exc).__name__}: {exc}"
        answer, usage, duration = "", None, 0

    missing_series = sorted(
        set(entry.get("expected_series") or []) - _used_series(state)
    )
    missing_tools = sorted(
        set(entry.get("expected_tools") or []) - _called_tools(state)
    )
    checks = [_run_check(spec, state, answer) for spec in entry.get("checks") or []]
    flags = state.verification.flags if state.verification else []

    return {
        "id": entry["id"],
        "soru": entry["question"],
        "kategori": entry.get("category"),
        "gecti": bool(
            error is None
            and not missing_series
            and not missing_tools
            and all(check["gecti"] for check in checks)
            and state.truncated_reason is None
        ),
        "eksik_seri": missing_series,
        "eksik_arac": missing_tools,
        "kontroller": checks,
        "bayrak": len(flags),
        "bayrak_metinleri": [flag.text for flag in flags],
        "dogrulandi": any(
            step.node == "verify"
            for step in state.steps
            if step.turn_no == state.turn_no
        ),
        "kesinti": state.truncated_reason,
        "sure_ms": duration,
        "token": usage.total_tokens if usage else 0,
        "cevap": answer,
        "hata": error,
    }


def run(deps: AgentDeps | None = None, *, path: Path | None = None) -> dict:
    """Score every question in the file and return the report.

    `deps` is the seam the tests drive: hand it a scripted `FakeLLM` and a
    fake registry and the whole runner works offline. Nothing here imports
    the fakes, which is what keeps the import-discipline test quiet.

    Writing is the caller's job -- `__main__` does it -- so a test can score
    a run without leaving a file behind.
    """
    entries = yaml.safe_load((path or QUESTIONS_PATH).read_text(encoding="utf-8"))
    deps = deps or _deps()
    rows = [_one(entry, deps) for entry in entries]
    return {
        "zaman": datetime.now(UTC).isoformat(),
        "model": deps.settings.mia_chat_model,
        "soru_sayisi": len(rows),
        "gecen": f"{sum(1 for row in rows if row['gecti'])}/{len(rows)}",
        "bayrakli": sum(1 for row in rows if row["bayrak"]),
        "kesintili": sum(1 for row in rows if row["kesinti"]),
        "sure_ms": sum(row["sure_ms"] for row in rows),
        "token": sum(row["token"] for row in rows),
        "sorular": rows,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    report = run()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULTS_ROOT / f"answer_eval_{report['zaman'][:10]}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"gecen: {report['gecen']}  bayrakli: {report['bayrakli']}  "
        f"kesintili: {report['kesintili']}  "
        f"({report['sure_ms'] / 1000:.0f} sn, {report['token']} token, "
        f"{report['model']})"
    )
    for row in report["sorular"]:
        mark = "gecti " if row["gecti"] else "DUSTU "
        print(f" {mark} {row['id']:>3}  {row['soru']}")
        for reason, detail in (
            ("eksik seri", ", ".join(row["eksik_seri"])),
            ("eksik arac", ", ".join(row["eksik_arac"])),
            ("kesinti", row["kesinti"] or ""),
            ("hata", row["hata"] or ""),
        ):
            if detail:
                print(f"          {reason}: {detail}")
        for check in row["kontroller"]:
            if not check["gecti"]:
                print(f"          {check['tip']}: {check['detay']}")
        if row["bayrak"]:
            print(f"          bayrak: {'; '.join(row['bayrak_metinleri'])}")
    print(f"yazildi: {out}")
