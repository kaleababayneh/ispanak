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
    ledger, and a person cancelling a fetch is not the agent failing to try.
    Filtered by turn because `state.steps` is never cleared.

    A name the registry does not know is the one thing that does not count.
    The loop records the attempt with the name the model invented, so
    `expected_tools` was satisfied by a hallucination -- a question could go
    green on a tool that does not exist.
    """
    return {
        step.tool
        for step in state.steps
        if step.turn_no == state.turn_no and step.tool and step.error != "unknown_tool"
    }


#: Characters of a tool argument kept in the report. Enough for an address,
#: short of pasting a whole table reference list into the JSON.
ARG_CHARS = 200


def _steps(state: SessionState) -> list[dict[str, Any]]:
    """What the turn actually called, in order, with what.

    Recorded because a red row is not self-explaining. Measured 23 Sep: a
    run's answer said a page had returned 404 while the fetch cache held a
    200 for it, and there was no way to tell which address the tool had been
    given -- the report knew a tool had been called, not with what. Reading
    a measurement means reading the steps.
    """
    return [
        {
            "arac": step.tool,
            "ok": step.ok,
            "hata": step.error,
            "args": {
                name: str(value)[:ARG_CHARS]
                for name, value in (step.args or {}).items()
            },
        }
        for step in state.steps
        if step.turn_no == state.turn_no and step.tool
    ]


def _table(state: SessionState, suffix: str) -> pd.DataFrame | None:
    """The table a given tool produced, by the suffix in its reference.

    References are generated as `t01_get_series`, `t02_causality`; the suffix
    is what the tool asked for, which is not always its own name
    (`test_causality` writes `causality` and `lag_corr`).

    The NEWEST match wins. References are numbered in the order they were
    made, and a tool can run twice: `get_series` with `into: "new"` mints a
    fresh reference every time, and refetching is ordinary behaviour. Taking
    the earliest scored A5 red against the very table the question is about
    -- the agent pulled the weekly rate into `t01`, converted to a monthly
    mean in `t02`, and the check read the Monday value out of `t01`.
    """
    for ref in sorted(state.tables, reverse=True):
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
    # Missing is not a verdict. `bool(None)` is False, so a table of empty
    # cells used to satisfy `beklenen: false` -- exactly the "the test never
    # ran" case this check exists to catch -- and `bool(pd.NA)` raises, which
    # took the whole run down with it.
    try:
        if cell is None or bool(pd.isna(cell)):
            return False
    except (TypeError, ValueError):
        return False  # a non-scalar cell is not a value either
    if isinstance(expected, bool):
        return bool(cell) is expected
    if isinstance(expected, int | float):
        margin = abs(expected) * tolerance_pct / 100 if expected else tolerance_pct
        try:
            return abs(float(cell) - float(expected)) <= margin
        except (TypeError, ValueError):
            # A number asked of a text or date column. Score it red; raising
            # would abort the run and lose every question already paid for.
            return False
    return _as_text(cell) == str(expected)


def _as_text(cell: Any) -> str:
    """A cell as the question would have written it, dates narrowed to a day."""
    if isinstance(cell, pd.Timestamp | datetime):
        return cell.strftime("%Y-%m-%d")
    return str(cell)


def _check_deger(spec: dict, state: SessionState) -> tuple[bool, str]:
    """One named cell of one table equals what was measured by hand."""
    suffix = spec.get("tablo")
    if suffix:
        frame = _table(state, suffix)
        if frame is None:
            return False, f"'{suffix}' tablosu uretilmedi"
    else:
        # Newest first, for the reason `_table` gives: a refetch writes the
        # column the answer actually rests on into a later table.
        frame = next(
            (
                state.tables[ref]
                for ref in sorted(state.tables, reverse=True)
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
    """Score one entry of a question's `checks:` list.

    A list entry carries exactly one check. Reading the first key and
    ignoring the rest let a hand-edited file lose a check silently, and
    silently is the wrong direction: the question then looks like it passed
    more than it did.
    """
    if not isinstance(spec, dict) or len(spec) != 1:
        return {
            "tip": "bicim",
            "gecti": False,
            "detay": f"bir kontrol girdisi tam olarak bir anahtar tasimali: {spec!r}",
        }
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
    entry = entry if isinstance(entry, dict) else {}
    question_id = str(entry.get("id") or "?")
    question = str(entry.get("question") or "")
    state = initial_state(f"eval-{question_id}")
    error = None if question else "bicim: soru metni yok"
    answer, duration = "", 0
    if question:
        try:
            result = asyncio.run(run_turn(state, deps, question))
            answer = result.answer or ""
            duration = result.duration_ms
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            logger.warning("%s dustu: %s", question_id, exc)
            error = f"{type(exc).__name__}: {exc}"

    # Scoring is inside the guard too. It reads a table a tool built and a
    # spec a person typed, so it has its own ways to raise -- a column that
    # is text where a number was expected, a half-written check. Outside the
    # guard those killed the whole run, and because the report is written
    # only after every question is scored, the measurement already paid for
    # was lost with it.
    try:
        missing_series = sorted(
            set(entry.get("expected_series") or []) - _used_series(state)
        )
        missing_tools = sorted(
            set(entry.get("expected_tools") or []) - _called_tools(state)
        )
        checks = [_run_check(spec, state, answer) for spec in entry.get("checks") or []]
    except Exception as exc:  # noqa: BLE001 - a bad check fails its row, not the run
        logger.warning("%s puanlanamadi: %s", question_id, exc)
        error = error or f"puanlama: {type(exc).__name__}: {exc}"
        missing_series, missing_tools, checks = [], [], []

    flags = state.verification.flags if state.verification else []
    # The real cost, not zero. `state.usage` is this turn's running total --
    # `_one` builds a fresh state per question -- so it holds what a crashed
    # question spent before it fell over, which `result.usage` never reaches.
    return {
        "id": question_id,
        "soru": question,
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
        "adimlar": _steps(state),
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
        "token": state.usage.total_tokens,
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
    entries = yaml.safe_load((path or QUESTIONS_PATH).read_text(encoding="utf-8")) or []
    deps = deps or _deps()
    rows = [_one(entry, deps) for entry in entries]
    return {
        "zaman": datetime.now(UTC).isoformat(),
        "model": deps.settings.mia_chat_model,
        "soru_sayisi": len(rows),
        "gecen": f"{sum(1 for row in rows if row['gecti'])}/{len(rows)}",
        "bayrakli": sum(1 for row in rows if row["bayrak"]),
        # A crash sets `truncated_reason` on its way out (`llm_error`), so a
        # crashed row would be counted here as well as under `hata`. One
        # failure, one count: this is the turns that ran out of budget or
        # steps without raising.
        "kesintili": sum(1 for row in rows if row["kesinti"] and not row["hata"]),
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
            # A turn that never reached the verifier produced an answer
            # nothing checked, which is worth seeing even when the row passed.
            ("dogrulanmadi", "" if row["dogrulandi"] else "verify adimi yok"),
        ):
            if detail:
                print(f"          {reason}: {detail}")
        for check in row["kontroller"]:
            if not check["gecti"]:
                print(f"          {check['tip']}: {check['detay']}")
        if not row["gecti"]:
            # A red row is not self-explaining: which tool, with what.
            for step in row["adimlar"]:
                mark = "" if step["ok"] else f" [{step['hata'] or 'basarisiz'}]"
                args = ", ".join(f"{k}={v}" for k, v in step["args"].items())
                print(f"          adim {step['arac']}{mark}: {args[:120]}")
        if row["bayrak"]:
            print(f"          bayrak: {'; '.join(row['bayrak_metinleri'])}")
    print(f"yazildi: {out}")
