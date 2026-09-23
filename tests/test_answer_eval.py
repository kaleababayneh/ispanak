"""Tests for the agent eval runner.

Offline throughout (CLAUDE.md #5). Most of these build a `SessionState` by
hand rather than driving the loop: the scoring rules are what is under test,
and a hand-built session says exactly which condition each test is about. The
two end-to-end cases at the bottom drive `run()` itself with a scripted
`FakeLLM`, to prove the wiring and the per-question isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from lakehouse_agent.agent import answer_eval
from lakehouse_agent.agent.state import (
    SessionState,
    Step,
    initial_state,
)
from lakehouse_agent.contracts import Source

FETCHED_AT = datetime(2026, 9, 23, tzinfo=UTC)
#: `QUESTIONS_PATH` is relative to the working directory, as `search_eval`'s
#: paths are. A test must not inherit that: anchored here, the suite passes
#: from any directory, the way every other fixture path in tests/ does.
SHIPPED_QUESTIONS = Path(__file__).parent.parent / answer_eval.QUESTIONS_PATH


def _state(
    *,
    tables: dict[str, pd.DataFrame] | None = None,
    series: Sequence[str] = (),
    urls: Sequence[str] = (),
    tools: Sequence[str] = (),
) -> SessionState:
    """A finished one-turn session, assembled from the parts scoring reads."""
    state = initial_state("test")
    state.turn_no = 1
    state.tables = tables or {}
    state.sources = [
        Source(series_id=sid, name=sid, fetched_at=FETCHED_AT) for sid in series
    ] + [Source(url=url, name=url, fetched_at=FETCHED_AT) for url in urls]
    state.steps = [
        Step(
            step_no=number,
            turn_no=1,
            node="act",
            tool=tool,
            started_at=FETCHED_AT,
        )
        for number, tool in enumerate(tools, start=1)
    ]
    return state


def _months(start: str, count: int) -> pd.Series:
    return pd.period_range(start, periods=count, freq="M").to_timestamp()


def _check(spec: dict, state: SessionState, answer: str = "") -> dict:
    return answer_eval._run_check(spec, state, answer)


# --- series and tools ------------------------------------------------------


def test_a_series_the_answer_never_used_is_missing():
    state = _state(series=["evds:TP.KTF12"])
    entry = {"expected_series": ["evds:TP.KTF12", "evds:TP.KFE.TR"]}

    assert sorted(set(entry["expected_series"]) - answer_eval._used_series(state)) == [
        "evds:TP.KFE.TR"
    ]


def test_extra_series_and_extra_tools_do_not_fail_a_question():
    """A superset passes: searching before fetching, or pulling a deflator
    nobody listed, is the agent doing its job, not drifting."""
    state = _state(
        series=["evds:TP.KTF12", "evds:TP.TUKFIY2025.GENEL"],
        tools=["search_catalog", "get_series", "transform_table"],
    )

    assert not set(["evds:TP.KTF12"]) - answer_eval._used_series(state)
    assert not set(["get_series"]) - answer_eval._called_tools(state)


def test_a_tool_cancelled_at_the_approval_gate_still_counts_as_called():
    """The fetch gate files a cancelled call under node="approval". Scoring
    filters on `tool`, not on the node, or the ledger would lose it."""
    state = _state()
    state.steps = [
        Step(
            step_no=1,
            turn_no=1,
            node="approval",
            tool="read_url",
            started_at=FETCHED_AT,
        )
    ]

    assert answer_eval._called_tools(state) == {"read_url"}


def test_steps_from_an_earlier_turn_are_not_counted():
    state = _state(tools=["get_series"])
    state.turn_no = 2

    assert answer_eval._called_tools(state) == set()


# --- cevapta_gecmeli -------------------------------------------------------


def test_the_answer_substring_check_ignores_case():
    state = _state()

    assert _check(
        {"cevapta_gecmeli": "Ağırlıklı Ortalama"}, state, "ağırlıklı ortalama alındı"
    )["gecti"]
    assert not _check({"cevapta_gecmeli": "medyan"}, state, "ağırlıklı ortalama")[
        "gecti"
    ]


# --- deger -----------------------------------------------------------------


def _rebased() -> dict[str, pd.DataFrame]:
    return {
        "t01_get_series": pd.DataFrame(
            {
                "period": _months("2021-01", 3),
                "TP_KFE_TR": [16.49, 16.98, 17.32],
                "TP_KFE_TR_endeks_202101": [100.0, 102.97, 105.03],
            }
        )
    }


def test_a_cell_inside_the_tolerance_passes_and_outside_it_fails():
    state = _state(tables=_rebased())
    spec = {
        "sutun": "TP_KFE_TR_endeks_202101",
        "donem": "2021-01",
        "tolerans_yuzde": 0.5,
    }

    assert _check({"deger": {**spec, "beklenen": 100}}, state)["gecti"]
    assert not _check({"deger": {**spec, "beklenen": 105}}, state)["gecti"]


def test_a_mean_and_a_sum_are_told_apart():
    """A5 in one assertion: 2026-06 has four weeks, mean 40,335 and sum
    161,34. Aggregating a rate by sum is the failure this catches."""
    state = _state(
        tables={
            "t01_get_series": pd.DataFrame(
                {"period": _months("2026-06", 1), "TP_KTF12": [40.335]}
            )
        }
    )
    spec = {"sutun": "TP_KTF12", "donem": "2026-06", "tolerans_yuzde": 0.5}

    assert _check({"deger": {**spec, "beklenen": 40.335}}, state)["gecti"]
    assert not _check({"deger": {**spec, "beklenen": 161.34}}, state)["gecti"]


def test_a_missing_period_is_reported_rather_than_crashing():
    state = _state(tables=_rebased())

    result = _check(
        {"deger": {"sutun": "TP_KFE_TR", "donem": "2025-01", "beklenen": 1}}, state
    )

    assert not result["gecti"]
    assert "2025-01" in result["detay"]


def test_a_named_table_that_was_never_produced_fails_clearly():
    result = _check(
        {"deger": {"tablo": "causality", "sutun": "anlamli", "beklenen": False}},
        _state(),
    )

    assert not result["gecti"]
    assert "causality" in result["detay"]


def test_a_date_cell_is_compared_as_a_date():
    state = _state(
        tables={
            "t02_detect_changes": pd.DataFrame(
                {"tarih": [pd.Timestamp("2023-07-01")], "sicrama_yuzde": [48.4]}
            )
        }
    )
    spec = {"tablo": "detect_changes", "sutun": "tarih"}

    assert _check({"deger": {**spec, "beklenen": "2023-07-01"}}, state)["gecti"]
    assert not _check({"deger": {**spec, "beklenen": "2023-08-01"}}, state)["gecti"]


def test_without_a_period_every_row_has_to_match():
    """`test_causality` writes one row per direction and which comes first
    depends on which column the model passed, so the verdict is checked
    across all rows rather than at an index."""
    both = pd.DataFrame({"yon": ["a -> b", "b -> a"], "anlamli": [False, False]})
    one = pd.DataFrame({"yon": ["a -> b", "b -> a"], "anlamli": [False, True]})
    spec = {"tablo": "causality", "sutun": "anlamli", "beklenen": False}

    assert _check({"deger": spec}, _state(tables={"t02_causality": both}))["gecti"]
    assert not _check({"deger": spec}, _state(tables={"t02_causality": one}))["gecti"]


def test_a_verdict_is_never_scored_as_arithmetic():
    """`isinstance(True, int)` holds in Python. Scored as a number with a
    tolerance, `beklenen: false` becomes "within 1 of zero", which `True`
    satisfies -- so the question would pass on the opposite verdict."""
    state = _state(tables={"t02_causality": pd.DataFrame({"anlamli": [True]})})

    result = _check(
        {
            "deger": {
                "tablo": "causality",
                "sutun": "anlamli",
                "beklenen": False,
                "tolerans_yuzde": 1,
            }
        },
        state,
    )

    assert not result["gecti"]


# --- tablo_var and kaynak_url_var ------------------------------------------


def test_a_table_is_found_by_the_suffix_its_tool_asked_for():
    state = _state(
        tables={"t01_get_series": pd.DataFrame(), "t03_lag_corr": pd.DataFrame()}
    )

    assert _check({"tablo_var": "lag_corr"}, state)["gecti"]
    assert not _check({"tablo_var": "detect_changes"}, state)["gecti"]


def test_a_web_source_is_required_only_where_the_question_asks_for_one():
    with_url = _state(urls=["https://www.tcmb.gov.tr/duy2023-25"])
    without = _state(series=["evds:TP.KTF12"])

    assert _check({"kaynak_url_var": True}, with_url)["gecti"]
    assert not _check({"kaynak_url_var": True}, without)["gecti"]


# --- tabloda_donem_yok -----------------------------------------------------


def test_an_invented_early_value_is_caught():
    state = _state(
        tables={
            "t01_get_series": pd.DataFrame(
                {"period": _months("2021-01", 3), "TP_HPBITABLO6_3": [1.0, 2.0, 3.0]}
            )
        }
    )

    result = _check(
        {"tabloda_donem_yok": {"sutun": "TP_HPBITABLO6_3", "once": "2024-06"}}, state
    )

    assert not result["gecti"]
    assert "2021-01" in result["detay"]


def test_empty_cells_before_the_series_started_are_not_an_invention():
    """EVDS hands back a cell for every period in the window and leaves it
    empty before the series was issued. Fetching 2021-2026 and getting empty
    2021 rows is correct behaviour, not a fabricated number."""
    state = _state(
        tables={
            "t01_get_series": pd.DataFrame(
                {
                    "period": _months("2024-04", 4),
                    "TP_HPBITABLO6_3": [None, None, 1.0, 2.0],
                }
            )
        }
    )

    assert _check(
        {"tabloda_donem_yok": {"sutun": "TP_HPBITABLO6_3", "once": "2024-06"}}, state
    )["gecti"]


def test_a_second_series_with_real_early_values_is_not_the_refused_one():
    """The check is scoped to one column: an agent that legitimately pulls
    CPI alongside must not fail for CPI having 2021 values."""
    state = _state(
        tables={
            "t01_get_series": pd.DataFrame(
                {
                    "period": _months("2021-01", 2),
                    "TP_HPBITABLO6_3": [None, None],
                    "TP_TUKFIY2025_GENEL": [16.1, 16.4],
                }
            )
        }
    )

    assert _check(
        {"tabloda_donem_yok": {"sutun": "TP_HPBITABLO6_3", "once": "2024-06"}}, state
    )["gecti"]


def test_an_unknown_check_type_fails_loudly_instead_of_passing():
    result = _check({"bilinmeyen_kontrol": 1}, _state())

    assert not result["gecti"]
    assert "bilinmeyen" in result["detay"]


# --- end to end ------------------------------------------------------------


def _questions(tmp_path, entries) -> object:
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
    return path


def _deps(script):
    from lakehouse_agent.agent.loop import AgentDeps
    from lakehouse_agent.config import AgentSettings
    from lakehouse_agent.testing import FakeLLM
    from lakehouse_agent.testing.fake_tools import make_fake_registry

    return AgentDeps(
        llm=FakeLLM(script),
        registry=make_fake_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        sink=None,
    )


def test_a_question_that_blows_up_does_not_take_the_run_with_it(tmp_path):
    from lakehouse_agent.testing import FakeLLM

    # The script runs out after the first question, so the second raises
    # inside `run_turn` exactly as a model error would.
    deps = _deps([FakeLLM.text("Plan."), FakeLLM.text("Faiz yukseldi.")])
    path = _questions(
        tmp_path,
        [
            {
                "id": "Q1",
                "question": "Faiz nasil?",
                "checks": [{"cevapta_gecmeli": "yukseldi"}],
            },
            {"id": "Q2", "question": "Ya endeks?"},
        ],
    )

    report = answer_eval.run(deps, path=path)

    assert [row["id"] for row in report["sorular"]] == ["Q1", "Q2"]
    assert report["sorular"][0]["gecti"]
    assert not report["sorular"][1]["gecti"]
    assert report["sorular"][1]["hata"]
    assert report["gecen"] == "1/2"


def test_each_question_starts_from_a_session_of_its_own(tmp_path):
    """Sharing one session, as the CLI does for follow-ups, would let the
    first question's table satisfy the second question's check."""
    from lakehouse_agent.testing import FakeLLM

    deps = _deps(
        [
            # Q1 fetches: plan returns the call, then observe, then answer.
            FakeLLM.tool_call(
                "get_series",
                {
                    "series_ids": ["evds:TP.KTF12"],
                    "start": "2021-01",
                    "end": "2025-12",
                    "into": "new",
                },
            ),
            FakeLLM.text("Tablo hazir."),
            FakeLLM.text("Faiz tablosu kuruldu."),
            # Q2 calls nothing: plan, then answer.
            FakeLLM.text("Arac gerekmiyor."),
            FakeLLM.text("Tabloya bakmadim."),
        ]
    )
    path = _questions(
        tmp_path,
        [
            {
                "id": "Q1",
                "question": "Faizi getir.",
                "checks": [{"tablo_var": "get_series"}],
            },
            {
                "id": "Q2",
                "question": "Baska bir sey.",
                "checks": [{"tablo_var": "get_series"}],
            },
        ],
    )

    report = answer_eval.run(deps, path=path)

    assert report["sorular"][0]["kontroller"][0]["gecti"]
    assert not report["sorular"][1]["kontroller"][0]["gecti"]


def test_the_report_carries_the_headline_keys(tmp_path):
    from lakehouse_agent.testing import FakeLLM

    deps = _deps([FakeLLM.text("Plan."), FakeLLM.text("Cevap.")])
    path = _questions(tmp_path, [{"id": "Q1", "question": "Soru?"}])

    report = answer_eval.run(deps, path=path)

    assert set(report) >= {
        "zaman",
        "model",
        "soru_sayisi",
        "gecen",
        "bayrakli",
        "kesintili",
        "sure_ms",
        "token",
        "sorular",
    }
    assert report["soru_sayisi"] == 1


def test_verifier_flags_are_reported_without_failing_the_row(tmp_path):
    """The verifier is itself under measurement; making a flag a pass
    condition would reward quietly loosening it. So an answer carrying a
    number no table backs is counted, named, and still passes its checks."""
    from lakehouse_agent.testing import FakeLLM

    deps = _deps([FakeLLM.text("Plan."), FakeLLM.text("Faiz 987,65 oldu.")])
    path = _questions(
        tmp_path,
        [{"id": "Q1", "question": "Faiz?", "checks": [{"cevapta_gecmeli": "faiz"}]}],
    )

    row = answer_eval.run(deps, path=path)["sorular"][0]

    assert row["bayrak"] == 1
    assert "987,65" in row["bayrak_metinleri"][0]
    assert row["dogrulandi"]
    assert row["gecti"]


# --- the shipped question file ---------------------------------------------


def test_every_shipped_question_uses_a_known_check_type():
    """The runner answers an unknown check with a failure, so a typo in the
    file would look like an agent regression on the day of a demo."""
    entries = yaml.safe_load(SHIPPED_QUESTIONS.read_text(encoding="utf-8"))
    known = {
        "cevapta_gecmeli",
        "deger",
        "tablo_var",
        "kaynak_url_var",
        "tabloda_donem_yok",
    }

    used = {
        kind
        for entry in entries
        for check in entry.get("checks") or []
        for kind in check
    }

    assert used <= known, f"bilinmeyen kontrol turu: {used - known}"
    assert {entry["id"] for entry in entries} == {f"A{n}" for n in range(1, 8)} | {
        f"B{n}" for n in range(1, 6)
    }


def test_the_shipped_questions_cover_every_registered_tool():
    """A tool with no question is a tool nothing measures."""
    from lakehouse_agent.tools.toolset import make_registry

    entries = yaml.safe_load(SHIPPED_QUESTIONS.read_text(encoding="utf-8"))
    asked = {tool for entry in entries for tool in entry.get("expected_tools") or []}

    assert set(make_registry().names()) <= asked


@pytest.mark.parametrize("entry_id", ["A2", "A3", "A5"])
def test_the_column_names_in_the_file_are_the_ones_the_tools_produce(entry_id):
    """These are generated, not chosen by the model: a question naming a
    column the tools never write can only ever fail."""
    from lakehouse_agent.tools.lakehouse import column_name

    entries = {
        entry["id"]: entry
        for entry in yaml.safe_load(SHIPPED_QUESTIONS.read_text(encoding="utf-8"))
    }
    expected = {
        "A2": f"{column_name('evds:TP.KFE.TR')}_endeks_202101",
        "A3": f"{column_name('evds:TP.AKONUTSAT2.KTRTOPLAM')}_pay",
        "A5": column_name("evds:TP.KTF12"),
    }

    named = entries[entry_id]["checks"][0]["deger"]["sutun"]

    assert named == expected[entry_id]


# --- what the 23 Sep review found ------------------------------------------


def test_a_broken_check_fails_its_row_and_the_run_goes_on(tmp_path):
    """Scoring used to sit outside the guard, so a half-written check killed
    the whole run -- and since the report is written only at the end, every
    question already paid for in MIA calls was lost with it."""
    from lakehouse_agent.testing import FakeLLM

    deps = _deps(
        [FakeLLM.text("P."), FakeLLM.text("C."), FakeLLM.text("P."), FakeLLM.text("C.")]
    )
    path = _questions(
        tmp_path,
        [
            {"id": "Q1", "question": "a?", "checks": [{"deger": {"beklenen": 1}}]},
            {"id": "Q2", "question": "b?"},
        ],
    )

    report = answer_eval.run(deps, path=path)

    assert [row["id"] for row in report["sorular"]] == ["Q1", "Q2"]
    assert not report["sorular"][0]["gecti"]
    assert "puanlama" in report["sorular"][0]["hata"]
    assert report["sorular"][1]["gecti"]


def test_a_refetch_is_read_from_the_newer_table():
    """A5: the agent pulls the weekly rate into t01, converts to a monthly
    mean in t02. Reading the earliest table scored the right answer red."""
    state = _state(
        tables={
            "t01_get_series": pd.DataFrame(
                {"period": _months("2026-06", 1), "TP_KTF12": [40.1]}
            ),
            "t02_get_series": pd.DataFrame(
                {"period": _months("2026-06", 1), "TP_KTF12": [40.335]}
            ),
        }
    )

    result = _check(
        {
            "deger": {
                "sutun": "TP_KTF12",
                "donem": "2026-06",
                "beklenen": 40.335,
                "tolerans_yuzde": 0.5,
            }
        },
        state,
    )

    assert result["gecti"], result["detay"]


def test_a_missing_cell_is_not_a_verdict():
    """An empty verdict column satisfied `beklenen: false`, which is the
    "the test never ran" case the check exists to catch."""
    empty = _state(tables={"t01_causality": pd.DataFrame({"anlamli": [None, None]})})
    spec = {"deger": {"tablo": "causality", "sutun": "anlamli", "beklenen": False}}

    assert not _check(spec, empty)["gecti"]


def test_pandas_na_does_not_take_the_run_down():
    state = _state(
        tables={
            "t01_causality": pd.DataFrame(
                {"anlamli": pd.array([None, True], dtype="boolean")}
            )
        }
    )

    result = _check(
        {"deger": {"tablo": "causality", "sutun": "anlamli", "beklenen": True}}, state
    )

    assert not result["gecti"]


def test_a_number_asked_of_a_text_column_fails_rather_than_raising():
    state = _state(tables={"t01_causality": pd.DataFrame({"yon": ["a -> b"]})})

    result = _check(
        {"deger": {"tablo": "causality", "sutun": "yon", "beklenen": 1}}, state
    )

    assert not result["gecti"]


def test_two_checks_in_one_entry_are_refused_instead_of_half_run():
    """Reading the first key and dropping the rest failed open: the question
    looked like it had passed more than it did."""
    result = _check({"cevapta_gecmeli": "x", "tablo_var": "y"}, _state(), "x")

    assert not result["gecti"]
    assert "bir anahtar" in result["detay"]


def test_a_crashed_question_reports_what_it_spent(tmp_path):
    from lakehouse_agent.testing import FakeLLM

    deps = _deps([FakeLLM.text("Plan.")])  # runs out before the answer
    path = _questions(tmp_path, [{"id": "Q1", "question": "a?"}])

    report = answer_eval.run(deps, path=path)
    row = report["sorular"][0]

    assert row["hata"]
    assert row["token"] > 0, "gercek maliyet state.usage'da duruyor"
    # `run_turn` stamps truncated_reason on its way out, so the row carries
    # both; counting it twice would overstate how many turns ran out of road.
    assert report["kesintili"] == 0


def test_a_tool_name_the_registry_does_not_know_is_not_a_call(tmp_path):
    from lakehouse_agent.testing import FakeLLM

    deps = _deps(
        [
            FakeLLM.tool_call("yok_boyle_arac", {}),
            FakeLLM.text("O."),
            FakeLLM.text("C."),
        ]
    )
    path = _questions(
        tmp_path,
        [{"id": "Q1", "question": "a?", "expected_tools": ["yok_boyle_arac"]}],
    )

    report = answer_eval.run(deps, path=path)

    assert report["sorular"][0]["eksik_arac"] == ["yok_boyle_arac"]


def test_a_question_without_a_text_fails_its_row_only(tmp_path):
    from lakehouse_agent.testing import FakeLLM

    deps = _deps([FakeLLM.text("P."), FakeLLM.text("C.")])
    path = _questions(tmp_path, [{"id": "Q1"}, {"id": "Q2", "question": "b?"}])

    report = answer_eval.run(deps, path=path)

    assert not report["sorular"][0]["gecti"]
    assert report["sorular"][1]["gecti"]


def test_an_empty_question_file_is_an_empty_report(tmp_path):
    path = tmp_path / "bos.yaml"
    path.write_text("", encoding="utf-8")

    report = answer_eval.run(_deps([]), path=path)

    assert report["soru_sayisi"] == 0
    assert report["gecen"] == "0/0"


def test_the_report_records_which_tool_was_called_with_what(tmp_path):
    """A red row is not self-explaining. Without the arguments there was no
    way to tell which address a failing read_url had been given."""
    from lakehouse_agent.testing import FakeLLM

    deps = _deps(
        [
            FakeLLM.tool_call(
                "get_series",
                {
                    "series_ids": ["evds:TP.KTF12"],
                    "start": "2021-01",
                    "end": "2025-12",
                    "into": "new",
                },
            ),
            FakeLLM.text("Tablo hazir."),
            FakeLLM.text("Cevap."),
        ]
    )
    path = _questions(tmp_path, [{"id": "Q1", "question": "a?"}])

    steps = answer_eval.run(deps, path=path)["sorular"][0]["adimlar"]

    assert [step["arac"] for step in steps] == ["get_series"]
    assert steps[0]["ok"] is True
    assert steps[0]["args"]["series_ids"].startswith("['evds:TP.KTF12'")


def test_a_long_argument_is_cut_before_it_reaches_the_report():
    state = _state()
    state.steps = [
        Step(
            step_no=1,
            turn_no=1,
            node="act",
            tool="read_url",
            args={"url": "https://example.com/" + "x" * 500},
            started_at=FETCHED_AT,
        )
    ]

    assert len(answer_eval._steps(state)[0]["args"]["url"]) == answer_eval.ARG_CHARS
