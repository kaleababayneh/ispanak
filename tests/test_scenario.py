"""The scenario's three questions, one session, real tools, real data.

Plan 3.3's definition of done, as a test: the brief's konut kredisi flow runs
end to end against the committed gold data through the production tool set,
with only the model scripted. What is asserted is what a jury would check by
hand -- the monthly average of a weekly rate, the months the question asks
about, a deflated value, and that each follow-up only *adds* columns.

No network: search has no embeddings here and falls back to keyword matching.
Over the whole pool (45k series, 22 Sep) that fallback no longer finds the
exact series a short query means -- it says so in its own warning, and a test
holds it to saying so. What the semantic search finds is measured separately
(`lake.search_eval`, eval/results).
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from lakehouse_agent.agent.loop import AgentDeps, run_turn
from lakehouse_agent.agent.state import initial_state
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.lake import gold
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.tools.lake_adapter import LakeAdapter
from lakehouse_agent.tools.toolset import make_registry

pytestmark = pytest.mark.skipif(
    not (gold.OBSERVATIONS_PATH.exists() and gold.SERIES_CATALOG_PATH.exists()),
    reason="gerçek gold verisi yok (data/gold/*.parquet)",
)

CREDIT_ID = "bddk:aylik:tuketici:konut:TL:sektor"
RATE_ID = "evds:TP.KTF12"
HPI_ID = "evds:TP.KFE.TR"
CPI_ID = "evds:TP.TUKFIY2025.GENEL"
CREDIT, RATE, HPI = "aylik_tuketici_konut_TL_sektor", "TP_KTF12", "TP_KFE_TR"
TABLE = "t01_get_series"


def _offline(*_args):
    raise RuntimeError("testte embedding yok")


def _call(name, args, n):
    return FakeLLM.tool_call(name, args, call_id=f"c{n}")


QUESTION_1 = (
    "2021-2025 yillari arasindaki 60 aylik konut kredilerini ve konut kredisi "
    "faiz oranlarini goster. Faiz dustugu halde kredinin artmadigi donem oldu mu?"
)
QUESTION_2 = "Tabloyu bozmadan sadece konut kredisi tutarlarini enflasyondan arindir."
QUESTION_3 = (
    "Bu tabloyu hic bozmadan yeni bir sutun olarak konut fiyat endeksini getir."
)

SCRIPT = [
    # Question 1
    _call("search_catalog", {"query": "konut kredisi bakiyesi"}, 1),
    _call("search_catalog", {"query": "konut kredisi faiz orani"}, 2),
    _call(
        "get_series",
        {"series_ids": [CREDIT_ID, RATE_ID], "start": "2021-01", "end": "2025-12"},
        3,
    ),
    _call("transform_table", {"op": "diff", "column": RATE}, 4),
    _call("transform_table", {"op": "diff", "column": CREDIT}, 5),
    _call(
        "transform_table",
        {
            "op": "mark",
            "name": "faiz_dustu_kredi_artmadi",
            "conditions": [
                {"column": f"{RATE}_diff", "op": "<", "value": 0},
                {"column": f"{CREDIT}_diff", "op": "<=", "value": 0},
            ],
        },
        6,
    ),
    FakeLLM.text("Aylar listelendi."),
    FakeLLM.text("Cevap 1."),
    # Question 2
    _call("transform_table", {"op": "deflate", "column": CREDIT}, 7),
    FakeLLM.text("Reel sutun eklendi."),
    FakeLLM.text("Cevap 2."),
    # Question 3
    _call(
        "get_series",
        {
            "series_ids": [HPI_ID],
            "start": "2021-01",
            "end": "2025-12",
            "into": "active",
        },
        8,
    ),
    FakeLLM.text("KFE eklendi."),
    FakeLLM.text("Cevap 3."),
]


@pytest.fixture(scope="module")
def session():
    """Run the three turns once; every test reads what they left behind."""
    deps = AgentDeps(
        llm=FakeLLM(list(SCRIPT)),
        registry=make_registry(),
        settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
        lake=LakeAdapter(embed_documents=_offline, embed_query=_offline),
    )
    state = initial_state("senaryo")
    snapshots, results = [], []
    for question in (QUESTION_1, QUESTION_2, QUESTION_3):
        results.append(asyncio.run(run_turn(state, deps, question)))
        snapshots.append(state.tables[TABLE].copy())
    return state, snapshots, results, deps.llm.calls


def _tool_message(state, call_id):
    return next(
        m["content"]
        for m in state.messages
        if m.get("role") == "tool" and m["tool_call_id"] == call_id
    )


def test_every_turn_finishes_with_an_answer_and_one_table(session):
    state, _, results, _ = session

    assert [r.answer for r in results] == ["Cevap 1.", "Cevap 2.", "Cevap 3."]
    assert all(r.truncated_reason is None for r in results)
    assert list(state.tables) == [TABLE]
    assert all(step.ok is not False for step in state.steps if step.node == "act")


def test_question_1_builds_sixty_months_of_credit_and_rate(session):
    _, (after_1, _, _), _, _ = session

    assert len(after_1) == 60
    assert after_1["period"].min() == pd.Timestamp("2021-01-01")
    assert after_1["period"].max() == pd.Timestamp("2025-12-01")
    assert after_1[CREDIT].notna().all()


def test_the_weekly_rate_is_the_average_of_the_weeks_observed_that_month(session):
    _, (after_1, _, _), _, _ = session

    weekly = gold.get([RATE_ID], "2021-01", "2025-12")
    observed = pd.to_datetime(weekly["obs_date"]).dt.to_period("M")
    by_month = weekly.groupby(observed)["value"].mean()
    table = after_1.set_index(after_1["period"].dt.to_period("M"))[RATE]
    pd.testing.assert_series_equal(table, by_month, check_names=False)


def test_the_keyword_search_offers_candidates_and_warns_they_are_coarse(session):
    """Without embeddings, search falls back to keyword matching over the
    whole pool. It still answers with candidates and never claims the pool
    lacks the data (planner rule 5) -- but the credit query's own series is
    no longer among them, and the warning the model reads says the ranking
    is coarse. Measured 22 Sep over the 45k pool: neither query's own series
    is among its five candidates, while the semantic search puts them 3rd and
    4th (eval/results/search_eval_2026-09-22.json). What is asserted is what
    holds whatever the ranking does -- candidates, and the warning that says
    not to read them as a semantic match. Which series come back is left out
    on purpose: it depends on data that is not in the repository.
    """
    state, _, _, _ = session
    credit, rate = _tool_message(state, "c1"), _tool_message(state, "c2")

    for message in (credit, rate):
        assert "aday seri bulundu" in message
        assert "kelime eslesmesiyle bulundu" in message
        # The preview writes its column names once and the rows as lists, so
        # the match kind is a value under the `eslesme` column, not a key.
        assert "eslesme" in message and '"kelime"' in message
        assert "eslesen seri yok" not in message


def test_the_conversion_to_months_reaches_the_answer(session):
    """Planner rule 8: a conversion the answer does not repeat makes it wrong.
    Checked in the answer prompt itself, which is what obliges the answer --
    the tool message alone proved only that the tool said it."""
    _, _, _, calls = session

    answer_prompt = next(
        call["messages"][0]["content"]
        for call in calls
        if call["tag"].startswith("answer@")
    )
    assert "TP_KTF12" in answer_prompt and "ortalamasi" in answer_prompt


def test_question_1_is_answered_by_the_tool_with_the_right_months(session):
    """The months where the rate fell and credit did not grow, recomputed
    from gold directly -- not from the table the tool built, which would agree
    with any misalignment inside it -- and compared with what the tool listed.
    The four months are the real answer recorded in docs/decisions.md."""
    state, (after_1, _, _), _, _ = session

    weekly = gold.get([RATE_ID], "2021-01", "2025-12")
    observed = pd.to_datetime(weekly["obs_date"]).dt.to_period("M")
    rate = weekly.groupby(observed)["value"].mean().diff()
    monthly = gold.get([CREDIT_ID], "2021-01", "2025-12")
    credit = monthly.set_index(monthly["period"].dt.to_period("M"))["value"].diff()
    months = rate.index[(rate < 0) & (credit.reindex(rate.index) <= 0)]
    expected = [month.strftime("%Y-%m") for month in months]

    assert expected == ["2021-02", "2021-05", "2023-12", "2024-01"]
    message = _tool_message(state, "c6")
    assert f"{len(expected)} ay kosullarin hepsini sagliyor: " in message
    assert f"sagliyor: {', '.join(expected)}." in message
    flags = after_1.set_index("period")["faiz_dustu_kredi_artmadi"]
    assert list(flags.index[flags.eq(True).fillna(False)].strftime("%Y-%m")) == expected


def test_question_2_adds_only_the_real_credit_column(session):
    _, (after_1, after_2, _), _, _ = session

    assert list(after_2.columns) == [*after_1.columns, f"reel_{CREDIT}"]
    pd.testing.assert_frame_equal(after_2[after_1.columns], after_1)


def test_question_2_deflates_with_cpi_2025_equals_100(session):
    _, (_, after_2, _), _, _ = session

    cpi = gold.get([CPI_ID], "2023-06", "2023-06")["value"].iloc[0]
    row = after_2.set_index("period").loc[pd.Timestamp("2023-06-01")]
    assert row[f"reel_{CREDIT}"] == pytest.approx(row[CREDIT] / (cpi / 100))


def test_question_2_cites_the_price_index_and_its_base(session):
    state, _, _, _ = session

    assert CPI_ID in {source.series_id for source in state.sources}
    lineage = state.lineage[f"{TABLE}.reel_{CREDIT}"]
    assert lineage.source_series == [CREDIT_ID, CPI_ID]
    assert "2025" in _tool_message(state, "c7")


def test_question_3_adds_the_house_price_index_without_touching_the_rest(session):
    _, (_, after_2, after_3), _, _ = session

    assert list(after_3.columns) == [*after_2.columns, HPI]
    pd.testing.assert_frame_equal(after_3[after_2.columns], after_2)
    kfe = gold.get([HPI_ID], "2021-01", "2021-01")["value"].iloc[0]
    assert after_3[HPI].iloc[0] == pytest.approx(kfe)


def test_the_session_can_trace_every_column_back_to_its_series(session):
    """The traceability panel's data: every column of the final table has a
    lineage entry, and every series behind them is a cited source."""
    state, (_, _, after_3), _, _ = session

    for column in after_3.columns.drop("period"):
        assert f"{TABLE}.{column}" in state.lineage, column
    cited = {source.series_id for source in state.sources}
    assert {CREDIT_ID, RATE_ID, CPI_ID, HPI_ID} <= cited
