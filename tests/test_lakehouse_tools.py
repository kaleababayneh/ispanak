"""Tests for the real Lakehouse tools (tools/lakehouse.py).

Two kinds of lake underneath. `FakeLake` for the behaviour, because its series
carry planted answers -- the months where the rate fell and credit did not
grow are known in advance -- so the expected values are written down, not
recomputed. The committed gold data for what only real data can show: a
weekly series really averaged into months, a series really starting late.
Neither needs the network: `FakeLake` searches by keyword, and the real-gold
search tests run with an embedding function that fails, i.e. on the keyword
fallback.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.contracts import ToolResult
from lakehouse_agent.testing.fake_lake import (
    CPI_SERIES,
    CREDIT_SERIES,
    FETCHED_AT,
    HPI_SERIES,
    PLANTED_RATE_DOWN_CREDIT_FLAT,
    RATE_SERIES,
    FakeLake,
)
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.tools import lakehouse
from lakehouse_agent.tools.lakehouse import (
    TOOLS,
    TransformArgs,
    column_name,
    make_transform_table,
)
from lakehouse_agent.tools.registry import ToolRegistry

REGISTRY = ToolRegistry.of(TOOLS)
RATE, CREDIT, HPI = (column_name(s) for s in (RATE_SERIES, CREDIT_SERIES, HPI_SERIES))


class Session:
    """The part of the loop these tests need: run a tool, merge what it wrote.

    Merging only on success mirrors `loop._merge_context`, so a chain of calls
    here sees exactly what it would see inside a turn.
    """

    def __init__(self, lake):
        self.lake = lake
        self.tables: dict[str, pd.DataFrame] = {}
        self.lineage: dict = {}
        self.active: str | None = None
        self.step = 0

    def run(self, tool: str, **raw) -> ToolResult:
        self.step += 1
        ctx = ToolContext(
            session_id="s1",
            step_no=self.step,
            llm=FakeLLM(),
            lake=self.lake,
            settings=AgentSettings(MIA_API_KEY="fake-key-unused-offline"),
            active_table=self.active,
            lineage=self.lineage,
            _tables=self.tables,
        )
        result = REGISTRY.execute(tool, REGISTRY.validate(tool, raw), ctx)
        if result.ok:
            self.tables.update(ctx.written)
            if ctx.new_active:
                self.active = ctx.new_active
            for column in result.columns:
                self.lineage[f"{result.table_ref}.{column.name}"] = column
        return result

    @property
    def table(self) -> pd.DataFrame:
        return self.tables[self.active]


@pytest.fixture
def session():
    return Session(FakeLake())


@pytest.fixture
def scenario(session):
    """Scenario step 1's table: housing credit and the rate, 2021-01..2025-12."""
    result = session.run(
        "get_series",
        series_ids=[CREDIT_SERIES, RATE_SERIES],
        start="2021-01",
        end="2025-12",
    )
    assert result.ok, result.summary
    return session


# --- search_catalog ---------------------------------------------------------


def test_search_hands_the_gate_the_fields_it_reads(session):
    result = session.run("search_catalog", query="konut kredisi faiz")

    assert result.ok
    assert REGISTRY.require("search_catalog").returns_candidates
    assert {"series_id", "name_tr", "unit", "freq", "kind", "score"} <= set(
        result.preview[0]
    )


def test_search_that_finds_nothing_says_the_pool_lacks_it(session):
    """The honest answer, phrased so the model does not substitute a series."""
    result = session.run("search_catalog", query="dolar kuru")

    assert result.ok
    assert result.preview == []
    assert "eslesen seri yok" in result.summary


# --- get_series -------------------------------------------------------------


def test_get_series_builds_one_monthly_table_and_makes_it_active(scenario):
    table = scenario.table

    assert list(table.columns) == ["period", CREDIT, RATE]
    assert len(table) == 60
    assert pd.api.types.is_datetime64_any_dtype(table["period"])


def test_a_weekly_rate_is_averaged_into_months_and_says_so(session):
    result = session.run(
        "get_series", series_ids=[RATE_SERIES], start="2021-01", end="2021-12"
    )

    assert any(RATE in w and "ortalamasi" in w for w in result.warnings)
    # The month a week counts in is the one it was observed in, and that is
    # said: until 22 Sep it was silently its Monday's.
    assert any("gozlendigi gunun" in w for w in result.warnings)
    column = result.columns[0]
    assert (column.freq, column.kind, column.op) == ("M", "rate", "to_freq")
    assert "ortalamasi" in column.formula


def test_every_series_gets_a_source_with_its_real_fetch_time(scenario):
    sources = {
        s.series_id: s
        for s in scenario.run(
            "get_series", series_ids=[HPI_SERIES], start="2021-01", end="2021-03"
        ).sources
    }

    # The lake's own stamp, not the time the tool ran.
    assert sources[HPI_SERIES].fetched_at == FETCHED_AT


def test_an_unknown_series_sends_the_model_back_to_search(session):
    result = session.run(
        "get_series", series_ids=["evds:TP.YOK"], start="2021-01", end="2021-12"
    )

    assert not result.ok
    assert "search_catalog" in result.summary


def test_a_window_that_ends_before_it_starts_is_refused(session):
    result = session.run(
        "get_series", series_ids=[RATE_SERIES], start="2022-01", end="2021-01"
    )

    assert not result.ok


def test_a_month_that_does_not_exist_is_rejected_by_the_schema(session):
    with pytest.raises(ValidationError):
        session.run(
            "get_series", series_ids=[RATE_SERIES], start="2021-13", end="2022-01"
        )


def test_into_active_adds_columns_and_leaves_the_rest_byte_identical(scenario):
    """Scenario step 3: "bu tabloyu hiç bozmadan yeni bir sütun"."""
    before = scenario.table.copy()
    ref = scenario.active

    result = scenario.run(
        "get_series",
        series_ids=[HPI_SERIES],
        start="2021-01",
        end="2025-12",
        into="active",
    )

    assert result.ok and result.table_ref == ref
    after = scenario.tables[ref]
    assert list(after.columns) == [*before.columns, HPI]
    pd.testing.assert_frame_equal(after[before.columns], before)


def test_into_active_refuses_a_column_that_is_already_there(scenario):
    result = scenario.run(
        "get_series",
        series_ids=[RATE_SERIES],
        start="2021-01",
        end="2025-12",
        into="active",
    )

    assert not result.ok
    assert "zaten var" in result.summary


def test_into_active_counts_the_gaps_in_the_table_it_lands_in(scenario):
    """Fetched for 2024-2025 onto a 2021-2025 table, the new column is empty
    for three years there, although the fetch itself had no gap."""
    result = scenario.run(
        "get_series",
        series_ids=[HPI_SERIES],
        start="2024-01",
        end="2025-12",
        into="active",
    )

    assert result.ok
    assert any(HPI in w and "60 ayin 36'inda deger yok" in w for w in result.warnings)
    assert "60 ay, 2021-01 - 2025-12" in result.summary


def test_into_active_does_not_blame_the_pool_for_the_window_that_was_asked(scenario):
    """The gap in front of an added column is the request's doing, not the
    pool's. Saying "seri havuzda 2024-01 itibariyla basliyor" about a series
    the pool holds from 2021 puts a false "we do not have it" in the answer
    (planner rule 5)."""
    result = scenario.run(
        "get_series",
        series_ids=[HPI_SERIES],
        start="2024-01",
        end="2025-12",
        into="active",
    )

    gap = next(w for w in result.warnings if HPI in w and "deger yok" in w)
    assert "havuzda" not in gap
    assert "yalnizca 2024-01 - 2025-12 icin getirildi" in gap
    # The pool really does hold it from the table's first month.
    assert scenario.lake.get([HPI_SERIES], "2021-01", "2021-03")["value"].notna().any()


# --- transform_table: the scenario's operations -----------------------------


def test_diff_adds_the_month_on_month_change(scenario):
    result = scenario.run("transform_table", op="diff", column=RATE)

    table = scenario.table
    expected = table[RATE].diff()
    pd.testing.assert_series_equal(table[f"{RATE}_diff"], expected, check_names=False)
    assert result.columns[0].unit == "yuzde puan"
    # Said in the summary too: the 22 Sep live answer turned a -0.2785 point
    # change into "%27,85 puan"; the unit is what the model reads first.
    assert "(birim: yuzde puan)" in result.summary
    assert pd.isna(table[f"{RATE}_diff"].iloc[0])  # no previous month, not zero


def test_mark_names_the_months_where_every_condition_holds(scenario):
    """Scenario step 1's question, answered by the tool, not by the model
    counting rows. The fake lake planted exactly these months."""
    scenario.run("transform_table", op="diff", column=RATE)
    scenario.run("transform_table", op="diff", column=CREDIT)

    result = scenario.run(
        "transform_table",
        op="mark",
        name="faiz_dustu_kredi_artmadi",
        conditions=[
            {"column": f"{RATE}_diff", "op": "<", "value": 0},
            {"column": f"{CREDIT}_diff", "op": "<=", "value": 0},
        ],
    )

    assert result.ok
    for month in PLANTED_RATE_DOWN_CREDIT_FLAT:
        assert month in result.summary
    flags = scenario.table.set_index("period")["faiz_dustu_kredi_artmadi"]
    flagged = list(flags.index[flags.eq(True).fillna(False)].strftime("%Y-%m"))
    assert flagged == PLANTED_RATE_DOWN_CREDIT_FLAT
    assert "1 ay eksik veri" in result.summary  # the first month has no diff


def test_a_flag_column_is_refused_by_the_arithmetic_operations(scenario):
    """A `mark` column is a yes/no, and dividing by it is not a question with
    an answer: `pct_change` raised `ZeroDivisionError` straight out of the
    tool (its first period is False, so the change from it divides by zero),
    and `diff` quietly added a column of XOR-ish numbers with no unit."""
    scenario.run("transform_table", op="diff", column=RATE)
    scenario.run(
        "transform_table",
        op="mark",
        conditions=[{"column": f"{RATE}_diff", "op": "<", "value": 0}],
    )

    for op, extra in (
        ("pct_change", {}),
        ("diff", {}),
        ("rebase", {"base_period": "2021-06"}),
    ):
        result = scenario.run("transform_table", op=op, column="isaret", **extra)
        assert not result.ok, op
        assert "isaret" in result.summary

    # Still usable where it means something: as a condition of another mark.
    again = scenario.run(
        "transform_table",
        op="mark",
        name="ayni_ay",
        conditions=[{"column": "isaret", "op": "==", "value": 1}],
    )
    assert again.ok


def test_deflate_refuses_a_table_the_price_index_does_not_reach(scenario, monkeypatch):
    """An all-empty column is not a deflated series. Returned as a success it
    invited the model to describe real amounts that were never computed; the
    refusal says what to do instead."""
    empty_cpi = pd.DataFrame(
        {
            "period": scenario.table["period"],
            "value": [float("nan")] * len(scenario.table),
        }
    )
    monkeypatch.setattr(
        lakehouse, "_cpi", lambda lake, start, end: empty_cpi, raising=True
    )

    result = scenario.run("transform_table", op="deflate", column=CREDIT)

    assert not result.ok
    assert "TUFE havuzda olmadigi icin" in result.summary
    assert f"reel_{CREDIT}" not in scenario.table.columns


def test_deflate_turns_a_money_amount_real_and_cites_the_price_index(scenario):
    """Scenario step 2: "sadece konut kredisi tutarlarını enflasyondan arındır"."""
    before = scenario.table.copy()

    result = scenario.run("transform_table", op="deflate", column=CREDIT)

    assert result.ok
    column = result.columns[0]
    assert column.name == f"reel_{CREDIT}"
    assert column.source_series == [CREDIT_SERIES, CPI_SERIES]
    assert [s.series_id for s in result.sources] == [CPI_SERIES]
    assert any("2025" in w for w in result.warnings)
    pd.testing.assert_frame_equal(scenario.table[before.columns], before)


def test_deflate_at_a_base_month_leaves_that_month_unchanged(scenario):
    result = scenario.run(
        "transform_table", op="deflate", column=CREDIT, base_period="2021-01"
    )

    first = scenario.table.iloc[0]
    assert result.columns[0].name == f"reel_{CREDIT}_202101"
    assert first[f"reel_{CREDIT}_202101"] == pytest.approx(first[CREDIT])


def test_deflate_base_month_may_lie_outside_the_table(session):
    """ "At December 2025 prices" for a 2021-2023 table. The CPI was read for
    the table's window only, so a month the pool holds was called missing."""
    session.run(
        "get_series", series_ids=[CREDIT_SERIES], start="2021-01", end="2023-12"
    )

    result = session.run(
        "transform_table", op="deflate", column=CREDIT, base_period="2025-12"
    )

    assert result.ok, result.summary
    lake = FakeLake()
    cpi = lake.wide([CPI_SERIES], "2021-01", "2025-12", "M").set_index("period")
    cpi = cpi[CPI_SERIES]
    first = session.table.iloc[0]
    expected = first[CREDIT] * cpi.iloc[-1] / cpi.iloc[0]
    assert first[f"reel_{CREDIT}_202512"] == pytest.approx(expected)


def test_deflate_at_two_base_months_gives_two_columns(scenario):
    """One name for every base month made the second request a refusal that
    pointed the model at the first month's prices."""
    first = scenario.run("transform_table", op="deflate", column=CREDIT)
    second = scenario.run(
        "transform_table", op="deflate", column=CREDIT, base_period="2021-01"
    )

    assert first.ok and second.ok
    assert {f"reel_{CREDIT}", f"reel_{CREDIT}_202101"} <= set(scenario.table.columns)


def test_a_real_column_is_not_deflated_twice(scenario):
    scenario.run("transform_table", op="deflate", column=CREDIT)

    result = scenario.run("transform_table", op="deflate", column=f"reel_{CREDIT}")

    assert not result.ok
    assert "zaten enflasyondan arindirilmis" in result.summary


def test_deflating_an_interest_rate_is_refused(scenario):
    """What the 22 Sep live run did with the fake tool: a "real" rate made by
    dividing a percentage by CPI, labelled with a 2025 base. Meaningless."""
    result = scenario.run("transform_table", op="deflate", column=RATE)

    assert not result.ok
    assert "oran" in result.summary
    assert f"reel_{RATE}" not in scenario.table.columns


def test_rebase_puts_the_base_month_at_100(session):
    session.run("get_series", series_ids=[HPI_SERIES], start="2021-01", end="2021-12")

    session.run("transform_table", op="rebase", column=HPI, base_period="2021-01")

    assert session.table[f"{HPI}_endeks_202101"].iloc[0] == pytest.approx(100.0)


def test_change_since_measures_against_the_base_month(session):
    session.run("get_series", series_ids=[CPI_SERIES], start="2021-01", end="2021-12")
    cpi = column_name(CPI_SERIES)

    session.run("transform_table", op="change_since", column=cpi, base_period="2021-01")

    series = session.table[f"{cpi}_degisim_202101"]
    table = session.table
    assert series.iloc[0] == pytest.approx(0.0)
    assert series.iloc[-1] == pytest.approx(
        (table[cpi].iloc[-1] / table[cpi].iloc[0] - 1) * 100
    )


def test_ratio_is_a_percentage_share(scenario):
    scenario.run("transform_table", op="ratio", column=CREDIT, denominator=CREDIT)

    assert scenario.table[f"{CREDIT}_pay"].dropna().eq(100.0).all()


def test_a_ratio_over_another_denominator_gets_its_own_column(scenario):
    """Both are "CREDIT_pay". Refused as "the same operation, use the existing
    column", the model was steered to answer with the other share."""
    scenario.run("transform_table", op="ratio", column=CREDIT, denominator=CREDIT)

    result = scenario.run(
        "transform_table", op="ratio", column=CREDIT, denominator=RATE
    )

    assert result.ok
    assert result.columns[0].name == f"{CREDIT}_pay_2"
    assert result.columns[0].inputs == [CREDIT, RATE]
    assert "baska bir hesapla dolu" in result.summary
    assert scenario.table[f"{CREDIT}_pay"].dropna().eq(100.0).all()


def test_a_second_mark_with_other_conditions_is_not_the_same_operation(scenario):
    scenario.run("transform_table", op="diff", column=RATE)
    scenario.run(
        "transform_table",
        op="mark",
        conditions=[{"column": f"{RATE}_diff", "op": "<", "value": 0}],
    )

    result = scenario.run(
        "transform_table",
        op="mark",
        conditions=[{"column": f"{RATE}_diff", "op": ">", "value": 0}],
    )

    assert result.ok
    assert result.columns[0].name == "isaret_2"
    rose, fell = scenario.table["isaret_2"], scenario.table["isaret"]
    assert not (rose.eq(True) & fell.eq(True)).any()


def test_yoy_uses_the_columns_own_frequency_for_the_lag(scenario):
    scenario.run("transform_table", op="yoy", column=CREDIT)

    values = scenario.table[f"{CREDIT}_yoy"]
    assert values.iloc[:12].isna().all()
    assert values.iloc[12:].notna().all()


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ({"op": "rebase", "column": "x"}, "base_period"),
        ({"op": "ratio", "column": "x"}, "denominator"),
        ({"op": "mark"}, "conditions"),
        ({"op": "diff"}, "column"),
    ],
)
def test_a_missing_argument_is_named_before_anything_runs(raw, field):
    """A validation error the loop feeds back once, naming what to add."""
    with pytest.raises(ValidationError, match=field):
        REGISTRY.validate("transform_table", raw)


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ({"op": "diff", "column": "x", "denominator": "y"}, "denominator"),
        ({"op": "pct_change", "column": "x", "base_period": "2021-01"}, "base_period"),
        ({"op": "diff", "column": "x", "name": "fark"}, "name"),
        (
            {
                "op": "diff",
                "column": "x",
                "conditions": [{"column": "x", "op": "<", "value": 0}],
            },
            "conditions",
        ),
    ],
)
def test_an_argument_the_operation_ignores_is_refused(raw, field):
    """A stray denominator on a diff went into the lineage as an input it
    never had; a base month on pct_change reads as a "change since"."""
    with pytest.raises(ValidationError, match=field):
        REGISTRY.validate("transform_table", raw)


def test_the_period_column_is_refused_with_the_usable_columns(scenario):
    """'period' passed the "column exists" check and then broke the tool with a
    bare KeyError, which told the model nothing about what to use."""
    result = scenario.run("transform_table", op="diff", column="period")

    assert not result.ok
    assert "sayisal" in result.summary and RATE in result.summary


def test_a_column_that_is_not_in_the_table_lists_the_ones_that_are(scenario):
    result = scenario.run("transform_table", op="diff", column="uydurma")

    assert not result.ok
    assert RATE in result.summary


def test_running_the_same_operation_twice_is_refused_not_overwritten(scenario):
    scenario.run("transform_table", op="diff", column=RATE)

    result = scenario.run("transform_table", op="diff", column=RATE)

    assert not result.ok
    assert "ayni islemle zaten hesaplanmis" in result.summary
    assert list(scenario.table.columns).count(f"{RATE}_diff") == 1


def test_no_lake_is_a_clear_failure_not_a_crash():
    result = Session(None).run("search_catalog", query="faiz")

    assert not result.ok


# --- real gold (committed, offline) ------------------------------------------


@pytest.fixture
def real():
    from lakehouse_agent.lake import gold
    from lakehouse_agent.tools.lake_adapter import LakeAdapter

    if not gold.OBSERVATIONS_PATH.exists():
        pytest.skip("gerçek gold verisi yok (data/gold/*.parquet)")
    return Session(LakeAdapter())


def test_real_weekly_rate_is_the_average_of_the_weeks_starting_in_that_month(real):
    from lakehouse_agent.lake import gold

    real.run("get_series", series_ids=["evds:TP.KTF12"], start="2021-01", end="2021-12")

    weekly = gold.get(["evds:TP.KTF12"], "2021-01", "2021-01")
    assert real.table["TP_KTF12"].iloc[0] == pytest.approx(weekly["value"].mean())


def test_real_series_that_starts_late_says_where_it_starts(real):
    result = real.run(
        "get_series",
        series_ids=["evds:TP.HPBITABLO6.3"],
        start="2021-01",
        end="2025-12",
    )

    assert any("2024-06 itibariyla basliyor" in w for w in result.warnings)


def test_real_pool_ending_before_the_window_says_where_it_ends(real):
    result = real.run(
        "get_series",
        series_ids=["bddk:aylik:tuketici:konut:TL:sektor"],
        start="2025-01",
        end="2027-12",
    )

    assert any("2026-06 ayinda bitiyor" in w for w in result.warnings)


def test_real_money_in_two_scales_in_one_table_is_pointed_out(real):
    """BDDK credit is in milyon TL, the TCMB weekly credit stock in bin TL: a
    thousandfold gap nobody converts (CLAUDE.md #3), so it is said instead."""
    result = real.run(
        "get_series",
        series_ids=["bddk:aylik:tuketici:konut:TL:sektor", "evds:TP.HPBITABLO6.3"],
        start="2025-01",
        end="2025-12",
    )

    scale = [w for w in result.warnings if "farkli olcekte" in w]
    assert scale and "bin TL" in scale[0] and "milyon TL" in scale[0]


def test_one_money_scale_raises_no_scale_warning(real):
    result = real.run(
        "get_series",
        series_ids=["bddk:aylik:tuketici:konut:TL:sektor"],
        start="2025-01",
        end="2025-12",
    )

    assert not any("farkli olcekte" in w for w in result.warnings)


def test_real_rate_difference_is_in_percentage_points(real):
    """The pool's rate is labelled "Ağırlıklı ortalama", not "%", so the unit
    check behind the 22 Sep "%27,85 puan" fix never fired on real data."""
    real.run("get_series", series_ids=["evds:TP.KTF12"], start="2021-01", end="2021-12")

    result = real.run("transform_table", op="diff", column="TP_KTF12")

    assert result.columns[0].unit == "yuzde puan"
    assert "(birim: yuzde puan)" in result.summary


def test_real_count_of_houses_is_not_deflated(real):
    """House sales are a flow, in "Adet": "at 2025 prices" means nothing."""
    real.run(
        "get_series",
        series_ids=["evds:TP.AKONUTSAT1.KTRTOPLAM"],
        start="2021-01",
        end="2021-12",
    )

    result = real.run("transform_table", op="deflate", column="TP_AKONUTSAT1_KTRTOPLAM")

    assert not result.ok
    assert "parasal bir tutar degil" in result.summary


def test_real_weekly_series_has_no_phantom_month_before_the_window(real):
    """Asked from 2020-12, the pool's single December week (Monday 28 Dec,
    observed Friday 1 Jan) came back as a December "monthly average". By
    observation date that week is January's, and there is no December."""
    result = real.run(
        "get_series", series_ids=["evds:TP.KTF12"], start="2020-12", end="2021-03"
    )

    months = list(real.table["period"].dt.strftime("%Y-%m"))
    assert months == ["2021-01", "2021-02", "2021-03"]
    assert not [w for w in result.warnings if "eksik haftayla" in w]
    assert any("Istenen baslangic 2020-12" in w for w in result.warnings)


def test_a_month_with_fewer_weeks_than_fridays_is_pointed_out():
    """January 2021 has five Fridays; a series with four of them makes a
    monthly average from part of the month, which the table cannot show."""
    from lakehouse_agent.tools.lakehouse import _partial_period_warnings

    fridays = ["2021-01-01", "2021-01-08", "2021-01-15", "2021-01-29"]
    long = pd.DataFrame(
        {
            "series_id": "evds:TP.X",
            "obs_date": pd.to_datetime(fridays),
            "value": 1.0,
        }
    )
    info = {"evds:TP.X": pd.Series({"freq": "W"})}

    warnings = _partial_period_warnings(
        long, ["evds:TP.X"], {"evds:TP.X": "TP_X"}, info, "M"
    )

    assert warnings == [
        "TP_X: su donemler eksik haftayla hesaplandi, deger donemin tamamini "
        "yansitmiyor: 2021-01 (4/5 hafta)."
    ]


def test_real_search_without_embeddings_says_its_ranking_is_coarse():
    """On the keyword fallback the model has to learn two things: these
    candidates were not ranked by meaning, and an empty answer would prove
    nothing about the pool (planner rule 5). Over 45k series a short query
    now matches something whatever it asks -- what it must not do is look
    like a semantic hit."""
    from lakehouse_agent.lake import gold
    from lakehouse_agent.tools.lake_adapter import LakeAdapter

    if not gold.SERIES_CATALOG_PATH.exists():
        pytest.skip("gerçek gold verisi yok (data/gold/*.parquet)")

    def unavailable(*_args):
        raise RuntimeError("MIA yok")

    session = Session(LakeAdapter(embed_documents=unavailable, embed_query=unavailable))

    result = session.run("search_catalog", query="enflasyon")
    assert result.ok
    assert any("kelime eslesmesiyle" in warning for warning in result.warnings)
    assert all(hit["eslesme"] == "kelime" for hit in result.preview)
    assert "eslesen seri yok" not in result.summary

    # What still works when the ranking is coarse: naming the series. A query
    # carrying an id is answered by that series alone, whatever the search is
    # able to do that day -- which is the escape hatch the model is told about.
    named = session.run("search_catalog", query="evds:TP.TUKFIY2025.GENEL verisi")
    assert [hit["series_id"] for hit in named.preview] == ["evds:TP.TUKFIY2025.GENEL"]
    assert named.preview[0]["eslesme"] == "kod"


def test_real_weekly_series_can_come_back_weekly(real):
    """ "Haftalik goster": the table keeps the weeks, and nothing is
    converted, so nothing is said about a conversion."""
    result = real.run(
        "get_series",
        series_ids=["evds:TP.KTF12"],
        start="2021-01",
        end="2021-02",
        freq="W",
    )

    assert result.ok, result.summary
    assert len(real.table) == 9  # Fridays 1 Jan .. 26 Feb 2021, by their Mondays
    assert result.columns[0].freq == "W"
    assert not any("cevrildi" in w for w in result.warnings)
    assert "hafta" in result.summary


def test_a_monthly_series_is_not_put_in_a_weekly_table(real):
    """Each week would repeat the month's value: data made up (CLAUDE.md #4)."""
    result = real.run(
        "get_series",
        series_ids=["bddk:aylik:tuketici:konut:TL:sektor"],
        start="2021-01",
        end="2021-03",
        freq="W",
    )

    assert not result.ok
    assert "veri uydurmak" in result.summary


def test_a_quarterly_table_takes_the_quarters_last_balance(real):
    from lakehouse_agent.lake import gold

    result = real.run(
        "get_series",
        series_ids=["bddk:aylik:tuketici:konut:TL:sektor"],
        start="2021-01",
        end="2021-06",
        freq="Q",
    )

    monthly = gold.get(["bddk:aylik:tuketici:konut:TL:sektor"], "2021-01", "2021-06")
    assert list(real.table.iloc[:, 1]) == [
        monthly["value"].iloc[2],
        monthly["value"].iloc[5],
    ]
    assert any("ceyregin son degeri" in w for w in result.warnings)
    assert "2021-Q1 - 2021-Q2" in result.summary


def test_into_active_refuses_another_frequency(real):
    real.run(
        "get_series",
        series_ids=["bddk:aylik:tuketici:konut:TL:sektor"],
        start="2021-01",
        end="2021-06",
    )

    result = real.run(
        "get_series",
        series_ids=["evds:TP.KTF12"],
        start="2021-01",
        end="2021-06",
        freq="W",
        into="active",
    )

    assert not result.ok
    assert "freq='M'" in result.summary


def test_ytd_rebuilds_the_running_total_of_a_monthly_flow(real):
    """House sales are a monthly flow: January's total is January's sales,
    February's is January plus February."""
    real.run(
        "get_series",
        series_ids=["evds:TP.AKONUTSAT1.KTRTOPLAM"],
        start="2024-01",
        end="2024-03",
    )

    result = real.run("transform_table", op="ytd", column="TP_AKONUTSAT1_KTRTOPLAM")

    sales = list(real.table["TP_AKONUTSAT1_KTRTOPLAM"])
    totals = list(real.table["TP_AKONUTSAT1_KTRTOPLAM_yilbasi"])
    assert result.ok
    assert totals == [sales[0], sales[0] + sales[1], sum(sales)]


def test_ytd_is_refused_for_a_balance(scenario):
    result = scenario.run("transform_table", op="ytd", column=CREDIT)

    assert not result.ok
    assert "aylik bir akim degil" in result.summary


# --- transform_table: top ----------------------------------------------------


def _top_context(frame, ref="t01_read_url"):
    return ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM([]),
        lake=None,
        settings=AgentSettings(MIA_API_KEY="test-only"),
        active_table=ref,
        _tables={ref: frame},
    )


def test_top_names_the_rows_where_a_column_is_highest():
    """ "Which month was the highest?" is counting, and the model may not
    count (planner rule 1). The rows are named and their values returned, so
    the answer can quote them."""
    frame = pd.DataFrame(
        {
            "period": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
            "hacim": [93824682381.0, 90534914196.0, 151349741320.0],
        }
    )
    spec = make_transform_table()

    result = spec.fn(
        TransformArgs(op="top", column="hacim", n=2, table_ref="t01_read_url"),
        _top_context(frame),
    )

    assert result.ok
    assert "2026-03" in result.summary and "151349741320" in result.summary
    assert "2026-02" not in result.summary  # only the two asked for
    assert [row["satir"] for row in result.preview] == ["2026-03", "2026-01"]


def test_top_works_on_a_table_that_has_no_period_column():
    """A document read from a URL is often a printed table whose rows are
    named, not dated; every other operation needs a period and this one
    cannot."""
    frame = pd.DataFrame(
        {"AY": ["Ocak", "Subat", "Mart"], "hacim": [93.0, 90.0, 151.0]}
    )
    spec = make_transform_table()

    result = spec.fn(
        TransformArgs(op="top", column="hacim", n=1, table_ref="t01_read_url"),
        _top_context(frame),
    )

    assert result.ok and "Mart: 151" in result.summary


def test_top_can_ask_for_the_lowest_and_says_what_it_could_not_rank():
    frame = pd.DataFrame(
        {"AY": ["Ocak", "Subat", "Mart"], "hacim": [93.0, float("nan"), 151.0]}
    )
    spec = make_transform_table()

    result = spec.fn(
        TransformArgs(
            op="top", column="hacim", n=2, ascending=True, table_ref="t01_read_url"
        ),
        _top_context(frame),
    )

    assert "en dusuk" in result.summary
    assert result.preview[0]["satir"] == "Ocak"
    assert "1 satir bos deger" in result.summary


def test_top_refuses_a_column_that_is_not_numbers():
    frame = pd.DataFrame({"AY": ["Ocak", "Subat"], "hacim": [1.0, 2.0]})
    spec = make_transform_table()

    result = spec.fn(
        TransformArgs(op="top", column="AY", table_ref="t01_read_url"),
        _top_context(frame),
    )

    assert result.ok is False and "sayisal" in result.summary


def test_top_adds_no_column():
    """It reports; it does not change the table a person is looking at."""
    frame = pd.DataFrame({"AY": ["Ocak", "Subat"], "hacim": [1.0, 2.0]})
    ctx = _top_context(frame)

    make_transform_table().fn(
        TransformArgs(op="top", column="hacim", table_ref="t01_read_url"), ctx
    )

    assert ctx.written == {}
