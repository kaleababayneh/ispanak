"""Tests for the synthetic data lane.

Two jobs. First, lock the shapes the agent lane is being built against, so that
swapping in the real lake later is a swap and not a rewrite. Second, guarantee
the planted fact the scenario depends on actually exists — otherwise the test
that asserts the agent finds it would pass without proving anything.
"""

import pandas as pd
import pytest

from lakehouse_agent.contracts import SeriesNotFound
from lakehouse_agent.testing.fake_lake import (
    CPI_BASE_YEAR,
    CPI_SERIES,
    CREDIT_SERIES,
    PLANTED_RATE_DOWN_CREDIT_FLAT,
    RATE_SERIES,
    RATE_STOCK_SERIES,
)

OBSERVATION_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
]


# --- shapes ---------------------------------------------------------------


def test_get_returns_the_seven_gold_columns_in_order(fake_lake):
    frame = fake_lake.get([CREDIT_SERIES], "2021-01", "2021-06")
    assert list(frame.columns) == OBSERVATION_COLUMNS


def test_period_is_a_datetime_column_not_python_dates(fake_lake):
    """pandas 3 merges date objects against datetimes into nothing, silently."""
    frame = fake_lake.get([CREDIT_SERIES], "2021-01", "2021-06")
    assert pd.api.types.is_datetime64_any_dtype(frame["period"])


def test_monthly_periods_are_the_first_of_the_month(fake_lake):
    frame = fake_lake.get([CREDIT_SERIES], "2021-01", "2021-06")
    assert (frame["period"].dt.day == 1).all()


def test_weekly_periods_are_mondays(fake_lake):
    """The data lane normalises weekly periods to the Monday of the week."""
    frame = fake_lake.get([RATE_SERIES], "2021-01", "2021-03")
    assert (frame["period"].dt.weekday == 0).all()


def test_unknown_series_raises_and_names_them(fake_lake):
    with pytest.raises(SeriesNotFound) as caught:
        fake_lake.get(["evds:YOK.1", CREDIT_SERIES], "2021-01", "2021-06")
    assert "evds:YOK.1" in str(caught.value)
    assert CREDIT_SERIES not in str(caught.value)  # only the missing one


# --- wide ----------------------------------------------------------------


def test_scenario_wide_is_sixty_months_by_five_columns(scenario_wide):
    assert scenario_wide.shape == (60, 5)
    assert scenario_wide.columns[0] == "period"


def test_wide_column_names_are_the_series_ids(scenario_wide):
    assert CREDIT_SERIES in scenario_wide.columns
    assert RATE_SERIES in scenario_wide.columns


def test_wide_has_no_gaps_for_the_scenario_series(scenario_wide):
    assert int(scenario_wide.isna().sum().sum()) == 0


def test_wide_period_stays_a_datetime_column(scenario_wide):
    assert pd.api.types.is_datetime64_any_dtype(scenario_wide["period"])


# --- frequency conversion -------------------------------------------------


def test_weekly_rate_becomes_the_monthly_mean(fake_lake):
    weekly = fake_lake.get([RATE_SERIES], "2022-03", "2022-06")[["period", "value"]]
    monthly = fake_lake.to_freq(weekly, "M", "rate")
    assert len(monthly) == 4

    march = weekly[weekly["period"].dt.month == 3]["value"].mean()
    assert monthly["value"].iloc[0] == pytest.approx(march, rel=1e-9)


def test_stock_takes_the_last_value_of_the_period(fake_lake):
    """A balance is not summed or averaged; the period's closing level is the value."""
    monthly = fake_lake.get([CREDIT_SERIES], "2021-01", "2021-03")[["period", "value"]]
    quarterly_last = fake_lake.to_freq(monthly, "M", "stock")
    assert quarterly_last["value"].iloc[-1] == monthly["value"].iloc[-1]


# --- the planted fact -----------------------------------------------------


def test_planted_window_is_exactly_the_months_the_scenario_needs(scenario_wide):
    """The scenario asks: did the rate ever fall while credit did not grow?

    If this ever fails, the agent test that finds those months is testing
    nothing, so it is asserted here at the source.
    """
    frame = scenario_wide.copy()
    frame["month"] = frame["period"].dt.strftime("%Y-%m")
    frame["rate_change"] = frame[RATE_SERIES].diff()
    frame["credit_change"] = frame[CREDIT_SERIES].diff()

    found = frame[(frame["rate_change"] < 0) & (frame["credit_change"] <= 0)]
    assert found["month"].tolist() == PLANTED_RATE_DOWN_CREDIT_FLAT


def test_rate_both_rises_and_falls_over_the_window(scenario_wide):
    changes = scenario_wide[RATE_SERIES].diff().dropna()
    assert (changes > 0).any()
    assert (changes < 0).any()


# --- CPI base -------------------------------------------------------------


def test_cpi_base_year_mean_is_exactly_one_hundred(fake_lake):
    """The real series is 2025=100, so deflating against it must behave the same."""
    cpi = fake_lake.get([CPI_SERIES], f"{CPI_BASE_YEAR}-01", f"{CPI_BASE_YEAR}-12")
    assert cpi["value"].mean() == pytest.approx(100.0, abs=1e-6)


def test_cpi_rises_monotonically(fake_lake):
    cpi = fake_lake.get([CPI_SERIES], "2021-01", "2026-06")
    assert (cpi["value"].diff().dropna() > 0).all()


# --- catalog search -------------------------------------------------------


def test_explicit_series_code_returns_exactly_one_hit(fake_lake):
    hits = fake_lake.search("TP.KTF12 serisini getir")
    assert len(hits) == 1
    assert hits[0].series_id == RATE_SERIES


def test_housing_rate_query_is_ambiguous_between_flow_and_stock(fake_lake):
    """Flow vs stock is the ambiguity the approval step exists to resolve."""
    hits = fake_lake.search("konut kredisi faiz orani")
    top_two = {hit.series_id for hit in hits[:2]}
    assert top_two == {RATE_SERIES, RATE_STOCK_SERIES}


def test_ambiguity_is_measured_as_a_ratio_not_an_absolute_gap(fake_lake):
    """Hybrid search scores are not 0..1, so only their ratio is meaningful."""
    hits = fake_lake.search("konut kredisi faiz orani")
    assert hits[1].score / hits[0].score >= 0.9


def test_search_of_nonsense_returns_nothing(fake_lake):
    assert fake_lake.search("zzzz qqqq") == []


def test_catalog_lookup_returns_all_fifteen_columns(fake_lake):
    frame = fake_lake.catalog_lookup([RATE_SERIES, CREDIT_SERIES])
    assert len(frame.columns) == 15
    assert len(frame) == 2


# --- quality card ---------------------------------------------------------


def test_profile_table_reports_coverage_and_frequency(fake_lake, scenario_wide):
    report = fake_lake.profile_table(scenario_wide)
    assert report.rows == 60
    assert report.date_range == ("2021-01", "2025-12")
    assert report.freq == "M"
    assert report.missing_pct == 0.0
    assert report.duplicates == 0


def test_profile_table_flags_duplicate_periods(fake_lake, scenario_wide):
    doubled = pd.concat([scenario_wide, scenario_wide.head(1)], ignore_index=True)
    report = fake_lake.profile_table(doubled)
    assert report.duplicates == 1
    assert report.issues


def test_prepare_table_sorts_and_drops_unparseable_dates(fake_lake):
    messy = pd.DataFrame(
        {"period": ["2021-03-01", "bozuk", "2021-01-01"], "value": [3.0, 2.0, 1.0]}
    )
    cleaned, report = fake_lake.prepare_table(messy)
    assert len(cleaned) == 2
    assert cleaned["value"].tolist() == [1.0, 3.0]
    assert report.rows == 2


def test_prepare_table_reports_how_many_rows_it_dropped(fake_lake):
    """`profile_table` runs on the cleaned frame, so the drop is invisible to it.
    Without this line the card describes a table the caller never handed over,
    and the quality gate a human approves is missing the one fact that mattered.
    """
    messy = pd.DataFrame(
        {"period": ["2021-01-01", "bozuk", "yine bozuk"], "value": [1.0, 2.0, 3.0]}
    )
    _, report = fake_lake.prepare_table(messy)
    assert any("2 satir" in issue for issue in report.issues), report.issues


def test_prepare_table_stays_quiet_when_nothing_was_dropped(fake_lake):
    clean = pd.DataFrame({"period": ["2021-01-01", "2021-02-01"], "value": [1.0, 2.0]})
    _, report = fake_lake.prepare_table(clean)
    assert not any("dusuruldu" in issue for issue in report.issues)


def test_missing_pct_is_zero_when_there_is_nothing_numeric_to_measure(fake_lake):
    """A table of labels has no cells whose absence could be counted. Reporting
    a percentage computed as if one numeric column existed would be a finding
    invented out of a denominator."""
    labels = pd.DataFrame(
        {
            "period": pd.date_range("2021-01-01", periods=3, freq="MS"),
            "etiket": ["a", "b", "c"],
        }
    )
    assert fake_lake.profile_table(labels).missing_pct == 0.0


# --- determinism ----------------------------------------------------------


def test_two_instances_produce_identical_data():
    """Tests must not depend on which instance built the frame."""
    from lakehouse_agent.testing.fake_lake import FakeLake

    left = FakeLake().wide([CREDIT_SERIES, RATE_SERIES], "2021-01", "2021-12")
    right = FakeLake().wide([CREDIT_SERIES, RATE_SERIES], "2021-01", "2021-12")
    pd.testing.assert_frame_equal(left, right)


# --- regressions found in review -----------------------------------------


def test_first_month_aggregates_to_its_intended_level(fake_lake):
    """The weekly series must not start before the pool does.

    Weeks reaching back into December 2020 were clamped onto January's level and
    then filtered out by date, leaving January with a partial set of weeks whose
    wobbles no longer cancelled. The month then averaged to 12.06 instead of the
    12.0 the rate path defines, which is exactly the kind of quiet drift a fake
    must not have.
    """
    weekly = fake_lake.get([RATE_SERIES], "2021-01", "2021-01")[["period", "value"]]
    monthly = fake_lake.to_freq(weekly, "M", "rate")
    assert monthly["value"].iloc[0] == pytest.approx(12.0, abs=1e-9)


def test_no_week_falls_before_the_pool_start(fake_lake):
    weekly = fake_lake.get([RATE_SERIES], "2021-01", "2026-06")
    assert weekly["period"].min() >= pd.Timestamp("2021-01-01")


def test_wide_rejects_an_empty_series_list(fake_lake):
    with pytest.raises(ValueError, match="en az bir"):
        fake_lake.wide([], "2021-01", "2021-06")


def test_wide_rejects_duplicate_series_ids(fake_lake):
    """A duplicate would come back as `<id>_x` / `<id>_y` from the merge and
    quietly break the one-column-per-series promise."""
    with pytest.raises(ValueError, match="tekrar"):
        fake_lake.wide([RATE_SERIES, RATE_SERIES], "2021-01", "2021-06")


def test_to_freq_rejects_unsupported_frequencies(fake_lake):
    """`Freq` declares D, W, M, Q and A; only two are implemented, and the other
    three must say so rather than surface as a bare KeyError."""
    frame = fake_lake.get([CREDIT_SERIES], "2021-01", "2021-06")[["period", "value"]]
    with pytest.raises(ValueError, match="Desteklenmeyen"):
        fake_lake.to_freq(frame, "Q", "stock")


def test_profile_table_counts_missing_periods(fake_lake, scenario_wide):
    """The quality card decides an approval, so it must not certify a table it
    never checked: a hole has to show up as a hole."""
    with_hole = scenario_wide.drop(index=[10, 11, 12]).reset_index(drop=True)
    report = fake_lake.profile_table(with_hole)
    assert report.missing_periods == 3
    assert report.missing_pct > 0
    assert any("eksik" in issue for issue in report.issues)


def test_profile_table_reports_no_gaps_for_a_complete_table(fake_lake, scenario_wide):
    report = fake_lake.profile_table(scenario_wide)
    assert report.missing_periods == 0
    assert report.issues == []


def test_profile_table_counts_outliers(fake_lake, scenario_wide):
    spiked = scenario_wide.copy()
    spiked.loc[5, CREDIT_SERIES] = spiked[CREDIT_SERIES].max() * 50
    assert fake_lake.profile_table(spiked).outliers >= 1
