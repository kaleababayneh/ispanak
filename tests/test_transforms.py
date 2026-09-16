"""Tests for lakehouse_agent.analysis.transforms. Pure functions, no
network/model/session -- small hand-computed examples only."""

import math
from datetime import date

import pandas as pd

from lakehouse_agent.analysis import transforms


def _series(periods, values):
    return pd.DataFrame({"period": periods, "value": values})


# --- to_freq ------------------------------------------------------------


def test_to_freq_stock_takes_last_value_in_target_period():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, formula = transforms.to_freq(df, "stock", "QS")

    assert list(result["value"]) == [30.0]
    assert "last" in formula


def test_to_freq_flow_sums_values_in_target_period():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, _ = transforms.to_freq(df, "flow", "QS")

    assert list(result["value"]) == [60.0]


def test_to_freq_rate_averages_values_in_target_period():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [10.0, 20.0, 30.0]
    )

    result, _ = transforms.to_freq(df, "rate", "QS")

    assert result["value"].iloc[0] == 20.0


def test_to_freq_flow_all_nan_window_stays_nan_not_zero():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [float("nan")] * 3
    )

    result, _ = transforms.to_freq(df, "flow", "QS")

    assert math.isnan(result["value"].iloc[0])


def test_to_freq_rejects_unknown_kind():
    df = _series([date(2024, 1, 1)], [1.0])
    try:
        transforms.to_freq(df, "bogus", "QS")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- deflate --------------------------------------------------------------


def test_deflate_divides_nominal_by_index_over_base():
    nominal = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])
    price_index = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])

    result, _ = transforms.deflate(nominal, price_index, base=100.0)

    assert [round(v, 6) for v in result["value"]] == [100.0, 100.0]


def test_deflate_missing_index_period_yields_nan_not_dropped_row():
    nominal = _series([date(2024, 1, 1), date(2024, 2, 1)], [100.0, 110.0])
    price_index = _series([date(2024, 1, 1)], [100.0])

    result, _ = transforms.deflate(nominal, price_index, base=100.0)

    assert len(result) == 2
    assert math.isnan(result["value"].iloc[1])


# --- rebase ---------------------------------------------------------------


def test_rebase_rescales_so_base_period_equals_base_value():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [50.0, 100.0, 150.0]
    )

    result, _ = transforms.rebase(df, date(2024, 2, 1), base_value=200.0)

    assert list(result["value"]) == [100.0, 200.0, 300.0]


def test_rebase_raises_when_base_period_not_in_series():
    df = _series([date(2024, 1, 1)], [50.0])
    try:
        transforms.rebase(df, date(2024, 6, 1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- pct_change / yoy / diff ----------------------------------------------


def test_pct_change_matches_hand_computed_values():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [100.0, 110.0, 121.0]
    )

    result, _ = transforms.pct_change(df)

    assert math.isnan(result["value"].iloc[0])
    assert round(result["value"].iloc[1], 6) == 10.0
    assert round(result["value"].iloc[2], 6) == 10.0


def test_yoy_uses_periods_per_year_as_the_lag():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2025, 1, 1), date(2025, 2, 1)],
        [100.0, 110.0, 120.0, 132.0],
    )

    result, _ = transforms.yoy(df, periods_per_year=2)

    assert math.isnan(result["value"].iloc[0])
    assert math.isnan(result["value"].iloc[1])
    assert round(result["value"].iloc[2], 6) == 20.0
    assert round(result["value"].iloc[3], 6) == 20.0


def test_diff_matches_hand_computed_values():
    df = _series(
        [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [100.0, 110.0, 90.0]
    )

    result, _ = transforms.diff(df)

    assert math.isnan(result["value"].iloc[0])
    assert result["value"].iloc[1] == 10.0
    assert result["value"].iloc[2] == -20.0


# --- align_join ------------------------------------------------------------


def test_align_join_outer_keeps_all_periods_missing_stays_nan():
    a = _series([date(2024, 1, 1), date(2024, 2, 1)], [1.0, 2.0])
    b = _series([date(2024, 2, 1), date(2024, 3, 1)], [20.0, 30.0])

    result, _ = transforms.align_join({"a": a, "b": b}, how="outer")

    assert list(result["period"]) == [
        date(2024, 1, 1),
        date(2024, 2, 1),
        date(2024, 3, 1),
    ]
    assert result["a"].iloc[0] == 1.0
    assert math.isnan(result["a"].iloc[2])
    assert math.isnan(result["b"].iloc[0])
    assert result["b"].iloc[2] == 30.0


def test_align_join_inner_keeps_only_common_periods():
    a = _series([date(2024, 1, 1), date(2024, 2, 1)], [1.0, 2.0])
    b = _series([date(2024, 2, 1), date(2024, 3, 1)], [20.0, 30.0])

    result, _ = transforms.align_join({"a": a, "b": b}, how="inner")

    assert list(result["period"]) == [date(2024, 2, 1)]
