"""Tests for the lead-lag analysis (analysis/causality.py).

Synthetic series with a lag built into them, so the expected answer is known
before the test runs rather than read off the output. Seeded: a test that
passes four times in five is worse than no test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.analysis.causality import (
    ALPHA,
    NotEnoughData,
    lagged_correlation,
    lead_lag,
)

LAG = 3


def _periods(n: int) -> pd.Series:
    return pd.date_range("2018-01-01", periods=n, freq="MS")


def _leading_pair(n: int = 90, lag: int = LAG, seed: int = 7) -> pd.DataFrame:
    """`kredi` follows `fiyat` by `lag` months, plus noise."""
    rng = np.random.default_rng(seed)
    fiyat = pd.Series(rng.normal(size=n)).cumsum() + 100
    kredi = 0.8 * fiyat.shift(lag) + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"period": _periods(n), "fiyat": fiyat, "kredi": kredi})


def _independent(n: int = 90, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "period": _periods(n),
            "a": pd.Series(rng.normal(size=n)).cumsum() + 50,
            "b": pd.Series(rng.normal(size=n)).cumsum() + 50,
        }
    )


# --- what the test finds -----------------------------------------------------


def test_the_leading_series_is_found_and_the_other_direction_is_not():
    """The pair is built so that `fiyat` moves first. One direction must come
    back significant and the other must not -- a test that called both ways
    significant would say nothing."""
    result, _correlation, _summary = lead_lag(
        _leading_pair(), "fiyat", "kredi", max_lag=6
    )

    forward = result[result["yon"] == "fiyat -> kredi"].iloc[0]
    backward = result[result["yon"] == "kredi -> fiyat"].iloc[0]
    assert forward["anlamli"]
    assert not backward["anlamli"]
    assert forward["p_degeri"] < backward["p_degeri"]


def test_the_lag_that_was_built_in_is_the_strongest_correlation():
    """Granger answers "is this chance?"; the correlation table answers "how
    strong, and when?" -- and it lands on the lag the pair was made with."""
    _result, correlation, _summary = lead_lag(
        _leading_pair(), "fiyat", "kredi", max_lag=6
    )

    peak = correlation.loc[correlation["korelasyon"].abs().idxmax()]
    assert int(peak["gecikme"]) == LAG  # positive: `fiyat` runs ahead
    assert peak["korelasyon"] > 0.5


def test_two_unrelated_walks_come_back_as_nothing_found():
    """The failure this whole module guards against: two series that only
    trend together testing as related. Differencing is what prevents it."""
    result, _correlation, summary = lead_lag(_independent(), "a", "b", max_lag=6)

    assert not result["anlamli"].any()
    assert "anlamli onculluk yok" in summary


def test_trying_six_lags_tightens_the_threshold():
    """Keeping the best of six lags turns a one-in-twenty false alarm into
    nearly one in four, so the threshold is divided by the number tried and
    the summary says so."""
    _result, _correlation, summary = lead_lag(
        _leading_pair(), "fiyat", "kredi", max_lag=6
    )

    assert f"{ALPHA / 6:.4f}" in summary
    assert "6 gecikme denendi" in summary


# --- how the input is prepared -----------------------------------------------


def test_a_trending_series_is_differenced_and_the_summary_says_which():
    result, _correlation, summary = lead_lag(
        _leading_pair(), "fiyat", "kredi", max_lag=4, log_x=True, log_y=True
    )

    assert set(result["donusum"]) == {"log-fark"}
    assert "duraganlastirildi" in summary


def test_a_stationary_series_is_left_alone():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "period": _periods(80),
            "a": rng.normal(size=80),  # white noise: already stationary
            "b": rng.normal(size=80),
        }
    )

    result, _correlation, _summary = lead_lag(frame, "a", "b", max_lag=4)

    assert set(result["donusum"]) == {"yok"}


def test_periods_missing_in_either_series_are_dropped_not_filled():
    frame = _leading_pair()
    frame.loc[10:14, "kredi"] = np.nan

    result, _correlation, summary = lead_lag(frame, "fiyat", "kredi", max_lag=4)

    # Three periods are already empty where `kredi` was shifted into being,
    # and five more were blanked here.
    assert "8 donem eksik veri yuzunden disarida kaldi" in summary
    assert result["gozlem"].iloc[0] <= len(frame) - 8


def test_a_short_series_is_refused_rather_than_answered():
    """Below three observations per lag the test still returns a number, and
    that number is noise."""
    with pytest.raises(NotEnoughData, match="az"):
        lead_lag(_leading_pair(n=12), "fiyat", "kredi", max_lag=6)


def test_a_series_too_short_for_six_lags_is_tested_with_fewer():
    """Shortened rather than refused: the answer is still worth having, and
    the summary says how many lags it stood on."""
    _result, correlation, summary = lead_lag(
        _leading_pair(n=28), "fiyat", "kredi", max_lag=6
    )

    assert "6 gecikme denendi" not in summary
    assert correlation["gecikme"].max() < 6


def test_two_series_that_never_overlap_say_so():
    frame = pd.DataFrame(
        {
            "period": _periods(10),
            "a": [1.0] * 5 + [np.nan] * 5,
            "b": [np.nan] * 5 + [1.0] * 5,
        }
    )

    with pytest.raises(NotEnoughData, match="ortak donem yok"):
        lead_lag(frame, "a", "b")


# --- the correlation table ---------------------------------------------------


def test_a_positive_lag_means_the_first_series_runs_ahead():
    """The sign convention, pinned: x[t-lag] against y[t]."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = x.shift(2)  # y repeats x two periods later

    table = lagged_correlation(x, y, max_lag=3).set_index("gecikme")

    assert table.loc[2, "korelasyon"] == pytest.approx(1.0)
    assert table.loc[2, "gozlem"] == 4


def test_a_correlation_from_too_few_points_is_left_empty():
    x = pd.Series([1.0, 2.0, 3.0])

    table = lagged_correlation(x, x, max_lag=2).set_index("gecikme")

    assert pd.isna(table.loc[2, "korelasyon"])  # one overlapping point


# --- what the summary is allowed to say --------------------------------------


def test_the_summary_always_says_this_is_not_proof_of_cause():
    """The sentence the answer is obliged to carry (prompts/answer.md #7)."""
    _result, _correlation, summary = lead_lag(
        _leading_pair(), "fiyat", "kredi", max_lag=4
    )

    assert "nedensellik kaniti degildir" in summary
    assert "ucuncu etkenler" in summary
