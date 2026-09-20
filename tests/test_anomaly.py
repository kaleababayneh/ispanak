"""Tests for lakehouse_agent.analysis.anomaly. No network access --
synthetic series with known injected outliers."""

import numpy as np
import pandas as pd

from lakehouse_agent.analysis import anomaly


def _monthly_periods(n: int):
    return pd.date_range("2020-01-01", periods=n, freq="MS").date


def _seasonal_series(n: int = 120, *, outlier_indices=(), outlier_delta: float = 4.0):
    # n=120 (10 years) and delta=4.0 are not arbitrary: STL's seasonal LOESS
    # smoothing leaks a fraction of any outlier into the same calendar month
    # one cycle away (measured) -- too short a series or too large a delta
    # pushes that leak's own z past the 3.5 threshold too. This combination
    # was verified to isolate the injected points cleanly (see PR notes).
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 12)
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=0.5, size=n)
    value = trend + seasonal + noise
    for idx in outlier_indices:
        value[idx] += outlier_delta
    return pd.DataFrame({"period": _monthly_periods(n), "value": value})


def test_detect_anomalies_stl_mad_finds_injected_outliers_with_few_false_positives():
    outlier_idx = [15, 60, 105]
    series = _seasonal_series(120, outlier_indices=outlier_idx)

    flagged, summary = anomaly.detect_anomalies(series)

    flagged_periods = set(flagged["dönem"])
    expected_periods = {series["period"].iloc[i] for i in outlier_idx}
    assert expected_periods.issubset(flagged_periods)
    assert len(flagged) - len(expected_periods) <= 1  # at most 1 false positive
    assert "anomali" in summary


def test_detect_anomalies_direction_matches_sign_of_deviation():
    series = _seasonal_series(outlier_indices=[60])

    flagged, _ = anomaly.detect_anomalies(series)

    row = flagged[flagged["dönem"] == series["period"].iloc[60]].iloc[0]
    assert row["yön"] == "üstünde"


def test_detect_anomalies_no_outliers_flags_nothing():
    series = _seasonal_series()

    flagged, summary = anomaly.detect_anomalies(series)

    assert flagged.empty
    assert "tespit edilmedi" in summary


def test_detect_anomalies_short_series_uses_iqr_fallback():
    periods = _monthly_periods(10)
    values = [10.0] * 9 + [50.0]
    series = pd.DataFrame({"period": periods, "value": values})

    flagged, _ = anomaly.detect_anomalies(series)

    assert len(flagged) == 1
    assert flagged.iloc[0]["dönem"] == periods[-1]
    assert flagged.iloc[0]["yön"] == "üstünde"


def test_detect_anomalies_drops_nan_rows_before_computing():
    series = _seasonal_series(outlier_indices=[60])
    series.loc[5, "value"] = float("nan")

    flagged, _ = anomaly.detect_anomalies(series)

    assert series["period"].iloc[5] not in set(flagged["dönem"])


def test_detect_anomalies_rejects_unknown_method():
    series = _seasonal_series()
    try:
        anomaly.detect_anomalies(series, method="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_detect_anomalies_calendar_gap_falls_back_to_iqr_no_false_positives():
    """Regression: dropping rows before STL (instead of gridding onto the
    calendar) compacts the series, so every point after a gap lands in the
    wrong seasonal phase. Measured (n=36, seasonal amplitude 5, noise 0.1,
    seed 1, months 15:21 removed): STL on the compacted series flags 4
    false anomalies (maxabsz 7.41). Gridding detects the gap and falls back
    to IQR on the genuinely observed points instead -- 0 anomalies."""
    n = 36
    t = np.arange(n)
    values = 5 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(
        scale=0.1, size=n
    )
    series = pd.DataFrame({"period": _monthly_periods(n), "value": values})
    gapped = pd.concat([series.iloc[:15], series.iloc[21:]]).reset_index(drop=True)

    flagged, summary = anomaly.detect_anomalies(gapped)

    assert flagged.empty
    assert anomaly.IQR_FALLBACK_NOTE in summary


def test_detect_anomalies_iqr_path_reports_fence_distance_not_a_degenerate_mad_z():
    """B'nin ölçtüğü: n=12, tek aykırı 100 -> işaretlendi ama z=0.00 (MAD
    çoğunluk aynı değerde olduğu için sıfıra düşüyor; ham IQR da aynı
    örnekte 0'a dejenere oluyor). Artık z yerine mutlak çit mesafesi
    raporlanıyor -- işaretlenen noktada asla 0 olamaz."""
    n = 12
    values = [10.0] * 11 + [100.0]
    periods = pd.date_range("2020-01-01", periods=n, freq="MS").date
    series = pd.DataFrame({"period": periods, "value": values})

    flagged, _ = anomaly.detect_anomalies(series)

    assert len(flagged) == 1
    assert flagged.iloc[0]["z"] > 0


def _gap_series(n: int = 72, *, outlier_indices=(), outlier_delta: float = 30.0):
    """A strongly seasonal monthly series, so losing the decomposition is
    visible in the result rather than only in the method name."""
    t = np.arange(n)
    values = (
        100
        + t * 0.3
        + 25 * np.sin(2 * np.pi * t / 12)
        + np.random.default_rng(7).normal(scale=0.4, size=n)
    )
    for index in outlier_indices:
        values[index] += outlier_delta
    return pd.DataFrame({"period": _monthly_periods(n), "value": values})


def test_a_single_missing_month_does_not_blind_the_detector():
    """Measured before this path existed: the complete series finds all three
    injected anomalies, and removing ONE month (1.4%) made the IQR fallback
    find ZERO. Silence reads as a clean bill of health, which is worse than
    the false alarms the gap check was added to prevent."""
    injected = (17, 40, 61)
    series = _gap_series(outlier_indices=injected)
    expected_periods = {series["period"].iloc[i] for i in injected}

    gapped = series.drop(index=[30]).reset_index(drop=True)
    flagged, summary = anomaly.detect_anomalies(gapped)

    assert expected_periods <= set(flagged["dönem"])
    assert anomaly.IQR_FALLBACK_NOTE not in summary


def test_an_interpolated_period_is_never_reported():
    """The interpolation feeds the decomposition and nothing else. A period
    that was never observed must not appear as a finding -- that would be a
    fabricated observation reaching the caller, which is the line CLAUDE.md #2
    actually draws."""
    series = _gap_series(outlier_indices=(17, 40, 61))
    dropped_period = series["period"].iloc[30]
    gapped = series.drop(index=[30]).reset_index(drop=True)

    flagged, _ = anomaly.detect_anomalies(gapped)

    assert dropped_period not in set(flagged["dönem"])
    assert set(flagged["dönem"]) <= set(gapped["period"])


def test_too_many_gaps_still_fall_back_rather_than_interpolate():
    """Past the ceiling the seasonal shape would rest mostly on interpolation,
    so the honest answer is the weaker method plus a note saying so."""
    series = _gap_series(outlier_indices=(17, 40, 61))
    gapped = series.drop(index=range(20, 32)).reset_index(drop=True)

    _, summary = anomaly.detect_anomalies(gapped)

    assert anomaly.IQR_FALLBACK_NOTE in summary


def test_the_summary_says_interpolation_was_used():
    """A human reading the answer has to be able to tell that part of the
    decomposition rested on filled-in values."""
    series = _gap_series(outlier_indices=(17,))
    gapped = series.drop(index=[30]).reset_index(drop=True)

    _, summary = anomaly.detect_anomalies(gapped)

    assert "ara değer" in summary
    assert "gözlenen" in summary
