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
