"""Tests for lakehouse_agent.analysis.change. No network access --
synthetic series with a known break location."""

import math

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.analysis import change


def _periods(n: int):
    return pd.date_range("2020-01-01", periods=n, freq="MS").date


def test_detect_changes_finds_level_break_within_one_period():
    n = 60
    break_idx = 30
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [
            10.0 + rng.normal(scale=0.3, size=break_idx),
            20.0 + rng.normal(scale=0.3, size=n - break_idx),
        ]
    )
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, summary = change.detect_changes(series)

    assert len(result) >= 1
    detected = result.iloc[0]["tarih"]
    expected = series["period"].iloc[break_idx]
    assert abs((detected - expected).days) <= 31
    assert result.iloc[0]["ortalama_öncesi"] < result.iloc[0]["ortalama_sonrası"]
    assert "kırılma" in summary


def test_seviye_degisim_and_sicrama_differ_for_a_slow_drift_then_jump():
    """The KFE bug: a series that drifts steadily before AND after a level
    shift reports a large ortalama-to-ortalama seviye_degisim_yuzde (the two
    section means are far apart mostly because of the drift, not because of
    a single-period spike), while sicrama_yuzde -- the value at the break
    vs. the value right before it -- reports the much smaller real jump.
    Measured (KFE): mean-to-mean %230.3 vs. a small actual one-period move.
    """
    n = 60
    break_idx = 30
    before = 10.0 + np.arange(break_idx) * 0.3
    after = 20.0 + np.arange(n - break_idx) * 0.3
    values = np.concatenate([before, after])
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, summary = change.detect_changes(series, min_size=10)

    assert len(result) >= 1
    # The headline row is whichever breakpoint has the largest seviye_degisim_yuzde
    # (that's what the summary picks too) -- its sicrama_yuzde must be much smaller.
    worst = result.iloc[result["seviye_degisim_yuzde"].abs().idxmax()]
    assert abs(worst["seviye_degisim_yuzde"]) > 3 * abs(worst["sicrama_yuzde"])
    assert "seviye_degisim_yuzde" not in summary  # summary is prose, not a column dump
    assert "seviye" in summary and "sıçrama" in summary
    assert f"%{worst['seviye_degisim_yuzde']:.1f}" in summary
    assert f"%{worst['sicrama_yuzde']:.1f}" in summary


def test_sicrama_yuzde_is_pct_change_between_break_and_prior_period():
    n = 40
    break_idx = 20
    values = np.concatenate([np.full(break_idx, 10.0), np.full(n - break_idx, 15.0)])
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, _ = change.detect_changes(series, min_size=10)

    assert len(result) >= 1
    row = result.iloc[0]
    # A clean step: value right at the break (15.0) vs. right before (10.0).
    assert row["sicrama_yuzde"] == pytest.approx(50.0)
    assert row["seviye_degisim_yuzde"] == pytest.approx(50.0)


def test_detect_changes_finds_slope_break_within_one_period():
    # A pure slope-only kink (same level, just a new slope) is not reliably
    # localizable by ruptures' rbf cost -- verified empirically (the exact
    # global optimum itself lands several periods off for that shape, it's
    # not a tuning issue). A combined level+slope break (e.g. a policy shock:
    # a jump *and* a new trend) is both realistic and cleanly detected.
    n = 60
    break_idx = 30
    rng = np.random.default_rng(0)
    t = np.arange(n)
    values = np.where(
        t < break_idx, 0.05 * t, 0.05 * break_idx + 8.0 + 0.3 * (t - break_idx)
    )
    values = values + rng.normal(scale=0.3, size=n)
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, _ = change.detect_changes(series)

    assert len(result) >= 1
    detected = result.iloc[0]["tarih"]
    expected = series["period"].iloc[break_idx]
    assert abs((detected - expected).days) <= 31
    assert result.iloc[0]["eğim_sonrası"] > result.iloc[0]["eğim_öncesi"]


def test_detect_changes_white_noise_finds_no_break():
    n = 60
    rng = np.random.default_rng(2)
    values = rng.normal(loc=0.0, scale=1.0, size=n)
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, summary = change.detect_changes(series)

    assert result.empty
    assert "tespit edilmedi" in summary


def test_detect_changes_drops_nan_rows_before_computing():
    n = 60
    break_idx = 30
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [
            10.0 + rng.normal(scale=0.3, size=break_idx),
            20.0 + rng.normal(scale=0.3, size=n - break_idx),
        ]
    )
    series = pd.DataFrame({"period": _periods(n), "value": values})
    series.loc[5, "value"] = float("nan")

    result, _ = change.detect_changes(series)

    assert len(result) >= 1


def test_detect_changes_all_zero_before_mean_falls_back_without_crashing():
    """Regression: kırılma öncesi bölüm ortalaması tam 0 ise
    seviye_degisim_yuzde NaN olur (0'a bölme tanımsız); tüm kırılmalarda bu
    doğruysa eski kod .abs().idxmax()'ı all-NaN kolonda çağırıp ValueError
    fırlatıyordu -- B'nin BDDK'da 0 tabanlı bir seride ölçtüğü çökme."""
    n = 40
    break_idx = 20
    values = np.concatenate([np.zeros(break_idx), np.full(n - break_idx, 100.0)])
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, summary = change.detect_changes(series, min_size=10)

    assert len(result) >= 1
    assert math.isnan(result.iloc[0]["seviye_degisim_yuzde"])
    assert isinstance(summary, str) and summary


def test_detect_changes_constant_series_returns_no_breakpoints_not_spurious_ones():
    """B'nin ölçtüğü: sabit seride std==0 -> pen=log(n)*0=0 -> PELT 9 sıfır-
    maliyetli kırılma üretiyordu, hepsi "%0.0" değişim. Sabit seride
    kırılacak bir şey yok -- boş sonuç dönmeli."""
    n = 60
    values = np.full(n, 42.0)
    series = pd.DataFrame({"period": _periods(n), "value": values})

    result, summary = change.detect_changes(series)

    assert result.empty
    assert "sabit" in summary


def test_detect_changes_rejects_unknown_method():
    series = pd.DataFrame({"period": _periods(10), "value": np.arange(10.0)})
    try:
        change.detect_changes(series, method="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
