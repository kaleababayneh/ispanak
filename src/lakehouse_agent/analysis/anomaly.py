"""Point-anomaly detection. Pure DataFrame -> (DataFrame, str), no model,
session, or tool knowledge (CLAUDE.md #7).

Why STL, why MAD (for the jury, verbatim): STL is used because seasonality
otherwise gets mistaken for a level shift and produces false alarms; MAD is
used because the mean and standard deviation are themselves distorted by
outliers, while the median and MAD are not.
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)

_MIN_PERIODS_FOR_STL = 24  # below this, STL has too few cycles to fit

# STL's seasonal period (in observations) implies the calendar step between
# periods -- used to reindex onto the full calendar (asfreq) before fitting,
# so a gap doesn't silently shift every later observation's seasonal phase.
# Measured: n=36, seasonal amplitude 5, noise 0.1, 6 months removed from the
# middle -> STL on the compacted series flagged 4 false anomalies (maxabsz
# 7.41); the same points via the IQR fallback below flag 0.
_PERIOD_TO_FREQ_ALIAS = {12: "MS", 52: "W-MON", 4: "QS", 7: "D", 1: "YS"}

# STL does not raise on NaN input -- it silently returns an all-NaN trend/
# seasonal/resid instead (measured: 2 NaN out of 48 points in -> 100% NaN
# out, no exception). So the calendar grid has to be complete before STL sees
# it, and the gaps are interpolated for that one purpose (see
# `_stl_on_calendar`). Past this share of missing periods the seasonal shape
# would rest mostly on interpolation, so the IQR fallback is used instead.
#
# Why not refuse STL at the first gap: measured on a 72-month series with a
# seasonal amplitude of 25 and three injected anomalies -- complete series,
# STL finds all three; remove ONE month (1.4%) and the IQR fallback finds
# ZERO. Silence reads as a clean bill of health, which is worse than the
# false alarms the gap check was added to prevent.
_MAX_MISSING_PCT_FOR_STL = 0.10

# Shared with tools/analysis.py: lets it tell -- from `summary` alone,
# without re-implementing the calendar-gap check -- that this call fell
# back to IQR so it can report the matching formula text (plan review #8).
IQR_FALLBACK_NOTE = "IQR (Tukey) kuralına düşüldü"


def _mad_z(residual: pd.Series) -> pd.Series:
    median = residual.median()
    mad = (residual - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=residual.index)
    return (residual - median) / (1.4826 * mad)


def _direction(observed: pd.Series, expected: pd.Series) -> pd.Series:
    return np.where(observed >= expected, "üstünde", "altında")


def _stl_on_calendar(
    gridded: pd.Series, observed_index: pd.DatetimeIndex, period: int
) -> tuple[pd.Series, pd.Series] | None:
    """Fit STL on the full calendar, then keep only the observed periods.

    The gaps are interpolated so STL sees a complete grid and keeps every
    later observation's seasonal phase. That interpolation is an input to the
    decomposition and nothing else: the fitted value and residual are read
    back only at periods that were actually observed, so no invented number is
    ever scored, flagged or reported. CLAUDE.md #2 is about observations that
    reach the caller, and none of these do.

    Returns None when interpolation cannot complete the grid (a gap at either
    end, which `limit_area="inside"` deliberately refuses to extrapolate).
    """
    filled = gridded.interpolate(method="time", limit_area="inside")
    if filled.isna().any():
        return None

    fit = STL(filled.to_numpy(), period=period).fit()
    fitted = pd.Series(fit.trend + fit.seasonal, index=filled.index)
    resid = pd.Series(fit.resid, index=filled.index)
    return fitted.reindex(observed_index), resid.reindex(observed_index)


def _iqr_fallback(clean: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Tukey 1.5*IQR rule on raw values -- used both for series too short
    for STL and for series with a calendar gap STL can't be trusted with.

    Reports a fence-distance score as "z", NOT a MAD-z score: MAD (and,
    equally, the raw IQR itself) degenerates to exactly 0 whenever at least
    half the values are identical -- exactly the shape this fallback is
    for (a short series with one outlier). Measured: n=12, 11x10.0 + one
    100.0 -> both MAD-z and a normalized IQR-distance report 0.00 for the
    flagged point. Fence distance is an un-normalized, absolute distance
    past the nearest fence (0 inside the fence), so it is nonzero exactly
    when a point is flagged and never divides by a degenerate spread.
    """
    expected = pd.Series(clean["value"].median(), index=clean.index)
    q1, q3 = clean["value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_anomaly = (clean["value"] < lower) | (clean["value"] > upper)
    above = (clean["value"] - upper).clip(lower=0)
    below = (lower - clean["value"]).clip(lower=0)
    z = above.combine(below, max)
    return expected, z, is_anomaly


def detect_anomalies(
    series: pd.DataFrame,
    *,
    method: str = "stl_mad",
    threshold: float = 3.5,
    period: int = 12,
) -> tuple[pd.DataFrame, str]:
    """Flag point anomalies in a "period"/"value" series.

    method="stl_mad" (the only supported method): STL-decompose (period=12
    for monthly, 52 for weekly), then a robust z-score
    z = (resid - median) / (1.4826 * MAD) on the residual; |z| > threshold
    is flagged. Series shorter than 24 periods fall back to a Tukey
    1.5*IQR rule on the raw values (too short to fit a seasonal cycle) --
    that fallback's own MAD-z is reported for schema consistency but the
    flag itself comes from the IQR fence, not `threshold`.

    Returns only the flagged rows (dönem, gözlenen, beklenen, z, yön) plus
    a two-sentence Turkish summary. If the series has a calendar gap STL
    can't be trusted with (see `_iqr_fallback`), the summary is prefixed
    with a note naming the missing-period percentage.
    """
    if method != "stl_mad":
        raise ValueError(f"Bilinmeyen method: {method!r}")

    clean = series.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
    fallback_note = None
    gap_note = None

    if len(clean) < _MIN_PERIODS_FOR_STL:
        expected, z, is_anomaly = _iqr_fallback(clean)
    else:
        freq_alias = _PERIOD_TO_FREQ_ALIAS.get(period)
        missing_pct = 0.0
        if freq_alias is not None:
            idx = pd.to_datetime(clean["period"])
            gridded = pd.Series(clean["value"].to_numpy(), index=idx).asfreq(freq_alias)
            missing_pct = float(gridded.isna().mean())

        on_calendar = None
        if freq_alias is not None and 0 < missing_pct <= _MAX_MISSING_PCT_FOR_STL:
            on_calendar = _stl_on_calendar(gridded, idx, period)

        if freq_alias is not None and missing_pct > _MAX_MISSING_PCT_FOR_STL:
            logger.warning(
                "detect_anomalies: takvimde %%%.1f eksik dönem -- mevsimsel şekil "
                "ağırlıkla ara değerlere dayanırdı, %s.",
                missing_pct * 100,
                IQR_FALLBACK_NOTE,
            )
            fallback_note = (
                f"Takvimde %{missing_pct * 100:.1f} eksik dönem var; "
                f"{IQR_FALLBACK_NOTE}."
            )
            expected, z, is_anomaly = _iqr_fallback(clean)
        elif on_calendar is not None:
            fitted, resid_observed = on_calendar
            expected = pd.Series(fitted.to_numpy(), index=clean.index)
            z = _mad_z(pd.Series(resid_observed.to_numpy(), index=clean.index))
            is_anomaly = z.abs() > threshold
            gap_note = (
                f"Takvimde %{missing_pct * 100:.1f} eksik dönem var; ayrıştırma "
                "için ara değer kullanıldı, puanlama yalnızca gözlenen "
                "dönemlerde yapıldı."
            )
        elif freq_alias is not None and missing_pct > 0:
            # The grid could not be completed -- a gap at one end, which
            # `limit_area="inside"` deliberately refuses to extrapolate.
            logger.warning(
                "detect_anomalies: takvim boşluğu serinin ucunda, %s.",
                IQR_FALLBACK_NOTE,
            )
            fallback_note = (
                f"Takvimde %{missing_pct * 100:.1f} eksik dönem var ve boşluk "
                f"serinin ucunda; {IQR_FALLBACK_NOTE}."
            )
            expected, z, is_anomaly = _iqr_fallback(clean)
        else:
            result = STL(clean["value"].to_numpy(), period=period).fit()
            resid = pd.Series(result.resid, index=clean.index)
            expected = pd.Series(result.trend + result.seasonal, index=clean.index)
            z = _mad_z(resid)
            is_anomaly = z.abs() > threshold

    flagged = pd.DataFrame(
        {
            "dönem": clean["period"],
            "gözlenen": clean["value"],
            "beklenen": expected,
            "z": z,
            "yön": _direction(clean["value"], expected),
        }
    )[is_anomaly].reset_index(drop=True)

    if flagged.empty:
        summary = (
            "Seride anomali tespit edilmedi. Tüm gözlemler beklenen aralık içinde."
        )
    else:
        worst = flagged.iloc[flagged["z"].abs().idxmax()]
        summary = (
            f"Seride {len(flagged)} anomali tespit edildi. "
            f"En büyük sapma {worst['dönem']} döneminde gözlendi (z={worst['z']:.2f})."
        )

    if fallback_note:
        summary = f"{fallback_note} {summary}"
    elif gap_note:
        summary = f"{gap_note} {summary}"

    return flagged, summary
