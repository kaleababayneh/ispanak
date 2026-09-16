"""Point-anomaly detection. Pure DataFrame -> (DataFrame, str), no model,
session, or tool knowledge (CLAUDE.md #7).

Why STL, why MAD (for the jury, verbatim): STL is used because seasonality
otherwise gets mistaken for a level shift and produces false alarms; MAD is
used because the mean and standard deviation are themselves distorted by
outliers, while the median and MAD are not.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

_MIN_PERIODS_FOR_STL = 24  # below this, STL has too few cycles to fit


def _mad_z(residual: pd.Series) -> pd.Series:
    median = residual.median()
    mad = (residual - median).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=residual.index)
    return (residual - median) / (1.4826 * mad)


def _direction(observed: pd.Series, expected: pd.Series) -> pd.Series:
    return np.where(observed >= expected, "üstünde", "altında")


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
    a two-sentence Turkish summary.
    """
    if method != "stl_mad":
        raise ValueError(f"Bilinmeyen method: {method!r}")

    clean = series.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)

    if len(clean) < _MIN_PERIODS_FOR_STL:
        expected = pd.Series(clean["value"].median(), index=clean.index)
        z = _mad_z(clean["value"])
        q1, q3 = clean["value"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_anomaly = (clean["value"] < lower) | (clean["value"] > upper)
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

    return flagged, summary
