"""Structural break (level/slope change) detection. Pure DataFrame ->
(DataFrame, str), no model, session, or tool knowledge (CLAUDE.md #7).
"""

import numpy as np
import pandas as pd
import ruptures as rpt


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def _pct_change(before_mean: float, after_mean: float) -> float:
    if before_mean == 0:
        return float("nan")
    return (after_mean - before_mean) / abs(before_mean) * 100


def detect_changes(
    series: pd.DataFrame,
    *,
    method: str = "pelt",
    model: str = "rbf",
    pen: float | None = None,
    min_size: int = 6,
) -> tuple[pd.DataFrame, str]:
    """Detect structural breaks (level/slope changes) in a "period"/"value"
    series using ruptures' PELT.

    The series is standardized (mean 0, std 1) before detection. If `pen`
    is not given, it defaults to log(n) * variance of the standardized
    signal -- a standard BIC-style penalty suggested in ruptures' own docs,
    not a bespoke heuristic. `jump=1` overrides ruptures' default of 5 (an
    exact search, not the coarser 5-period grid) so breakpoints can land
    within one period of the true change.

    Returns one row per breakpoint (tarih, ortalama_öncesi, ortalama_sonrası,
    eğim_öncesi, eğim_sonrası, yüzde_değişim), computed on the original
    (non-standardized) values, plus a two-sentence Turkish summary.
    """
    if method != "pelt":
        raise ValueError(f"Bilinmeyen method: {method!r}")

    clean = series.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
    n = len(clean)
    values = clean["value"].to_numpy()

    std = values.std()
    signal = (values - values.mean()) / std if std != 0 else values - values.mean()

    if pen is None:
        pen = float(np.log(n) * signal.var())

    # jump=1 (not ruptures' default of 5): the default only considers
    # breakpoint candidates on a 5-period grid, which can miss the true
    # break by several periods -- exact search is what "±1 period" needs.
    breakpoints = (
        rpt.Pelt(model=model, min_size=min_size, jump=1).fit(signal).predict(pen=pen)
    )
    interior = breakpoints[:-1]  # last entry is the sentinel n, not a real breakpoint

    boundaries = [0, *breakpoints]
    rows = []
    for i, b in enumerate(interior):
        before = values[boundaries[i] : boundaries[i + 1]]
        after = values[boundaries[i + 1] : boundaries[i + 2]]
        before_mean, after_mean = before.mean(), after.mean()
        rows.append(
            {
                "tarih": clean["period"].iloc[b],
                "ortalama_öncesi": before_mean,
                "ortalama_sonrası": after_mean,
                "eğim_öncesi": _slope(before),
                "eğim_sonrası": _slope(after),
                "yüzde_değişim": _pct_change(before_mean, after_mean),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "tarih",
            "ortalama_öncesi",
            "ortalama_sonrası",
            "eğim_öncesi",
            "eğim_sonrası",
            "yüzde_değişim",
        ],
    )

    if result.empty:
        summary = "Seride kırılma tespit edilmedi. Seviye ve eğim boyunca istikrarlı."
    else:
        worst = result.iloc[result["yüzde_değişim"].abs().idxmax()]
        summary = (
            f"Seride {len(result)} kırılma noktası tespit edildi. "
            f"En büyük değişim {worst['tarih']} döneminde, %{worst['yüzde_değişim']:.1f} ile gerçekleşti."
        )

    return result, summary
