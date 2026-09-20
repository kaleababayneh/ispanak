"""Structural break (level/slope change) detection. Pure DataFrame ->
(DataFrame, str), no model, session, or tool knowledge (CLAUDE.md #7).
"""

import numpy as np
import pandas as pd
import ruptures as rpt

_MIN_SIGNAL_VAR = 1e-9  # below this, the series is constant -- pen would be ~0
_MIN_PEN = 1e-6


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def _pct_change(before: float, after: float) -> float:
    if before == 0:
        return float("nan")
    return (after - before) / abs(before) * 100


def _instant_jump_pct(values: np.ndarray, break_idx: int) -> float:
    """Percent change between the value AT the breakpoint and the value the
    period immediately before it -- the actual single-period jump.

    Distinct on purpose from `seviye_degisim_yuzde` (before/after MEAN
    comparison): a slow drift either side of the break can make the two
    section means differ by a lot with almost no jump at the break itself.
    Measured bug: KFE reported "%230.3 değişim" from mean-to-mean alone,
    which reads as an implausible single-period spike ("endeks bir ayda
    %230 mü arttı?") -- it wasn't one. Reporting both numbers, labeled,
    resolves the ambiguity instead of picking one.
    """
    return _pct_change(values[break_idx - 1], values[break_idx])


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
    eğim_öncesi, eğim_sonrası, seviye_degisim_yuzde, sicrama_yuzde), computed
    on the original (non-standardized) values, plus a two-sentence Turkish
    summary that names both percentages separately -- see
    `seviye_degisim_yuzde` vs `sicrama_yuzde` below, they answer different
    questions and reporting only one is misleading.
    """
    if method != "pelt":
        raise ValueError(f"Bilinmeyen method: {method!r}")

    clean = series.dropna(subset=["value"]).sort_values("period").reset_index(drop=True)
    n = len(clean)
    values = clean["value"].to_numpy()

    std = values.std()
    signal = (values - values.mean()) / std if std != 0 else values - values.mean()

    if signal.var() < _MIN_SIGNAL_VAR:
        # Measured: a perfectly constant series (std==0) drove pen to
        # exactly 0 (log(n)*0), and PELT inserted 9 zero-cost breakpoints,
        # all reporting "%0.0" change -- nothing to break in a flat series.
        empty = pd.DataFrame(
            columns=[
                "tarih",
                "ortalama_öncesi",
                "ortalama_sonrası",
                "eğim_öncesi",
                "eğim_sonrası",
                "seviye_degisim_yuzde",
                "sicrama_yuzde",
            ]
        )
        return (
            empty,
            "Seri sabit (varyans yok); kırılma tespiti anlamsız, hiçbir kırılma raporlanmadı.",
        )

    if pen is None:
        pen = float(np.log(n) * signal.var())
    pen = max(pen, _MIN_PEN)

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
                "seviye_degisim_yuzde": _pct_change(before_mean, after_mean),
                "sicrama_yuzde": _instant_jump_pct(values, b),
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
            "seviye_degisim_yuzde",
            "sicrama_yuzde",
        ],
    )

    if result.empty:
        summary = "Seride kırılma tespit edilmedi. Seviye ve eğim boyunca istikrarlı."
    else:
        if result["seviye_degisim_yuzde"].notna().any():
            worst = result.iloc[result["seviye_degisim_yuzde"].abs().idxmax()]
        elif result["sicrama_yuzde"].notna().any():
            worst = result.iloc[result["sicrama_yuzde"].abs().idxmax()]
        else:
            worst = result.iloc[0]
        summary = (
            f"Seride {len(result)} kırılma noktası tespit edildi. "
            f"En büyük seviye değişimi {worst['tarih']} döneminde: "
            f"öncesi/sonrası ortalama %{worst['seviye_degisim_yuzde']:.1f}, "
            f"kırılma anındaki tek dönem sıçraması %{worst['sicrama_yuzde']:.1f}."
        )

    return result, summary
