"""Lead-lag analysis: does one series run ahead of another?

Two measurements over the same pair, because they answer different halves of
the question a person actually asks:

* **Granger** -- "is this more than chance?" It asks whether the past of X
  improves a forecast of Y beyond Y's own past. The name is unfortunate: it
  is a test about *prediction*, not about causes. A third factor driving both
  series produces the same result, and so does a market that prices in what
  it expects, so the wording it comes back with says that outright.
* **Lagged correlation** -- "how strong, and at which lag?" A p-value says
  nothing about size: with enough observations a negligible relationship is
  still "significant". The correlation table is the readable half.

Both need stationary input. Two series that merely trend upward together
test as related whatever they are about, so a series the ADF test cannot
call stationary is differenced first (in logs where the tool says the
quantity is a level, which turns a difference into a growth rate), and the
transform applied travels with the result.

Pure: DataFrames in, (DataFrame, DataFrame, str) out. No model, no session,
no tools (CLAUDE.md #7). The entry point is `lead_lag`, not `test_causality`:
the tool the model calls is named for the question it answers, but a module
function whose name starts with "test_" is collected by pytest as a test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

#: The usual 0.05, before the correction for trying several lags.
ALPHA = 0.05
#: Observations a lag needs before its test means anything: three per lag
#: plus a floor. Below that the test still returns a number, and that number
#: is noise.
PER_LAG = 3
FLOOR = 10
#: ADF below this calls the series stationary.
ADF_ALPHA = 0.05

RESULT_COLUMNS = [
    "yon",
    "en_iyi_gecikme",
    "p_degeri",
    "anlamli",
    "gozlem",
    "donusum",
]


class NotEnoughData(ValueError):
    """Too few overlapping observations for the smallest useful test."""


def _stationary(values: pd.Series) -> bool:
    """ADF: does the series look stationary? A constant or near-constant
    series is called stationary rather than crashing the test."""
    if values.nunique() <= 2 or values.std(ddof=0) < 1e-12:
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(adfuller(values, autolag="AIC")[1] < ADF_ALPHA)
        except Exception:  # noqa: BLE001 - an undecidable series is differenced
            return False


def _prepare(values: pd.Series, *, log: bool) -> tuple[pd.Series, str, bool]:
    """(series, what was done, is it stationary now).

    Differenced until ADF says stationary, at most twice. One difference was
    applied and never re-checked, which is not the same as the "durağanlık
    şart" the contract claims: measured 23 Sep on 120 independent
    twice-integrated pairs, 117 were still non-stationary after the single
    difference and 27,5% of the pairs came back as "öncül". A series that is
    still non-stationary after two differences is returned as it is, with
    `False`, so the caller can say so instead of quietly testing it.
    """
    if _stationary(values):
        return values, "yok", True
    series = np.log(values) if log and (values > 0).all() else values
    label = "log-fark" if series is not values else "fark"
    for round_no in (1, 2):
        series = series.diff().dropna()
        if _stationary(series):
            return series, label if round_no == 1 else f"{label} x2", True
    return series, f"{label} x2", False


def lagged_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    """Correlation of `x` with `y` at each lag from -max_lag to +max_lag.

    A positive lag means `x` runs ahead: x[t-lag] against y[t]. Pearson on
    whatever overlaps at that shift, with the count kept so a correlation
    computed on six points is visible as such.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = x.shift(lag)
        pair = pd.concat([shifted, y], axis=1).dropna()
        rows.append(
            {
                "gecikme": lag,
                "korelasyon": (
                    round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 4)
                    if len(pair) > 2
                    else float("nan")
                ),
                "gozlem": len(pair),
            }
        )
    return pd.DataFrame(rows)


def _granger(effect: pd.Series, cause: pd.Series, max_lag: int) -> tuple[int, float]:
    """(lag with the smallest p, that p) for "does `cause` lead `effect`".

    statsmodels tests whether the second column Granger-causes the first, so
    the order here is [effect, cause] -- getting it backwards reverses every
    conclusion, which is why it is written out rather than inlined.
    """
    data = pd.concat([effect, cause], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = grangercausalitytests(data, maxlag=max_lag)
    best_lag, best_p = 1, 1.0
    for lag, (tests, _models) in results.items():
        p = float(tests["ssr_ftest"][1])
        if p < best_p:
            best_lag, best_p = int(lag), p
    return best_lag, best_p


def lead_lag(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    max_lag: int = 6,
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Does `x` lead `y`, or `y` lead `x`?

    `frame` holds "period" and both columns. Periods where either value is
    missing are dropped -- never filled (CLAUDE.md #2) -- and how many went
    is in the summary.

    Returns (Granger table, lagged-correlation table, summary). The Granger
    table has one row per direction: the lag with the smallest p-value, that
    p-value, and whether it clears the threshold. The threshold is
    `ALPHA / max_lag`: trying six lags and keeping the best of them turns a
    one-in-twenty false alarm into nearly one in four, so it is corrected
    (Bonferroni) rather than quoted as if a single test had been run.
    """
    pair = frame[["period", x, y]].dropna()
    dropped = len(frame) - len(pair)
    if pair.empty:
        raise NotEnoughData("Iki seride ortak donem yok.")

    pair = pair.sort_values("period")
    # Per series, not per call: a price index and a balance can sit in the
    # same table, and differencing the log of a level is a growth rate while
    # differencing an index is just its change.
    series_x, transform_x, stationary_x = _prepare(pair[x].astype(float), log=log_x)
    series_y, transform_y, stationary_y = _prepare(pair[y].astype(float), log=log_y)
    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    aligned.columns = [x, y]
    n = len(aligned)

    usable = min(max_lag, (n - FLOOR) // PER_LAG)
    if usable < 1:
        raise NotEnoughData(
            f"{n} gozlem tek gecikmelik bir test icin bile az "
            f"(en az {FLOOR + PER_LAG} gerekiyor)."
        )

    # Two directions x `usable` lags, and the summary presents one verdict
    # per pair, so the family is 2*usable. Dividing by the lags alone made
    # the sentence the model repeats untrue: measured at 22 observations,
    # 9% of independent pairs came back with a direction flagged while the
    # text said the bar was 5%.
    tests_run = 2 * usable
    threshold = ALPHA / tests_run
    rows = []
    for cause, effect in ((x, y), (y, x)):
        lag, p = _granger(aligned[effect], aligned[cause], usable)
        rows.append(
            {
                "yon": f"{cause} -> {effect}",
                "en_iyi_gecikme": lag,
                "p_degeri": round(p, 6),
                "anlamli": bool(p < threshold),
                "gozlem": n,
                # Both, named: the column used to carry only the cause's
                # transform, so a row could read "yok" while the effect had
                # been differenced -- a level regressed on a difference,
                # labelled as if nothing had happened.
                "donusum": f"{x}: {transform_x} | {y}: {transform_y}",
            }
        )
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    correlation = lagged_correlation(aligned[x], aligned[y], usable)
    summary = _summary(
        result, correlation, x, y, usable, threshold, dropped, transform_x, transform_y
    )
    if not (stationary_x and stationary_y):
        # Said, not hidden. Granger on a series with a unit root left in it
        # is the classic spurious-regression setup, and the p-value it
        # produces looks exactly like a real one.
        still = ", ".join(
            name for name, ok in ((x, stationary_x), (y, stationary_y)) if not ok
        )
        summary += (
            f" DIKKAT: {still} iki farktan sonra da duragan degil; bu testin "
            "p-degeri guvenilir degil, sonucu kullanma."
        )
    return result, correlation, summary


def _summary(  # noqa: PLR0913 - one sentence per fact, all of them needed
    result: pd.DataFrame,
    correlation: pd.DataFrame,
    x: str,
    y: str,
    max_lag: int,
    threshold: float,
    dropped: int,
    transform_x: str,
    transform_y: str,
) -> str:
    """What the model is allowed to repeat, in the words it should use."""
    parts = []
    for row in result.itertuples(index=False):
        if row.anlamli:
            parts.append(
                f"{row.yon}: Granger anlaminda oncul (gecikme {row.en_iyi_gecikme}, "
                f"p={row.p_degeri:.4f})."
            )
        else:
            parts.append(
                f"{row.yon}: anlamli onculluk yok (en iyi gecikme "
                f"{row.en_iyi_gecikme}, p={row.p_degeri:.4f})."
            )
    parts.append(
        f"{max_lag} gecikme, iki yon denendi; esik Bonferroni ile "
        f"{threshold:.4f} ({ALPHA} / {2 * max_lag})."
    )

    usable = correlation.dropna(subset=["korelasyon"])
    if not usable.empty:
        best = usable.loc[usable["korelasyon"].abs().idxmax()]
        lag = int(best["gecikme"])
        if lag > 0:
            lead = f"{x}, {y}'den {lag} donem once"
        elif lag < 0:
            lead = f"{y}, {x}'den {abs(lag)} donem once"
        else:
            lead = "ayni donem"
        parts.append(
            f"En guclu gecikmeli korelasyon: {lead}, r={best['korelasyon']:.2f} "
            f"({int(best['gozlem'])} gozlem)."
        )

    transforms = {transform_x, transform_y} - {"yok"}
    if transforms:
        parts.append(
            "Seriler duraganlastirildi (" + ", ".join(sorted(transforms)) + ")."
        )
    if dropped:
        parts.append(f"{dropped} donem eksik veri yuzunden disarida kaldi.")
    parts.append(
        "Bu nedensellik kaniti degildir: ucuncu etkenler (faiz, gelir, "
        "duzenleme) dislanmadi ve test yalnizca dogrusal iliskiyi olcer."
    )
    return " ".join(parts)
