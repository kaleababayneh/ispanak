"""Pure DataFrame -> DataFrame transforms. No model, session, or tool
knowledge (CLAUDE.md #7) -- every function here takes plain pandas
DataFrames with a "period"/"value" shape and returns the same shape, plus
the formula it applied as a string.

NaN is never filled/interpolated (CLAUDE.md #2): a period missing from the
input stays missing, and an aggregation window with no real observations
stays NaN rather than becoming 0.
"""

from collections.abc import Mapping

import pandas as pd

_KIND_TO_AGG = {"stock": "last", "flow": "sum", "rate": "mean", "index": "mean"}
_KIND_TO_FORMULA = {
    "stock": "value[period] = last(value[alt_dönem])",
    "flow": "value[period] = sum(value[alt_dönem])",
    "rate": "value[period] = mean(value[alt_dönem])",
    "index": "value[period] = mean(value[alt_dönem])",
}


def to_freq(df: pd.DataFrame, kind: str, freq: str) -> tuple[pd.DataFrame, str]:
    """Resample a "period"/"value" series to a lower frequency, aggregating
    per `kind` (stock->last, flow->sum, rate->avg, index->avg).

    `freq` is a pandas offset alias for the target period's first day
    (e.g. "MS", "QS", "YS", "W-MON") -- matching the gold contract's "period
    = first day of period" convention.
    """
    if kind not in _KIND_TO_AGG:
        raise ValueError(f"Bilinmeyen kind: {kind!r}")

    working = df.copy()
    working["period"] = pd.to_datetime(working["period"])
    grouped = working.set_index("period")["value"].resample(freq)

    if _KIND_TO_AGG[kind] == "sum":
        resampled = grouped.sum(min_count=1)  # all-NaN window -> NaN, not 0
    else:
        resampled = getattr(grouped, _KIND_TO_AGG[kind])()

    result = resampled.reset_index()
    result["period"] = result["period"].dt.date
    return result, _KIND_TO_FORMULA[kind]


def deflate(
    nominal: pd.DataFrame, price_index: pd.DataFrame, *, base: float = 100.0
) -> tuple[pd.DataFrame, str]:
    """Deflate a nominal "period"/"value" series by a price index series
    (also "period"/"value"), base=100. Every period present in `nominal` is
    kept; a period with no matching index value gets NaN, not a dropped row.
    """
    merged = nominal.merge(
        price_index, on="period", how="left", suffixes=("", "_index")
    )
    merged["value"] = merged["value"] / (merged["value_index"] / base)
    result = merged[["period", "value"]]
    formula = f"value_real = value_nominal / (index / {base})"
    return result, formula


def rebase(
    df: pd.DataFrame, base_period, *, base_value: float = 100.0
) -> tuple[pd.DataFrame, str]:
    """Rescale an index series so `base_period` equals `base_value`."""
    base_rows = df.loc[df["period"] == base_period, "value"]
    if base_rows.empty:
        raise ValueError(f"base_period {base_period} seride yok")
    base_raw = base_rows.iloc[0]

    result = df.copy()
    result["value"] = df["value"] / base_raw * base_value
    formula = f"value_rebased = value / value[{base_period}] * {base_value}"
    return result, formula


def pct_change(df: pd.DataFrame, periods: int = 1) -> tuple[pd.DataFrame, str]:
    """Percent change vs. `periods` observations back (not necessarily YoY --
    see `yoy` for that). NaN propagates through a missing input, never 0."""
    result = df.copy()
    result["value"] = df["value"].pct_change(periods=periods) * 100
    formula = f"pct_change = (value[t] - value[t-{periods}]) / value[t-{periods}] * 100"
    return result, formula


def yoy(df: pd.DataFrame, periods_per_year: int) -> tuple[pd.DataFrame, str]:
    """Year-over-year percent change. `periods_per_year` is the series'
    own observation count per year (12 monthly, 4 quarterly, 52 weekly, 1 yearly)."""
    result, _ = pct_change(df, periods=periods_per_year)
    formula = f"yoy = (value[t] - value[t-{periods_per_year}]) / value[t-{periods_per_year}] * 100"
    return result, formula


def diff(df: pd.DataFrame, periods: int = 1) -> tuple[pd.DataFrame, str]:
    """First difference vs. `periods` observations back."""
    result = df.copy()
    result["value"] = df["value"].diff(periods=periods)
    formula = f"diff = value[t] - value[t-{periods}]"
    return result, formula


def align_join(
    series: Mapping[str, pd.DataFrame], *, how: str = "outer"
) -> tuple[pd.DataFrame, str]:
    """Merge several "period"/"value" series into one wide frame (one column
    per name in `series`), keyed on period. Periods missing from a given
    series stay NaN in that column -- never filled."""
    if not series:
        raise ValueError("align_join: en az bir seri gerekli")

    names = list(series.keys())
    result = series[names[0]][["period", "value"]].rename(columns={"value": names[0]})
    for name in names[1:]:
        other = series[name][["period", "value"]].rename(columns={"value": name})
        result = result.merge(other, on="period", how=how)

    result = result.sort_values("period").reset_index(drop=True)
    formula = f"align_join(how={how}): dönemler birleştirilir, eksikler NaN kalır (doldurma yok)"
    return result, formula
