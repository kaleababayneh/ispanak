"""Pure DataFrame -> DataFrame transforms. No model, session, or tool
knowledge (CLAUDE.md #7) -- every function here takes plain pandas
DataFrames with a "period"/"value" shape and returns the same shape, plus
the formula it applied as a string.

NaN is never filled/interpolated (CLAUDE.md #2): a period missing from the
input stays missing, and an aggregation window with no real observations
stays NaN rather than becoming 0.
"""

import logging
import operator
from collections.abc import Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_KIND_TO_AGG = {"stock": "last", "flow": "sum", "rate": "mean", "index": "mean"}
_KIND_TO_FORMULA = {
    "stock": "value[period] = last(value[alt_dönem])",
    "flow": "value[period] = sum(value[alt_dönem])",
    "rate": "value[period] = mean(value[alt_dönem])",
    "index": "value[period] = mean(value[alt_dönem])",
}

# The gold contract's Freq (D|W|M|Q|A, contracts.py) -> the pandas offset
# alias whose period lands on the FIRST day, matching the contract's "period
# = first day of period" convention. Measured: passing the bare contract
# letter straight to `.resample()` crashes on pandas 3.x --
# `to_freq(df, "rate", "M")` -> ValueError: "'M' is no longer supported for
# offsets. Please use 'ME' instead." -- and "ME"/"QE"/"YE" (period END) would
# silently violate the contract even if they didn't crash. W maps to
# W-MON specifically to match lake/silver.py's own Monday-anchored weekly
# period convention.
_FREQ_CONTRACT_TO_PANDAS = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "A": "YS"}


def _resolve_freq_alias(freq: str) -> str:
    """Accept the gold contract's Freq value as first-class input,
    translating it to the matching pandas offset alias. A pandas alias
    passed directly (already "MS", or "ME" if a caller genuinely wants
    period-end) passes through unchanged."""
    return _FREQ_CONTRACT_TO_PANDAS.get(freq, freq)


def to_freq(df: pd.DataFrame, kind: str, freq: str) -> tuple[pd.DataFrame, str]:
    """Resample a "period"/"value" series to a lower frequency, aggregating
    per `kind` (stock->last, flow->sum, rate->avg, index->avg).

    `freq` is the gold contract's Freq value (D|W|M|Q|A, contracts.py) --
    translated internally to the pandas offset alias for that period's
    FIRST day (MS/QS/YS/W-MON/D), matching the gold contract's "period =
    first day of period" convention. A raw pandas offset alias (e.g.
    already "MS", or "ME" for period-end) is also accepted as-is.
    """
    if kind not in _KIND_TO_AGG:
        raise ValueError(f"Bilinmeyen kind: {kind!r}")

    working = df.copy()
    working["period"] = pd.to_datetime(working["period"])
    grouped = working.set_index("period")["value"].resample(_resolve_freq_alias(freq))

    if _KIND_TO_AGG[kind] == "sum":
        resampled = grouped.sum(min_count=1)  # all-NaN window -> NaN, not 0
    else:
        resampled = getattr(grouped, _KIND_TO_AGG[kind])()

    result = resampled.reset_index()
    return result, _KIND_TO_FORMULA[kind]


def _infer_freq_alias(idx: pd.DatetimeIndex) -> str:
    """Infer the series' own calendar step from the MINIMUM gap between
    sorted, deduped periods -- robust to an occasional missing period,
    unlike `pd.infer_freq`, which demands a perfectly regular index and
    returns None the moment there's a gap (exactly the case this needs to
    tolerate: at least one genuinely-adjacent pair elsewhere in the data is
    enough to infer the right step).
    """
    unique_sorted = pd.DatetimeIndex(idx).unique().sort_values()
    if len(unique_sorted) < 2:
        raise ValueError("takvim adımı çıkarmak için en az 2 farklı dönem gerekli")
    min_days = int(pd.Series(unique_sorted).diff().dropna().dt.days.min())
    if min_days <= 1:
        return "D"
    if min_days == 7:
        return "W-MON"
    if 28 <= min_days <= 31:
        return "MS"
    if 89 <= min_days <= 92:
        return "QS"
    if 365 <= min_days <= 366:
        return "YS"
    raise ValueError(f"takvim adımı çıkarılamadı (min gün farkı={min_days})")


def _try_infer_freq_alias(df: pd.DataFrame) -> str | None:
    """Best-effort `_infer_freq_alias`: None when there isn't enough data
    to infer a cadence (fewer than 2 distinct periods) -- callers should
    treat None as "can't tell, don't block", not as an error."""
    unique = pd.to_datetime(df["period"]).unique()
    if len(unique) < 2:
        return None
    return _infer_freq_alias(pd.DatetimeIndex(unique))


def _gridded_value_series(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Sort by period and reindex "value" onto the series' own inferred
    calendar grid (gaps become explicit NaN rows) so a later positional
    `.shift()`/`.pct_change()`/`.diff()` lands on the calendar-correct
    period instead of silently using whatever row happens to be N rows
    away. Returns (sorted_df, gridded_series, original_period_index) --
    callers reindex the shifted result back onto `original_period_index`
    to keep the original row count/order.
    """
    sorted_df = df.sort_values("period").reset_index(drop=True)
    idx = pd.to_datetime(sorted_df["period"])
    gridded = pd.Series(sorted_df["value"].to_numpy(), index=idx).asfreq(
        _infer_freq_alias(idx)
    )
    return sorted_df, gridded, idx


def deflate(
    nominal: pd.DataFrame, price_index: pd.DataFrame, *, base: float = 100.0
) -> tuple[pd.DataFrame, str]:
    """Deflate a nominal "period"/"value" series by a price index series
    (also "period"/"value"), base=100. Every period present in `nominal` is
    kept; a period with no matching index value gets NaN, not a dropped row.

    Raises if the two series have genuinely different calendar frequencies
    (e.g. monthly nominal vs. weekly index) -- merging those on exact
    period match silently produces mostly-NaN output with no error
    (measured: monthly nominal + weekly index -> 9/12 NaN). A same-frequency
    series with merely fewer periods than `nominal` is not a mismatch (that
    stays NaN, as before) -- only resolved when both sides have >= 2
    periods to infer a cadence from.
    """
    nominal_freq = _try_infer_freq_alias(nominal)
    index_freq = _try_infer_freq_alias(price_index)
    if nominal_freq and index_freq and nominal_freq != index_freq:
        raise ValueError(
            f"deflate: nominal ({nominal_freq}) ve fiyat endeksi ({index_freq}) "
            "farklı frekanslarda -- önce to_freq ile aynı frekansa hizalayın."
        )

    merged = nominal.merge(
        price_index, on="period", how="left", suffixes=("", "_index")
    )
    unmatched = int(merged["value_index"].isna().sum())
    if unmatched:
        logger.warning(
            "deflate: %d/%d dönemde fiyat endeksi eşleşmedi (NaN kaldı)",
            unmatched,
            len(merged),
        )
    merged["value"] = merged["value"] / (merged["value_index"] / base)
    result = merged[["period", "value"]]
    formula = f"value_real = value_nominal / (index / {base})"
    return result, formula


def _base_value(df: pd.DataFrame, base_period) -> float:
    """The value at `base_period`, refusing a base that cannot be divided by."""
    base_rows = df.loc[df["period"] == base_period, "value"]
    if base_rows.empty:
        raise ValueError(f"base_period {base_period} seride yok")
    base_raw = base_rows.iloc[0]
    if pd.isna(base_raw) or base_raw == 0:
        raise ValueError(
            f"base_period {base_period} değeri geçersiz (NaN veya 0): "
            f"{base_raw!r} -- 0'a bölünemez."
        )
    return base_raw


def rebase(
    df: pd.DataFrame, base_period, *, base_value: float = 100.0
) -> tuple[pd.DataFrame, str]:
    """Rescale an index series so `base_period` equals `base_value`."""
    base_raw = _base_value(df, base_period)
    result = df.copy()
    result["value"] = df["value"] / base_raw * base_value
    formula = f"value_rebased = value / value[{base_period}] * {base_value}"
    return result, formula


def change_since(df: pd.DataFrame, base_period) -> tuple[pd.DataFrame, str]:
    """Percent change of every period against one fixed `base_period`.

    "How much has CPI risen since the start of 2021" is this, not
    `pct_change`: that one compares each period with the one before it.
    """
    base_raw = _base_value(df, base_period)
    result = df.copy()
    result["value"] = (df["value"] / base_raw - 1) * 100
    formula = f"change = (value / value[{base_period}] - 1) * 100"
    return result, formula


def ratio(
    numerator: pd.DataFrame, denominator: pd.DataFrame, *, scale: float = 100.0
) -> tuple[pd.DataFrame, str]:
    """`numerator / denominator * scale` per period: a share, in percent by
    default. Every period of `numerator` is kept; a period missing from the
    denominator, or a zero denominator, gives NaN -- never 0 and never inf.

    Refuses two different calendar frequencies for the same reason `deflate`
    does: an exact-period merge of monthly against weekly is mostly NaN and
    raises nothing.
    """
    num_freq = _try_infer_freq_alias(numerator)
    den_freq = _try_infer_freq_alias(denominator)
    if num_freq and den_freq and num_freq != den_freq:
        raise ValueError(
            f"ratio: pay ({num_freq}) ve payda ({den_freq}) farklı frekanslarda -- "
            "önce to_freq ile aynı frekansa hizalayın."
        )
    merged = numerator.merge(
        denominator, on="period", how="left", suffixes=("", "_den")
    )
    denominator_values = merged["value_den"].where(merged["value_den"] != 0)
    merged["value"] = merged["value"] / denominator_values * scale
    formula = f"ratio = numerator / denominator * {scale:g}"
    return merged[["period", "value"]], formula


_COMPARATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def mark_periods(
    series: Mapping[str, pd.DataFrame],
    conditions: Sequence[tuple[str, str, float]],
) -> tuple[pd.DataFrame, str]:
    """Flag the periods where every condition holds.

    `series` maps a name to a "period"/"value" frame; each condition is
    `(name, comparator, threshold)`, e.g. `("faiz_diff", "<", 0)`, and all of
    them must hold (AND). This is how "months where the rate fell but credit
    did not grow" is answered by the tool rather than by the model counting
    rows in its head.

    The result is "period"/"value" with a nullable boolean. A period where a
    condition cannot be evaluated because its input is missing is <NA>, not
    False: the first month of a difference has no previous month, which is
    not the same as "the rate did not fall". AND follows Kleene logic, so a
    period where some condition is definitely False is False regardless.
    """
    if not conditions:
        raise ValueError("mark_periods: en az bir koşul gerekli")
    names = [name for name, _, _ in conditions]
    missing = [name for name in names if name not in series]
    if missing:
        raise ValueError(f"mark_periods: bilinmeyen seri(ler): {', '.join(missing)}")

    joined, _ = align_join({name: series[name] for name in dict.fromkeys(names)})
    flags = pd.Series(True, index=joined.index, dtype="boolean")
    parts = []
    for name, comparator, threshold in conditions:
        compare = _COMPARATORS.get(comparator)
        if compare is None:
            raise ValueError(
                f"mark_periods: bilinmeyen karşılaştırma {comparator!r}; "
                f"geçerli olanlar: {', '.join(_COMPARATORS)}"
            )
        values = pd.to_numeric(joined[name], errors="coerce")
        holds = compare(values, threshold).astype("boolean")
        holds[values.isna()] = pd.NA
        flags = flags & holds
        parts.append(f"{name} {comparator} {threshold:g}")

    result = pd.DataFrame({"period": joined["period"], "value": flags})
    return result, "mark = " + " AND ".join(parts)


def pct_change(df: pd.DataFrame, periods: int = 1) -> tuple[pd.DataFrame, str]:
    """Percent change vs. `periods` observations back (not necessarily YoY --
    see `yoy` for that). NaN propagates through a missing input, never 0.

    Shifts over the series' own inferred CALENDAR, not row position: a
    missing period (a dropped row, not a NaN value) used to silently shift
    every later row's comparison onto the wrong calendar period. Input is
    sorted by period first -- an unsorted/reversed input used to flip the
    sign.
    """
    formula = f"pct_change = (value[t] - value[t-{periods}]) / value[t-{periods}] * 100"
    if df["period"].nunique() < 2:
        result = df.sort_values("period").reset_index(drop=True).copy()
        result["value"] = float("nan")
        return result, formula

    sorted_df, gridded, idx = _gridded_value_series(df)
    result = sorted_df.copy()
    result["value"] = (
        (gridded.pct_change(periods=periods) * 100).reindex(idx).to_numpy()
    )
    return result, formula


def yoy(df: pd.DataFrame, periods_per_year: int) -> tuple[pd.DataFrame, str]:
    """Year-over-year percent change. `periods_per_year` is the series'
    own observation count per year (12 monthly, 4 quarterly, 52 weekly, 1 yearly)."""
    result, _ = pct_change(df, periods=periods_per_year)
    formula = f"yoy = (value[t] - value[t-{periods_per_year}]) / value[t-{periods_per_year}] * 100"
    return result, formula


def diff(df: pd.DataFrame, periods: int = 1) -> tuple[pd.DataFrame, str]:
    """First difference vs. `periods` observations back. Calendar-shifted
    and sorted first, same reasoning as `pct_change`."""
    formula = f"diff = value[t] - value[t-{periods}]"
    if df["period"].nunique() < 2:
        result = df.sort_values("period").reset_index(drop=True).copy()
        result["value"] = float("nan")
        return result, formula

    sorted_df, gridded, idx = _gridded_value_series(df)
    result = sorted_df.copy()
    result["value"] = gridded.diff(periods=periods).reindex(idx).to_numpy()
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
