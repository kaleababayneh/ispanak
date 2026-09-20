"""A synthetic stand-in for the data lane, with the same function signatures.

The agent lane cannot wait for the real gold layer, so it is built against this
and swapped later: when `lake/gold.py` lands, only the object passed to the
agent changes, not the loop, the tools or their tests.

The numbers are invented but the *shapes* are real: the seven observation
columns in order, `period` as the first day of the period, weekly rates that
average into months, a CPI whose 2025 mean is exactly 100, and one planted fact
the scenario depends on — a window where the rate falls while the loan balance
does not grow. Without that window the scenario's first question has no answer
to find, and the test asserting it would pass vacuously.

`period` is a real datetime64 column everywhere (pandas 3 resolves it to
microsecond resolution by default). In pandas 3, merging a column of
`datetime.date` objects against a datetime column either raises or silently
matches nothing, and `get_series(into="active")` plus `deflate` both do exactly
that merge, so the dtype is asserted in the tests rather than assumed.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from lakehouse_agent.contracts import CatalogHit, QualityReport, SeriesNotFound

FETCHED_AT = datetime(2026, 9, 15, 6, 0, tzinfo=UTC)
START = "2021-01"
END = "2026-06"
CPI_BASE_YEAR = 2025

#: Months where the rate fell versus the previous month but the loan balance did
#: not grow. The scenario's first question is exactly "did that ever happen?".
PLANTED_RATE_DOWN_CREDIT_FLAT = ["2022-04", "2022-05", "2022-06"]

RATE_SERIES = "evds:TP.KTF12"
CREDIT_SERIES = "bddk:aylik:tuketici:konut:TL:sektor"
CPI_SERIES = "evds:TP.TUKFIY2025.GENEL"
HPI_SERIES = "evds:TP.KFE.TR"
RATE_STOCK_SERIES = "evds:TP.BKR.TRY.18"

SCENARIO_SERIES: dict[str, dict[str, Any]] = {
    RATE_SERIES: {
        "name_tr": "Konut Kredisi Faiz Orani (TL, Akim, Agirlikli Ortalama)",
        "name_en": "Housing Loan Interest Rate (TRY, Flow, Weighted Average)",
        "source": "EVDS",
        "freq": "W",
        "unit": "%",
        "kind": "rate",
        "agg_method": "avg",
        "cum_type": "none",
        "topic": "Faiz Istatistikleri",
        "tags": "konut, kredi, faiz, akim, haftalik",
    },
    RATE_STOCK_SERIES: {
        "name_tr": "Konut Kredisi Faiz Orani (TL, Stok)",
        "name_en": "Housing Loan Interest Rate (TRY, Stock)",
        "source": "EVDS",
        "freq": "M",
        "unit": "%",
        "kind": "rate",
        "agg_method": "avg",
        "cum_type": "none",
        "topic": "Faiz Istatistikleri",
        "tags": "konut, kredi, faiz, stok, aylik",
    },
    CREDIT_SERIES: {
        "name_tr": "Tuketici Kredileri - Konut (TL, Sektor)",
        "name_en": "Consumer Loans - Housing (TRY, Sector)",
        "source": "BDDK-Aylik",
        "freq": "M",
        "unit": "milyon TL",
        "kind": "stock",
        "agg_method": "last",
        "cum_type": "none",
        "topic": "Krediler",
        "tags": "konut, kredi, bakiye, tuketici",
    },
    CPI_SERIES: {
        "name_tr": "Tuketici Fiyat Endeksi (2025=100, Genel)",
        "name_en": "Consumer Price Index (2025=100, General)",
        "source": "EVDS",
        "freq": "M",
        "unit": "endeks",
        "kind": "index",
        "agg_method": "avg",
        "cum_type": "none",
        "topic": "Fiyat Endeksleri",
        "tags": "tufe, enflasyon, fiyat, endeks",
    },
    HPI_SERIES: {
        "name_tr": "Konut Fiyat Endeksi (Turkiye)",
        "name_en": "House Price Index (Turkiye)",
        "source": "EVDS",
        "freq": "M",
        "unit": "endeks",
        "kind": "index",
        "agg_method": "avg",
        "cum_type": "none",
        "topic": "Fiyat Endeksleri",
        "tags": "konut, fiyat, endeks, kfe",
    },
}

_CATALOG_COLUMNS = [
    "series_id",
    "name_tr",
    "name_en",
    "source",
    "freq",
    "unit",
    "kind",
    "agg_method",
    "cum_type",
    "start_date",
    "end_date",
    "topic",
    "tags",
    "metadata_url",
    "notes",
]

_OBSERVATION_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
]

_AGG_FOR_KIND = {"stock": "last", "flow": "sum", "rate": "mean", "index": "mean"}


def _months() -> pd.DatetimeIndex:
    return pd.date_range(START + "-01", END + "-01", freq="MS")


def _mondays() -> pd.DatetimeIndex:
    """Weekly periods normalised to the Monday of the week (the data lane's rule).

    Starts at the first Monday *on or after* the start date, never before it.
    Reaching back into the previous month would put weeks in the series that
    `get()` then filters out by date, which leaves the first month with an
    incomplete set of weeks whose wobbles no longer cancel, so its monthly
    average drifts off the intended level.
    """
    start = pd.Timestamp(START + "-01")
    end = pd.Timestamp(END + "-01") + pd.offsets.MonthEnd(0)
    days_until_monday = (7 - start.weekday()) % 7
    first_monday = start + pd.Timedelta(days=days_until_monday)
    return pd.date_range(first_monday, end, freq="W-MON")


def _monthly_rate() -> pd.Series:
    """Rate path: low in 2021, peaks in early 2024, falls back through 2026.

    The shape matters more than the level: the scenario asks about periods where
    the rate fell, so there have to be several of them.
    """
    months = _months()
    steps = range(len(months))
    values = []
    for i in steps:
        if i < 15:  # 2021-01 .. 2022-03, slow climb
            value = 12.0 + i * 0.45
        elif i < 24:  # 2022-04 .. 2022-12, the planted dip then recovery
            value = 18.5 - (i - 14) * 0.6
        elif i < 38:  # 2023, steep climb
            value = 14.0 + (i - 23) * 2.4
        elif i < 44:  # early 2024 peak
            value = 47.0 + (i - 37) * 0.3
        else:  # decline to ~32
            value = 48.8 - (i - 43) * 0.75
        values.append(round(max(value, 8.0), 2))
    return pd.Series(values, index=months, name="rate")


def _monthly_credit() -> pd.Series:
    """Loan balance in million TRY: grows, but deliberately flat in the planted window."""
    months = _months()
    planted = {pd.Timestamp(m + "-01") for m in PLANTED_RATE_DOWN_CREDIT_FLAT}
    values = []
    balance = 250_000.0
    for month in months:
        if month in planted:
            growth = -0.002  # shrinks slightly: rate down, credit not growing
        elif month.year <= 2022:
            growth = 0.012
        elif month.year <= 2024:
            growth = 0.028
        else:
            growth = 0.019
        balance *= 1 + growth
        values.append(round(balance, 1))
    return pd.Series(values, index=months, name="credit")


def _monthly_cpi() -> pd.Series:
    """CPI rebased so the 2025 mean is exactly 100, matching the real series' base."""
    months = _months()
    raw = []
    level = 100.0
    for month in months:
        monthly_inflation = (
            0.031 if month.year <= 2022 else 0.042 if month.year <= 2024 else 0.025
        )
        level *= 1 + monthly_inflation
        raw.append(level)
    series = pd.Series(raw, index=months)
    base = series[series.index.year == CPI_BASE_YEAR].mean()
    return (series / base * 100).round(3).rename("cpi")


def _monthly_hpi() -> pd.Series:
    """House price index, 2021-01 = 100, rising faster than CPI after 2022."""
    months = _months()
    values = []
    level = 100.0
    for month in months:
        growth = 0.022 if month.year <= 2022 else 0.035 if month.year <= 2024 else 0.021
        level *= 1 + growth
        values.append(round(level, 2))
    return pd.Series(values, index=months, name="hpi")


def _weekly_from_monthly(monthly: pd.Series) -> pd.Series:
    """Spread a monthly path over weeks so each month's weekly mean equals it.

    The wobble inside a month is symmetric around zero on purpose. Weekly noise
    that does not cancel would shift the monthly averages by a little, and that
    was enough to make one of the planted months stop qualifying once the rate
    was aggregated back up — a fake that quietly destroys the fact it exists to
    provide.
    """
    weeks = _mondays()
    by_month: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for week in weeks:
        month = pd.Timestamp(year=week.year, month=week.month, day=1)
        month = min(max(month, monthly.index[0]), monthly.index[-1])
        by_month.setdefault(month, []).append(week)

    values: dict[pd.Timestamp, float] = {}
    for month, month_weeks in by_month.items():
        level = float(monthly.loc[month])
        count = len(month_weeks)
        centre = (count - 1) / 2
        spread = max(centre, 1.0)
        for index, week in enumerate(month_weeks):
            wobble = 0.02 * (index - centre) / spread
            values[week] = round(level * (1 + wobble), 4)
    return pd.Series([values[w] for w in weeks], index=weeks, name="weekly")


def synthetic_observations() -> pd.DataFrame:
    """The whole fake pool in long form: exactly the seven gold columns."""
    frames: list[pd.DataFrame] = []

    def add(series_id: str, values: pd.Series) -> None:
        meta = SCENARIO_SERIES[series_id]
        frames.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "period": values.index,
                    "freq": meta["freq"],
                    "value": values.to_numpy(dtype="float64"),
                    "unit": meta["unit"],
                    "source": meta["source"],
                    "fetched_at": FETCHED_AT,
                }
            )
        )

    monthly_rate = _monthly_rate()
    add(RATE_SERIES, _weekly_from_monthly(monthly_rate))
    add(RATE_STOCK_SERIES, (monthly_rate.rolling(6, min_periods=1).mean()).round(2))
    add(CREDIT_SERIES, _monthly_credit())
    add(CPI_SERIES, _monthly_cpi())
    add(HPI_SERIES, _monthly_hpi())

    out = pd.concat(frames, ignore_index=True)
    out["period"] = pd.to_datetime(out["period"])
    return out[_OBSERVATION_COLUMNS]


class FakeLake:
    """Same call signatures as the real data lane, synthetic numbers behind them."""

    def __init__(self) -> None:
        self._observations = synthetic_observations()

    # -- reads -------------------------------------------------------------

    def get(self, series_ids: list[str], start: str, end: str) -> pd.DataFrame:
        """Long form, exactly the seven gold columns, `period` as datetime64."""
        missing = [sid for sid in series_ids if sid not in SCENARIO_SERIES]
        if missing:
            raise SeriesNotFound(missing, "Once search_catalog ile arama yapin.")

        lo = pd.Timestamp(start + "-01")
        hi = pd.Timestamp(end + "-01") + pd.offsets.MonthEnd(0)
        frame = self._observations
        mask = (
            frame["series_id"].isin(series_ids)
            & (frame["period"] >= lo)
            & (frame["period"] <= hi)
        )
        return frame.loc[mask].reset_index(drop=True)

    def wide(
        self, series_ids: list[str], start: str, end: str, freq: str = "M"
    ) -> pd.DataFrame:
        """One `period` column plus one column per series, aligned on period.

        Rejects an empty list and duplicates rather than limping on: both break
        the promise that every column is named by exactly one series id, and a
        duplicate would silently come back as `<id>_x` / `<id>_y` from the merge.
        """
        if not series_ids:
            raise ValueError("wide() icin en az bir series_id gerekli.")
        duplicates = sorted({sid for sid in series_ids if series_ids.count(sid) > 1})
        if duplicates:
            raise ValueError(
                f"series_ids listesinde tekrar var: {', '.join(duplicates)}"
            )

        long = self.get(series_ids, start, end)
        columns: list[pd.DataFrame] = []
        for series_id in series_ids:
            part = long[long["series_id"] == series_id][["period", "value"]]
            meta = SCENARIO_SERIES[series_id]
            if meta["freq"] != freq:
                part = self.to_freq(part, freq, meta["kind"], meta["agg_method"])
            columns.append(part.rename(columns={"value": series_id}))

        out = columns[0]
        for part in columns[1:]:
            out = out.merge(part, on="period", how="outer")
        out = out.sort_values("period").reset_index(drop=True)
        out["period"] = pd.to_datetime(out["period"])
        return out

    def to_freq(
        self,
        df: pd.DataFrame,
        target: str,
        kind: str,
        agg_method: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate to a lower frequency using the rule implied by `kind`.

        Balances take the period's last value, flows sum, rates and indices
        average. Going the other way (monthly to weekly) would be inventing
        data and is not supported.
        """
        how = agg_method or _AGG_FOR_KIND.get(kind, "mean")
        how = {"avg": "mean", "last": "last", "first": "first", "sum": "sum"}.get(
            how, how
        )
        offsets = {"M": "MS", "W": "W-MON"}
        if target not in offsets:
            # `Freq` also declares D, Q and A. Only the two the pool actually
            # uses are implemented, and an unsupported one must say so rather
            # than surface as a bare KeyError from a dict lookup.
            raise ValueError(
                f"Desteklenmeyen hedef frekans: {target!r}. "
                f"Desteklenenler: {', '.join(sorted(offsets))}."
            )
        offset = offsets[target]
        grouped = (
            df.set_index("period")["value"]
            .resample(offset)
            .agg(how)
            .dropna()
            .reset_index()
        )
        grouped["period"] = pd.to_datetime(grouped["period"])
        return grouped

    def catalog_lookup(self, series_ids: list[str]) -> pd.DataFrame:
        missing = [sid for sid in series_ids if sid not in SCENARIO_SERIES]
        if missing:
            raise SeriesNotFound(missing)
        rows = []
        for series_id in series_ids:
            meta = SCENARIO_SERIES[series_id]
            rows.append(
                {
                    "series_id": series_id,
                    "name_tr": meta["name_tr"],
                    "name_en": meta["name_en"],
                    "source": meta["source"],
                    "freq": meta["freq"],
                    "unit": meta["unit"],
                    "kind": meta["kind"],
                    "agg_method": meta["agg_method"],
                    "cum_type": meta["cum_type"],
                    "start_date": START,
                    "end_date": END,
                    "topic": meta["topic"],
                    "tags": meta["tags"],
                    "metadata_url": None,
                    "notes": "sentetik",
                }
            )
        return pd.DataFrame(rows, columns=_CATALOG_COLUMNS)

    def search(self, query: str, k: int = 20) -> list[CatalogHit]:
        """Keyword overlap, plus an exact-match shortcut for explicit series codes.

        The scores are not similarities in 0..1. The real catalog uses hybrid
        search whose fused scores are only comparable within one query, so
        ambiguity downstream is judged as a *ratio* between the top two hits,
        never as an absolute gap. This fake reproduces that property.
        """
        code_match = re.search(r"\b(?:TP|TP\.)[A-Z0-9._]+\b", query.upper())
        if code_match:
            wanted = code_match.group(0).lstrip(".")
            for series_id in SCENARIO_SERIES:
                if series_id.split(":", 1)[-1].upper() == wanted:
                    return [self._hit(series_id, 1.0)]

        tokens = {t for t in re.split(r"\W+", query.casefold()) if len(t) > 2}
        scored: list[tuple[float, str]] = []
        for series_id, meta in SCENARIO_SERIES.items():
            haystack = f"{meta['name_tr']} {meta['tags']} {meta['topic']}".casefold()
            overlap = sum(1 for token in tokens if token in haystack)
            if overlap:
                rank_score = overlap / (1 + len(tokens))
                scored.append((rank_score, series_id))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._hit(series_id, score) for score, series_id in scored[:k]]

    def _hit(self, series_id: str, score: float) -> CatalogHit:
        meta = SCENARIO_SERIES[series_id]
        return CatalogHit(
            series_id=series_id,
            name_tr=meta["name_tr"],
            name_en=meta["name_en"],
            source=meta["source"],
            freq=meta["freq"],
            unit=meta["unit"],
            kind=meta["kind"],
            agg_method=meta["agg_method"],
            cum_type=meta["cum_type"],
            start_date=START,
            end_date=END,
            score=round(score, 4),
        )

    # -- demo-time helpers (thin until the real ones land) -----------------

    def prepare_table(
        self, df: pd.DataFrame, hints: dict[str, Any] | None = None
    ) -> tuple[pd.DataFrame, QualityReport]:
        """Minimal clean-up for a table that arrived from outside the pool.

        Rows whose period cannot be parsed are dropped, and the count is put on
        the quality card. Dropping them silently would defeat the card's whole
        purpose: `profile_table` runs on the *cleaned* frame, so without this
        the report describes data the caller never saw and a human approving a
        quality gate would have no idea anything went missing.
        """
        out = df.copy()
        dropped = 0
        if "period" in out.columns:
            out["period"] = pd.to_datetime(out["period"], errors="coerce")
            dropped = int(out["period"].isna().sum())
            out = out.dropna(subset=["period"]).sort_values("period")
        out = out.reset_index(drop=True)

        report = self.profile_table(out)
        if dropped:
            report = report.model_copy(
                update={
                    "issues": [
                        f"{dropped} satir tarihi okunamadigi icin dusuruldu.",
                        *report.issues,
                    ]
                }
            )
        return out, report

    def profile_table(self, df: pd.DataFrame) -> QualityReport:
        """The quality card: coverage, gaps, duplicates, outliers.

        Every field it reports, it actually computes. That matters because this
        card is what the quality-gate approval decides on: a card that says
        `missing_periods=0` for a table with a hole in it would wave through
        exactly the data a human was asked to look at.

        `unit_guess` is the one field left as None. Guessing a unit from column
        names would be a guess presented as a measurement; the real lake knows
        the unit from the catalog and fills it in there.
        """
        date_range: tuple[str, str] | None = None
        freq: str | None = None
        duplicates = 0
        missing_periods = 0
        issues: list[str] = []

        periods = None
        if "period" in df.columns and len(df):
            periods = pd.to_datetime(df["period"]).dropna()

        if periods is not None and len(periods):
            date_range = (
                periods.min().strftime("%Y-%m"),
                periods.max().strftime("%Y-%m"),
            )
            duplicates = int(periods.duplicated().sum())
            if len(periods) > 2:
                gap = periods.sort_values().diff().dt.days.median()
                freq = (
                    "M"
                    if 27 <= gap <= 32
                    else "W"
                    if 6 <= gap <= 8
                    else "D"
                    if gap <= 2
                    else None
                )

            offset = {"M": "MS", "W": "W-MON", "D": "D"}.get(freq or "")
            if offset:
                expected = pd.date_range(periods.min(), periods.max(), freq=offset)
                missing_periods = max(len(expected) - periods.nunique(), 0)

        # A table with no numeric column has no cells whose absence could be
        # measured, so the percentage is 0 rather than a number computed as if
        # one column existed -- a fabricated denominator reads like a finding.
        numeric = df.select_dtypes("number")
        column_count = numeric.shape[1]
        if column_count:
            nan_cells = int(numeric.isna().to_numpy().sum())
            expected_cells = (len(df) + missing_periods) * column_count
            missing_cells = nan_cells + missing_periods * column_count
            missing_pct = (
                (missing_cells / expected_cells * 100) if expected_cells else 0.0
            )
        else:
            missing_pct = 0.0

        outliers = 0
        for column in numeric.columns:
            values = numeric[column].dropna()
            if len(values) < 4:
                continue
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            spread = q3 - q1
            if spread > 0:
                outliers += int(
                    ((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum()
                )

        if duplicates:
            issues.append(f"{duplicates} tekrarli donem var.")
        if missing_periods:
            issues.append(f"{missing_periods} donem eksik.")
        if freq is None and periods is not None and len(periods) > 2:
            issues.append("Frekans cikarilamadi; donemler duzensiz.")

        return QualityReport(
            rows=len(df),
            date_range=date_range,
            freq=freq,
            missing_periods=missing_periods,
            missing_pct=round(missing_pct, 2),
            duplicates=duplicates,
            outliers=outliers,
            issues=issues,
        )
