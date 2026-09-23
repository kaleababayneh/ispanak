"""Data quality report over the gold layer (plan 2.6).

Pure DataFrame -> DataFrame computation (`build_quality_report`, CLAUDE.md
#7: no model/session/tool knowledge) plus a thin markdown renderer/writer,
runnable standalone as `python -m lakehouse_agent.lake.quality`.

Per series: coverage (first/last observed period), missing-period count,
duplicate-period count, simple-IQR outlier count, last update. Missing
periods are counted only from a series' OWN first observed value onward --
never before it. This matters concretely for TP.HPBITABLO6.3: EVDS's bulk
endpoint returns every week of a requested year even before a series
existed, as real JSON `null` (docs/data_dictionary.md), so
`data/gold/observations.parquet` actually carries rows for that series back
to 2020-12-28 with `value=NaN`, though the series itself only starts
2024-06-24. Counting from the raw min(period) would report ~182 weeks of
"missing" data that was never expected to exist. `pd.Timestamp` and
`datetime.date` compare unequal even for the same calendar day (verified:
`pd.Timestamp("2024-06-24") == date(2024, 6, 24)` is False) -- gold's
`period` column is `datetime.date` objects (parquet DATE -> pandas object
dtype), so the expected-calendar comparison is done in `.date` space
throughout, not `Timestamp` space, to avoid a silent all-missing false
positive.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lakehouse_agent.contracts import QualityReport

logger = logging.getLogger(__name__)

GOLD_ROOT = Path("data/gold")
OBSERVATIONS_PATH = GOLD_ROOT / "observations.parquet"
SERIES_CATALOG_PATH = GOLD_ROOT / "series_catalog.parquet"
REPORT_PATH = Path("docs/data_quality.md")
#: Per-series rows, for a pool too large to list in the markdown.
DETAIL_PATH = GOLD_ROOT / "quality_by_series.parquet"
#: Up to this many series the markdown lists every one of them.
DETAIL_ROWS_IN_REPORT = 200

QUALITY_COLUMNS = [
    "series_id",
    "name_tr",
    "source",
    "freq",
    "start",
    "end",
    "n",
    "missing",
    "duplicates",
    "outliers",
    "last_updated",
]


def _expected_periods(freq: str, start, end):
    """Full calendar between `start` and `end`, as `datetime.date` objects.
    `freq` is the gold contract's Freq literal (D|W|M|Q|A). Kept for callers
    that need the dates themselves; the report counts with
    `_expected_count`."""
    aliases = {"M": "MS", "W": "7D", "D": "D", "Q": "QS", "A": "YS"}
    if freq not in aliases:
        raise ValueError(f"quality.py bu frekansı desteklemiyor: {freq!r}")
    return set(pd.date_range(start, end, freq=aliases[freq]).date)


def _iqr_outlier_count(values: pd.Series) -> int:
    """Simple Tukey 1.5*IQR rule, same shape as analysis/anomaly.py's
    short-series fallback. Fewer than 4 points can't split into quartiles
    meaningfully, so those series report 0 rather than a degenerate value.
    """
    if len(values) < 4:
        return 0
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < lower) | (values > upper)).sum())


def _expected_count(
    freq: str, start: pd.Series, end: pd.Series, weekdays_only: pd.Series
) -> pd.Series:
    """How many periods lie between each series' own first and last value,
    both included. A daily series that never has a weekend value is a
    business-day series (EVDS "İŞ GÜNÜ" and markets): its weekends are not
    missing. Public holidays still count as missing -- no holiday calendar
    is assumed."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    days = (end - start).dt.days
    business = pd.Series(
        np.busday_count(
            start.to_numpy(dtype="datetime64[D]"),
            (end + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[D]"),
        ),
        index=start.index,
    )
    calendar = {
        "M": months + 1,
        "Q": months // 3 + 1,
        "A": end.dt.year - start.dt.year + 1,
        "W": days // 7 + 1,
        "D": days.where(~weekdays_only, business - 1) + 1,
    }
    out = pd.Series(0, index=start.index)
    for code, counts in calendar.items():
        out = out.where(freq != code, counts)
    unknown = ~freq.isin(list(calendar))
    if unknown.any():
        raise ValueError(
            f"quality.py bu frekansı desteklemiyor: {sorted(set(freq[unknown]))}"
        )
    return out.astype(int)


def build_quality_report(
    observations: pd.DataFrame, series_catalog: pd.DataFrame
) -> pd.DataFrame:
    """One row per `series_catalog` series_id, with coverage/missing/
    duplicate/outlier counts computed from `observations`, all series at
    once (the pool is tens of thousands of series; a filter per series was
    minutes). A series with no non-null observations reports n=0/missing=0
    rather than raising.
    """
    catalog = series_catalog[
        ["series_id", "name_tr", "source", "freq"]
    ].drop_duplicates("series_id")
    valid = observations.dropna(subset=["value"]).copy()
    valid["period"] = pd.to_datetime(valid["period"])
    valid["value"] = valid["value"].astype("float64")

    duplicates = (
        observations.groupby("series_id")["period"].agg(lambda p: len(p) - p.nunique())
        if len(observations)
        else pd.Series(dtype=int)
    )
    grouped = valid.groupby("series_id")
    stats = pd.DataFrame(
        {
            "start": grouped["period"].min(),
            "end": grouped["period"].max(),
            "n": grouped["period"].nunique(),
            "last_updated": grouped["fetched_at"].max(),
            "weekdays_only": grouped["period"].agg(
                lambda p: bool((p.dt.weekday < 5).all())
            ),
        }
    )
    q1 = grouped["value"].quantile(0.25)
    q3 = grouped["value"].quantile(0.75)
    bounds = pd.DataFrame({"q1": q1, "q3": q3, "count": grouped["value"].size()})
    bounds["iqr"] = bounds["q3"] - bounds["q1"]
    joined = valid.join(bounds, on="series_id")
    outside = (joined["value"] < joined["q1"] - 1.5 * joined["iqr"]) | (
        joined["value"] > joined["q3"] + 1.5 * joined["iqr"]
    )
    usable = (joined["count"] >= 4) & (joined["iqr"] > 0)
    stats["outliers"] = (outside & usable).groupby(joined["series_id"]).sum()

    report = catalog.merge(stats, left_on="series_id", right_index=True, how="left")
    has_data = report["n"].notna() & (report["n"] > 0)
    expected = pd.Series(0, index=report.index)
    if has_data.any():
        expected[has_data] = _expected_count(
            report.loc[has_data, "freq"],
            report.loc[has_data, "start"],
            report.loc[has_data, "end"],
            report.loc[has_data, "weekdays_only"].astype(bool),
        )
    report["n"] = report["n"].fillna(0).astype(int)
    report["missing"] = (expected - report["n"]).clip(lower=0).astype(int)
    report["outliers"] = report["outliers"].fillna(0).astype(int)
    report["duplicates"] = report["series_id"].map(duplicates).fillna(0).astype(int)
    for column in ("start", "end"):
        report[column] = pd.Series(
            [
                stamp.date() if ok and pd.notna(stamp) else None
                for stamp, ok in zip(
                    pd.to_datetime(report[column]), has_data, strict=True
                )
            ],
            index=report.index,
            dtype=object,
        )
    report["last_updated"] = pd.Series(
        [
            v if ok else None
            for v, ok in zip(report["last_updated"], has_data, strict=True)
        ],
        index=report.index,
        dtype=object,
    )
    return report[QUALITY_COLUMNS].reset_index(drop=True)


def _guess_freq(periods: pd.Series) -> str | None:
    """Best-effort gold-contract Freq (D/W/M/Q/A) guess from the MINIMUM
    gap between sorted unique periods -- same idea as
    analysis/transforms.py's `_infer_freq_alias`, robust to an occasional
    missing period unlike `pd.infer_freq`. None if there isn't enough data
    to guess from."""
    unique_sorted = pd.DatetimeIndex(pd.to_datetime(pd.Series(periods).unique()))
    unique_sorted = unique_sorted.sort_values()
    if len(unique_sorted) < 2:
        return None
    min_days = int(pd.Series(unique_sorted).diff().dropna().dt.days.min())
    if min_days <= 1:
        return "D"
    if min_days == 7:
        return "W"
    if 28 <= min_days <= 31:
        return "M"
    if 89 <= min_days <= 92:
        return "Q"
    if 365 <= min_days <= 366:
        return "A"
    return None


def profile_table(df: pd.DataFrame, *, table_ref: str | None = None) -> QualityReport:
    """Quality profile for an ad-hoc "period"/"value" table arriving at
    demo time (get_series/read_url) -- not one of the project's own catalog
    series, so freq/unit are unknown rather than looked up. Thin wrapper:
    reuses this module's own IQR outlier rule and expected-calendar
    computation; row/duplicate/date-range counts are direct pandas ops.
    """
    rows = len(df)
    if rows == 0 or "period" not in df.columns or "value" not in df.columns:
        return QualityReport(
            table_ref=table_ref,
            rows=rows,
            issues=["boş veya period/value sütunu yok"],
        )

    valid = df.dropna(subset=["value"])
    duplicates = int(len(df) - df["period"].nunique())
    outliers = _iqr_outlier_count(valid["value"]) if not valid.empty else 0

    date_range = None
    freq = None
    missing_periods = 0
    issues: list[str] = []
    if not valid.empty:
        start, end = valid["period"].min(), valid["period"].max()
        date_range = (f"{start:%Y-%m}", f"{end:%Y-%m}")
        freq = _guess_freq(valid["period"])
        if freq is not None:
            expected = _expected_periods(freq, start, end)
            missing_periods = len(expected - set(valid["period"]))
        else:
            issues.append("frekans belirlenemedi; eksik dönem sayısı hesaplanamadı")
    else:
        issues.append("hiçbir gözlem yok (tümü NaN)")

    total = missing_periods + len(valid)
    return QualityReport(
        table_ref=table_ref,
        rows=rows,
        date_range=date_range,
        freq=freq,
        missing_periods=missing_periods,
        missing_pct=(missing_periods / total) if total else 0.0,
        duplicates=duplicates,
        outliers=outliers,
        unit_guess=None,
        issues=issues,
    )


def _fmt_date(value) -> str:
    return "-" if value is None or pd.isna(value) else str(value)


def render_quality_markdown(
    quality: pd.DataFrame, *, dropped_duplicates: int = 0
) -> str:
    """Pure DataFrame -> markdown string (no I/O), so content is testable
    independent of the file write.

    `dropped_duplicates`: total (series_id, period) rows silver's own
    dedup absorbed before gold ever saw them (plan 2.1). The per-series
    `duplicates` column below is computed from `observations` -- which is
    already post-dedup -- so it is structurally always 0; this is the real
    count, surfaced separately.
    """
    lines = [
        "# Veri Kalite Raporu",
        "",
        (
            "Otomatik üretildi: `python -m lakehouse_agent.lake.quality`. Kaynak: "
            "`data/gold/observations.parquet` + `data/gold/series_catalog.parquet`."
        ),
        "",
    ]

    if dropped_duplicates:
        lines += [
            f"Silver aşamasında birleştirilen mükerrer (aynı seri+dönem) "
            f"satır sayısı: {dropped_duplicates}.",
            "",
        ]

    flagged = quality[quality["missing"] > 0]
    if not flagged.empty:
        lines += ["## ⚠️ Eksik dönem uyarısı", ""]
        worst = flagged.sort_values("missing", ascending=False)
        for _, row in worst.head(DETAIL_ROWS_IN_REPORT).iterrows():
            lines.append(
                f"- **{row['series_id']}** ({row['name_tr']}): "
                f"{row['missing']} eksik dönem"
            )
        if len(worst) > DETAIL_ROWS_IN_REPORT:
            lines.append(
                f"- ... ve {len(worst) - DETAIL_ROWS_IN_REPORT} seri daha "
                f"(tamamı `{DETAIL_PATH.as_posix()}`)"
            )
        lines.append("")

    lines += ["## Kaynak bazında özet", ""]
    lines += ["| Kaynak | Seri sayısı | Toplam gözlem | Toplam eksik | Toplam aykırı |"]
    lines += ["|---|---|---|---|---|"]
    by_source = quality.groupby("source", dropna=False, sort=True).agg(
        series_count=("series_id", "count"),
        total_n=("n", "sum"),
        total_missing=("missing", "sum"),
        total_outliers=("outliers", "sum"),
    )
    for source, row in by_source.iterrows():
        lines.append(
            f"| {source} | {row['series_count']} | {row['total_n']} | "
            f"{row['total_missing']} | {row['total_outliers']} |"
        )
    lines.append("")

    lines += ["## Seri detayları", ""]
    if len(quality) > DETAIL_ROWS_IN_REPORT:
        lines += [
            f"{len(quality)} seri; seri başına satırlar `{DETAIL_PATH.as_posix()}` "
            "dosyasında (aynı sütunlar).",
            "",
        ]
        return "\n".join(lines)
    lines += ["| series_id | ad | frekans | ilk | son | n | eksik | tekrar | aykırı |"]
    lines += ["|---|---|---|---|---|---|---|---|---|"]
    for _, row in quality.sort_values("series_id").iterrows():
        lines.append(
            f"| {row['series_id']} | {row['name_tr']} | {row['freq']} | "
            f"{_fmt_date(row['start'])} | {_fmt_date(row['end'])} | {row['n']} | "
            f"{row['missing']} | {row['duplicates']} | {row['outliers']} |"
        )
    lines.append("")

    return "\n".join(lines)


def write_quality_report(
    quality: pd.DataFrame, *, out_path: Path = REPORT_PATH, dropped_duplicates: int = 0
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_quality_markdown(quality, dropped_duplicates=dropped_duplicates),
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    observations = pd.read_parquet(OBSERVATIONS_PATH)
    series_catalog = pd.read_parquet(SERIES_CATALOG_PATH)
    quality = build_quality_report(observations, series_catalog)

    dedup_stats_path = GOLD_ROOT / "dedup_stats.json"
    dropped_duplicates = 0
    if dedup_stats_path.exists():
        dropped_duplicates = json.loads(
            dedup_stats_path.read_text(encoding="utf-8")
        ).get("dropped_duplicates", 0)

    path = write_quality_report(quality, dropped_duplicates=dropped_duplicates)
    quality.astype({"start": "string", "end": "string"}).to_parquet(
        DETAIL_PATH, index=False
    )

    flagged = int((quality["missing"] > 0).sum())
    logger.info(
        "kalite raporu yazıldı: %s (%d seri, %d serinin eksik dönemi var)",
        path,
        len(quality),
        flagged,
    )
