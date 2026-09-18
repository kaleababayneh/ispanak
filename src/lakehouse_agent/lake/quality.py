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

import pandas as pd

logger = logging.getLogger(__name__)

GOLD_ROOT = Path("data/gold")
OBSERVATIONS_PATH = GOLD_ROOT / "observations.parquet"
SERIES_CATALOG_PATH = GOLD_ROOT / "series_catalog.parquet"
REPORT_PATH = Path("docs/data_quality.md")

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
    """Full calendar between `start` and `end` (a series' own first/last
    observed period), as `datetime.date` objects -- matching the dtype
    `observations.parquet`'s `period` column actually has, not `Timestamp`
    (see module docstring). Only the two frequencies this project's 9-series
    scenario uses are supported.
    """
    if freq.startswith("AYLIK"):
        return set(pd.date_range(start, end, freq="MS").date)
    if freq.startswith("HAFTALIK"):
        # Periods are already Monday-aligned by lake/silver.py's
        # `_period_for`, so a plain 7-day step is exact -- no weekday logic
        # needed here.
        return set(pd.date_range(start, end, freq="7D").date)
    raise ValueError(f"quality.py bu frekansı desteklemiyor: {freq!r}")


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


def _quality_row(series_id: str, cat_row: pd.Series, group: pd.DataFrame) -> dict:
    valid = group.dropna(subset=["value"])
    duplicates = int(len(group) - group["period"].nunique()) if len(group) else 0

    if valid.empty:
        return {
            "series_id": series_id,
            "name_tr": cat_row.get("name_tr"),
            "source": cat_row.get("source"),
            "freq": cat_row.get("freq"),
            "start": None,
            "end": None,
            "n": 0,
            "missing": 0,
            "duplicates": duplicates,
            "outliers": 0,
            "last_updated": None,
        }

    start = valid["period"].min()
    end = valid["period"].max()
    expected = _expected_periods(cat_row["freq"], start, end)
    observed = set(valid["period"])
    missing = len(expected - observed)

    return {
        "series_id": series_id,
        "name_tr": cat_row.get("name_tr"),
        "source": cat_row.get("source"),
        "freq": cat_row.get("freq"),
        "start": start,
        "end": end,
        "n": len(valid),
        "missing": missing,
        "duplicates": duplicates,
        "outliers": _iqr_outlier_count(valid["value"]),
        "last_updated": valid["fetched_at"].max(),
    }


def build_quality_report(
    observations: pd.DataFrame, series_catalog: pd.DataFrame
) -> pd.DataFrame:
    """One row per `series_catalog` series_id, with coverage/missing/
    duplicate/outlier counts computed from `observations`. A series with no
    non-null observations at all (not expected for this project's 9 series,
    but not an error either) reports n=0/missing=0 rather than raising.
    """
    rows = [
        _quality_row(
            cat_row["series_id"],
            cat_row,
            observations[observations["series_id"] == cat_row["series_id"]],
        )
        for _, cat_row in series_catalog.iterrows()
    ]
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def _fmt_date(value) -> str:
    return "-" if value is None or pd.isna(value) else str(value)


def render_quality_markdown(quality: pd.DataFrame) -> str:
    """Pure DataFrame -> markdown string (no I/O), so content is testable
    independent of the file write.
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

    flagged = quality[quality["missing"] > 0]
    if not flagged.empty:
        lines += ["## ⚠️ Eksik dönem uyarısı", ""]
        for _, row in flagged.sort_values("missing", ascending=False).iterrows():
            lines.append(
                f"- **{row['series_id']}** ({row['name_tr']}): "
                f"{row['missing']} eksik dönem"
            )
        lines.append("")

    lines += ["## Kaynak bazında özet", ""]
    lines += ["| Kaynak | Seri sayısı | Toplam gözlem | Toplam eksik | Toplam aykırı |"]
    lines += ["|---|---|---|---|---|"]
    by_source = quality.groupby("source", dropna=False).agg(
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
    quality: pd.DataFrame, *, out_path: Path = REPORT_PATH
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_quality_markdown(quality), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    observations = pd.read_parquet(OBSERVATIONS_PATH)
    series_catalog = pd.read_parquet(SERIES_CATALOG_PATH)
    quality = build_quality_report(observations, series_catalog)
    path = write_quality_report(quality)

    flagged = int((quality["missing"] > 0).sum())
    logger.info(
        "kalite raporu yazıldı: %s (%d seri, %d serinin eksik dönemi var)",
        path,
        len(quality),
        flagged,
    )
