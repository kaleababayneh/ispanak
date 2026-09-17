"""Bronze -> silver (long, one row per series x period).

EVDS: bronze is wide (one parquet per datagroup x year). Reuses
`ingest.evds._parse_tarih` and `denormalize_response_columns` rather than
reimplementing tarih parsing or the dot/underscore column reversal -- those
are owned by ingest/evds.py.

BDDK: bronze is already one row per (series_id, month) with a raw `cell_json`
array (see ingest/bddk_monthly.py); silver picks the value out by
`cell_index` and only then applies Turkish number-format conversion.
"""

import json
import re
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from lakehouse_agent.ingest import bddk_monthly
from lakehouse_agent.ingest.evds import (
    CATALOG_PATH,
    DATAGROUPS_DATA_ROOT,
    _parse_tarih,
    denormalize_response_columns,
)

# Debug/inspection artifact only -- NOT part of the CLAUDE.md gold contract.
# gold.py never reads this path; it calls build_silver_observations() /
# build_bddk_silver_observations() directly, in memory. This exists purely
# so `python -m lakehouse_agent.lake.silver` produces something visible on
# disk to sanity-check bronze -> silver before running gold.
SILVER_DEBUG_PATH = Path("data/silver/observations.parquet")

OBSERVATION_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
]

_MISSING_TOKENS = {"", "-", "n.a.", "n.a", "na", "NA"}


def _to_float(raw: object) -> float:
    """ "-"/""/"n.a." -> NaN. Never filled with 0 or interpolated (CLAUDE.md #2)."""
    if raw is None:
        return float("nan")
    if isinstance(raw, int | float):
        return float(raw)
    text = str(raw).strip()
    if text in _MISSING_TOKENS:
        return float("nan")
    return float(text)


def _period_for(tarih: str, freq_str: str) -> date:
    """Period = the first day of the period. Weekly Tarih values are the
    anchor weekday (e.g. Friday for HAFTALIK(CUMA)) -- rounded down to that
    week's Monday. Monthly/quarterly/yearly Tarih values already parse to
    the first day of their period.
    """
    d = _parse_tarih(str(tarih))
    if freq_str.upper().startswith("HAFTALIK"):
        return d - timedelta(days=d.weekday())
    return d


def melt_bronze_datagroup(
    bronze: pd.DataFrame,
    serie_codes: Iterable[str],
    freq_str: str,
    unit: str | None,
    *,
    source: str = "evds",
) -> pd.DataFrame:
    """One bronze wide datagroup/year DataFrame -> long observation rows for
    the given serie_codes (raw, undotted EVDS SERIE_CODE form). Columns of
    `bronze` not in serie_codes (other series in the same datagroup, or
    "Tarih"/"_fetched_at") are ignored/carried as id columns.
    """
    serie_codes = list(serie_codes)
    denormalized = denormalize_response_columns(bronze, serie_codes)
    value_cols = [c for c in serie_codes if c in denormalized.columns]
    if not value_cols:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    id_cols = ["Tarih"]
    has_fetched_at = "_fetched_at" in denormalized.columns
    if has_fetched_at:
        id_cols.append("_fetched_at")

    long = denormalized.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="_code",
        value_name="_raw_value",
    )
    long["period"] = long["Tarih"].map(lambda t: _period_for(t, freq_str))
    long["series_id"] = source + ":" + long["_code"]
    long["value"] = long["_raw_value"].map(_to_float)
    long["freq"] = freq_str
    long["unit"] = unit
    long["source"] = source
    long["fetched_at"] = long["_fetched_at"] if has_fetched_at else None

    return long[OBSERVATION_COLUMNS]


def _dedupe_latest_fetch(long: pd.DataFrame) -> pd.DataFrame:
    """Same (series_id, period) fetched more than once (e.g. a week
    straddling two year-chunk requests) -> keep the row with the latest
    fetched_at."""
    long = long.sort_values("fetched_at")
    long = long.drop_duplicates(subset=["series_id", "period"], keep="last")
    return long.sort_values(["series_id", "period"]).reset_index(drop=True)


def build_silver_observations(
    series_ids: Iterable[str], *, source: str = "evds"
) -> pd.DataFrame:
    """series_ids in "evds:SERIE.CODE" form. Looks up each series'
    DATAGROUP_CODE / FREQUENCY_STR / BIRIMI in the bronze catalog, melts
    every cached bronze year file for that datagroup, and returns one long
    frame matching the observations.parquet contract (kind/cum_type are not
    catalog fields -- gold.py attaches those from the scenario mapping).
    """
    prefix = f"{source}:"
    raw_codes = {sid[len(prefix) :] for sid in series_ids if sid.startswith(prefix)}
    if not raw_codes:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    catalog = pd.read_parquet(CATALOG_PATH)
    catalog = catalog[catalog["SERIE_CODE"].isin(raw_codes)]

    frames = []
    for datagroup_code, group in catalog.groupby("DATAGROUP_CODE"):
        serie_codes = group["SERIE_CODE"].tolist()
        freq_str = group["FREQUENCY_STR"].iloc[0]
        unit = group["BIRIMI"].iloc[0] if "BIRIMI" in group.columns else None

        year_dir = DATAGROUPS_DATA_ROOT / datagroup_code
        for path in sorted(year_dir.glob("*.parquet")):
            bronze = pd.read_parquet(path)
            frames.append(
                melt_bronze_datagroup(
                    bronze, serie_codes, freq_str, unit, source=source
                )
            )

    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    long = pd.concat(frames, ignore_index=True)
    return _dedupe_latest_fetch(long)


# --- BDDK -----------------------------------------------------------------

BDDK_SOURCE = "BDDK-Aylik"  # gold series_catalog contract value for this source

_TURKISH_NUMBER_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$")


def _to_float_bddk(raw: object) -> float:
    """Same missing-token handling as `_to_float`, plus Turkish thousand/decimal
    separator conversion ("1.234,5" -> 1234.5) when the value arrives as text.
    A value that already arrives as a native JSON number is never touched
    (CLAUDE.md #3: "gelmiyorsa dokunma").
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, int | float):
        return float(raw)
    text = str(raw).strip()
    if text in _MISSING_TOKENS:
        return float("nan")
    if _TURKISH_NUMBER_RE.match(text):
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def build_bddk_silver_observations(series_ids: Iterable[str]) -> pd.DataFrame:
    """series_ids in "bddk:..." form. Reads every cached bronze month file
    (ingest/bddk_monthly.py), picks each row's value out of its raw
    `cell_json` by the cell_index declared in ingest/bddk_rows.yaml, and
    returns one long frame matching the observations.parquet contract.
    """
    wanted = {sid for sid in series_ids if sid.startswith("bddk:")}
    if not wanted:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    row_defs = {rd.series_id: rd for rd in bddk_monthly.load_row_definitions()}

    frames = []
    for path in sorted(bddk_monthly.BRONZE_ROOT.glob("*.parquet")):
        bronze = pd.read_parquet(path)
        bronze = bronze[bronze["series_id"].isin(wanted)]
        if bronze.empty:
            continue

        rows = []
        for _, r in bronze.iterrows():
            row_def = row_defs.get(r["series_id"])
            if row_def is None:
                continue
            cell = json.loads(r["cell_json"])
            rows.append(
                {
                    "series_id": r["series_id"],
                    "period": date(int(r["yil"]), int(r["ay"]), 1),
                    "freq": "AYLIK",
                    "value": _to_float_bddk(cell[row_def.cell_index]),
                    "unit": row_def.unit,
                    "source": BDDK_SOURCE,
                    "fetched_at": r["fetched_at"],
                }
            )
        frames.append(pd.DataFrame(rows, columns=OBSERVATION_COLUMNS))

    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    long = pd.concat(frames, ignore_index=True)
    return _dedupe_latest_fetch(long)


if __name__ == "__main__":
    import logging

    from lakehouse_agent.lake.gold import default_series_ids

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    all_ids = default_series_ids()
    evds_ids = [sid for sid in all_ids if sid.startswith("evds:")]
    bddk_ids = [sid for sid in all_ids if sid.startswith("bddk:")]

    combined = pd.concat(
        [
            build_silver_observations(evds_ids, source="evds"),
            build_bddk_silver_observations(bddk_ids),
        ],
        ignore_index=True,
    )

    SILVER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SILVER_DEBUG_PATH, index=False)
    logger.info(
        "silver (debug) yazıldı: %s (%d satır, evds=%d bddk=%d seri)",
        SILVER_DEBUG_PATH,
        len(combined),
        len(evds_ids),
        len(bddk_ids),
    )
