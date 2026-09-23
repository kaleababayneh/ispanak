"""Bronze -> silver (long, one row per series x period).

EVDS: bronze is wide (one parquet per datagroup x year). Reuses
`ingest.evds._parse_tarih` and `denormalize_response_columns` rather than
reimplementing tarih parsing or the dot/underscore column reversal -- those
are owned by ingest/evds.py.

BDDK: lake/bddk.py reads the bulletin's table files; the Turkish
number-format conversion it applies (`_to_float_bddk`) lives here with the
other value rules.
"""

import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from lakehouse_agent.ingest.evds import (
    CATALOG_PATH,
    DATAGROUPS_DATA_ROOT,
    _parse_tarih,
    denormalize_response_columns,
)

logger = logging.getLogger(__name__)

# Debug/inspection artifact only -- NOT part of the CLAUDE.md gold contract.
# gold.py never reads this path; it streams `iter_evds_silver()` and
# `lake.bddk.build_monthly()` directly. This exists purely so
# `python -m lakehouse_agent.lake.silver` produces something visible on disk
# to sanity-check bronze -> silver for the scenario series before gold.
SILVER_DEBUG_PATH = Path("data/silver/observations.parquet")

OBSERVATION_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
    "obs_date",
]

_MISSING_TOKENS = {"", "-", "n.a.", "n.a", "na", "NA"}

# Raw EVDS/BDDK FREQUENCY_STR -> the gold contract's Freq literal
# (D|W|M|Q|A, contracts.py). HAFTALIK(*) (CUMA/ÇARŞAMBA anchor) is always W.
FREQ_STR_TO_CONTRACT = {
    "AYLIK": "M",
    "GÜNLÜK": "D",
    "İŞ GÜNÜ": "D",
    "ÜÇ AYLIK": "Q",
    "YILLIK": "A",
}


def map_freq(freq_str: str) -> str:
    """Map a raw EVDS/BDDK FREQUENCY_STR to the gold contract's Freq literal.
    Used by both observations (this module) and series_catalog (gold.py) so
    the two tables never disagree on what a given frequency is called."""
    upper = freq_str.strip().upper()
    if upper.startswith("HAFTALIK"):
        return "W"
    try:
        return FREQ_STR_TO_CONTRACT[upper]
    except KeyError:
        raise ValueError(f"Bilinmeyen frekans: {freq_str!r}") from None


def _to_float(raw: object) -> float:
    """ "-"/""/"n.a." -> NaN. Never filled with 0 or interpolated (CLAUDE.md #2).
    A token this doesn't recognize (typo, new EVDS missing-value spelling)
    also becomes NaN, logged as a warning -- never crashes the whole ingest
    run over one bad cell."""
    if raw is None:
        return float("nan")
    if isinstance(raw, int | float):
        return float(raw)
    text = str(raw).strip()
    if text in _MISSING_TOKENS:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        logger.warning("_to_float: tanınmayan jeton NaN'a düşürüldü: %r", raw)
        return float("nan")


def _to_float_series(raw: pd.Series) -> pd.Series:
    """`_to_float` for a whole column at once: the pool is millions of cells,
    and a Python call per cell was most of silver's time. Same rules: the
    missing tokens are NaN, an unrecognised token is NaN and logged."""
    if pd.api.types.is_numeric_dtype(raw):
        return raw.astype("float64")
    text = raw.astype("string").str.strip()
    missing = text.isna() | text.isin(_MISSING_TOKENS)
    values = pd.to_numeric(text.where(~missing), errors="coerce")
    unknown = values.isna() & ~missing
    if unknown.any():
        logger.warning(
            "_to_float: %d tanınmayan jeton NaN'a düşürüldü (ör. %r)",
            int(unknown.sum()),
            raw[unknown].iloc[0],
        )
    return values.astype("float64")


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
    unit: str | None | Mapping[str, str | None],
    *,
    source: str = "evds",
) -> pd.DataFrame:
    """One bronze wide datagroup/year DataFrame -> long observation rows for
    the given serie_codes (raw, undotted EVDS SERIE_CODE form). Columns of
    `bronze` not in serie_codes (other series in the same datagroup, or
    "Tarih"/"_fetched_at") are ignored/carried as id columns.

    `unit` is one unit for the whole group (EVDS's BIRIMI is per datagroup)
    or a mapping per SERIE_CODE (gold's derived units, lake/classify.py).
    `obs_date` is the date EVDS attached to the value; `period` is the first
    day of its period, which for a weekly series is that week's Monday.
    Both are computed once per distinct "Tarih", not once per cell.
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
    tarihler = long["Tarih"].astype(str).unique()
    observed = {tarih: _parse_tarih(tarih) for tarih in tarihler}
    periods = {tarih: _period_for(tarih, freq_str) for tarih in tarihler}
    long["obs_date"] = long["Tarih"].astype(str).map(observed)
    long["period"] = long["Tarih"].astype(str).map(periods)
    long["series_id"] = source + ":" + long["_code"]
    long["value"] = _to_float_series(long["_raw_value"])
    long["freq"] = map_freq(freq_str)
    if isinstance(unit, Mapping):
        long["unit"] = long["_code"].map(unit)
    else:
        long["unit"] = unit
    long["source"] = source.upper()
    long["fetched_at"] = long["_fetched_at"] if has_fetched_at else None

    return long[OBSERVATION_COLUMNS]


def _dedupe_latest_fetch(long: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Same (series_id, period) fetched more than once (e.g. a week
    straddling two year-chunk requests) -> keep the row with the latest
    fetched_at.

    Returns (deduped, dropped_count). Plan 2.1 wants the dropped count
    surfaced in the quality report; previously it was silently absorbed
    here, before gold ever saw it -- gold's own duplicate count was
    therefore structurally always 0, not a real "no duplicates" signal.
    """
    before = len(long)
    long = long.sort_values("fetched_at", kind="stable")
    long = long.drop_duplicates(subset=["series_id", "period"], keep="last")
    deduped = long.sort_values(["series_id", "period"]).reset_index(drop=True)
    return deduped, before - len(deduped)


def build_silver_observations(
    series_ids: Iterable[str], *, source: str = "evds"
) -> tuple[pd.DataFrame, int]:
    """series_ids in "evds:SERIE.CODE" form. Looks up each series'
    DATAGROUP_CODE / FREQUENCY_STR / BIRIMI in the bronze catalog, melts
    every cached bronze year file for that datagroup, and returns
    (long frame matching the observations.parquet contract, dropped
    duplicate row count) -- kind/cum_type are not catalog fields, gold.py
    attaches those from the scenario mapping.
    """
    prefix = f"{source}:"
    raw_codes = {sid[len(prefix) :] for sid in series_ids if sid.startswith(prefix)}
    if not raw_codes:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS), 0

    catalog = pd.read_parquet(CATALOG_PATH)
    catalog = catalog[catalog["SERIE_CODE"].isin(raw_codes)]

    frames, dropped = [], 0
    for frame, dropped_here in iter_evds_silver(catalog, source=source):
        frames.append(frame)
        dropped += dropped_here
    if not frames:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS), 0
    return pd.concat(frames, ignore_index=True), dropped


def iter_evds_silver(
    catalog: pd.DataFrame,
    *,
    units: Mapping[str, str | None] | None = None,
    source: str = "evds",
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Silver observations one datagroup at a time, deduplicated, with the
    count of duplicate rows each one dropped.

    `catalog` is bronze catalog rows (one per series). One datagroup at a
    time because the full pool is millions of rows: gold writes each group
    as it comes instead of holding them all. A series belongs to exactly one
    datagroup, so deduplicating per group is deduplicating per series.
    `units` maps SERIE_CODE -> unit (gold's derived units); without it each
    group gets its catalog BIRIMI, as before.
    """
    for datagroup_code, group in catalog.groupby("DATAGROUP_CODE", sort=True):
        files = sorted((DATAGROUPS_DATA_ROOT / datagroup_code).glob("*.parquet"))
        if not files:
            continue
        serie_codes = group["SERIE_CODE"].tolist()
        freq_str = group["FREQUENCY_STR"].iloc[0]
        if units is not None:
            unit: str | None | Mapping[str, str | None] = {
                code: units.get(code) for code in serie_codes
            }
        else:
            unit = group["BIRIMI"].iloc[0] if "BIRIMI" in group.columns else None
        frames = [
            melt_bronze_datagroup(
                pd.read_parquet(path), serie_codes, freq_str, unit, source=source
            )
            for path in files
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            continue
        yield _dedupe_latest_fetch(pd.concat(frames, ignore_index=True))


# --- BDDK -----------------------------------------------------------------

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
    try:
        return float(text)
    except ValueError:
        logger.warning("_to_float_bddk: tanınmayan jeton NaN'a düşürüldü: %r", raw)
        return float("nan")


if __name__ == "__main__":
    from lakehouse_agent.lake.bddk import build_monthly
    from lakehouse_agent.lake.gold import SCENARIO_SERIES

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    evds_obs, evds_dropped = build_silver_observations(SCENARIO_SERIES, source="evds")
    bddk_obs, _catalog, bddk_dropped = build_monthly()
    combined = pd.concat([evds_obs, bddk_obs], ignore_index=True)

    SILVER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SILVER_DEBUG_PATH, index=False)
    logger.info(
        "silver (debug) yazıldı: %s (%d satır, %d mükerrer birleştirildi)",
        SILVER_DEBUG_PATH,
        len(combined),
        evds_dropped + bddk_dropped,
    )
