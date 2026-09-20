"""Gold layer: observations.parquet + series_catalog.parquet + a read-only
DuckDB view over both.

Column names/order here are the CLAUDE.md contract other tracks code
against -- do not rename or reorder without updating docs/contracts.md and
asking first.

The single entry point into this layer is `get()`/`wide()` -- do not read
`observations.parquet`/`series_catalog.parquet` directly with
`pd.read_parquet` and try to merge that against `get()`'s output on
"period": parquet's own `period` column comes back as `object`/
`datetime.date`, while `get()`/`wide()` (via DuckDB) return `datetime64[us]`.
Merging the two silently matches ZERO rows (pandas raises no error joining
across dtypes, it just finds no matches) -- see
`test_reading_parquet_directly_yields_different_period_dtype_than_get`.
"""

import calendar
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from lakehouse_agent.ingest import bddk_monthly
from lakehouse_agent.ingest.evds import CATALOG_PATH
from lakehouse_agent.lake.silver import (
    BDDK_SOURCE,
    build_bddk_silver_observations,
    build_silver_observations,
    map_freq,
)

logger = logging.getLogger(__name__)

BDDK_METADATA_URL = "https://www.bddk.org.tr/BultenAylik/"

GOLD_ROOT = Path("data/gold")
OBSERVATIONS_PATH = GOLD_ROOT / "observations.parquet"
SERIES_CATALOG_PATH = GOLD_ROOT / "series_catalog.parquet"
GOLD_DB_PATH = GOLD_ROOT / "catalog.duckdb"
DEDUP_STATS_PATH = GOLD_ROOT / "dedup_stats.json"

OBSERVATIONS_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
]
SERIES_CATALOG_COLUMNS = [
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

# Senaryo serileri (16 Eylül entegrasyon) -- kind/cum_type EVDS katalogunda
# yok, burada elle eşlenir. Şema değişikliği gerekirse docs/contracts.md
# güncellenir.
SCENARIO_SERIES: dict[str, dict] = {
    "evds:TP.KTF12": {"kind": "rate", "agg_method": "avg", "cum_type": "none"},
    "evds:TP.KFE.TR": {"kind": "index", "agg_method": "avg", "cum_type": "none"},
    "evds:TP.TUKFIY2025.GENEL": {
        "kind": "index",
        "agg_method": "avg",
        "cum_type": "none",
    },
    "evds:TP.AKONUTSAT1.KTRTOPLAM": {
        "kind": "flow",
        "agg_method": "sum",
        "cum_type": "none",
    },
    "evds:TP.AKONUTSAT2.KTRTOPLAM": {
        "kind": "flow",
        "agg_method": "sum",
        "cum_type": "none",
    },
    "evds:TP.HPBITABLO6.3": {"kind": "stock", "agg_method": "last", "cum_type": "none"},
}


def build_series_catalog(
    series_ids: Iterable[str], *, source: str = "evds"
) -> pd.DataFrame:
    """series_catalog.parquet rows for the given series_ids, joined against
    the EVDS bronze catalog for name/freq/unit/dates/topic/tags, with
    kind/agg_method/cum_type filled in from SCENARIO_SERIES (not an EVDS
    catalog field).
    """
    prefix = f"{source}:"
    raw_codes = {sid[len(prefix) :] for sid in series_ids if sid.startswith(prefix)}
    bronze_catalog = pd.read_parquet(CATALOG_PATH)
    bronze_catalog = bronze_catalog[bronze_catalog["SERIE_CODE"].isin(raw_codes)]

    rows = []
    for _, row in bronze_catalog.iterrows():
        series_id = f"{source}:{row['SERIE_CODE']}"
        mapping = SCENARIO_SERIES.get(series_id)
        if mapping is None:
            raise KeyError(
                f"{series_id} için kind/cum_type eşlemesi yok (SCENARIO_SERIES)"
            )
        rows.append(
            {
                "series_id": series_id,
                "name_tr": row.get("SERIE_NAME"),
                "name_en": row.get("SERIE_NAME_ENG"),
                "source": source.upper(),
                "freq": map_freq(row.get("FREQUENCY_STR")),
                "unit": row.get("BIRIMI"),
                "kind": mapping["kind"],
                "agg_method": mapping["agg_method"],
                "cum_type": mapping["cum_type"],
                "start_date": row.get("START_DATE"),
                "end_date": row.get("END_DATE"),
                "topic": row.get("TOPIC_TITLE_TR"),
                "tags": row.get("TAG"),
                "metadata_url": row.get("METADATA_LINK"),
                "notes": row.get("NOTE"),
            }
        )
    return pd.DataFrame(rows, columns=SERIES_CATALOG_COLUMNS)


def build_bddk_series_catalog(series_ids: Iterable[str]) -> pd.DataFrame:
    """series_catalog.parquet rows for BDDK series_ids, sourced directly from
    ingest/bddk_rows.yaml -- there is no BDDK equivalent of the EVDS bronze
    catalog to join against, the yaml mapping IS the metadata.
    """
    wanted = {sid for sid in series_ids if sid.startswith("bddk:")}
    row_defs = {rd.series_id: rd for rd in bddk_monthly.load_row_definitions()}

    rows = []
    for series_id in wanted:
        row_def = row_defs.get(series_id)
        if row_def is None:
            raise KeyError(f"{series_id} için ingest/bddk_rows.yaml'da eşleme yok")
        rows.append(
            {
                "series_id": series_id,
                "name_tr": row_def.satir_etiketi,
                "name_en": None,
                "source": BDDK_SOURCE,
                "freq": map_freq("AYLIK"),
                "unit": row_def.unit,
                "kind": row_def.kind,
                "agg_method": "last" if row_def.kind == "stock" else None,
                "cum_type": row_def.cum_type,
                "start_date": None,
                "end_date": None,
                "topic": None,
                "tags": None,
                "metadata_url": BDDK_METADATA_URL,
                "notes": None,
            }
        )
    return pd.DataFrame(rows, columns=SERIES_CATALOG_COLUMNS)


def default_series_ids() -> list[str]:
    """Full known series universe: every EVDS SCENARIO_SERIES key + every
    BDDK bddk_rows.yaml row. Used by the `python -m lakehouse_agent.lake.gold`
    (and `.silver`) CLI convenience entrypoints so both process EVDS and
    BDDK together instead of silently defaulting to EVDS-only.
    """
    return list(SCENARIO_SERIES.keys()) + [
        row_def.series_id for row_def in bddk_monthly.load_row_definitions()
    ]


def _validate_columns(df: pd.DataFrame, expected: list[str], *, label: str) -> None:
    """`df[expected]`/`.reindex(columns=expected)` silently turns a missing
    or renamed source column into an all-NaN column with no error at all
    (measured: silver writing "frekans" instead of "freq" produced a
    completely NaN freq column, no exception). Fail loudly instead, naming
    exactly what's wrong.
    """
    actual = set(df.columns)
    missing = set(expected) - actual
    extra = actual - set(expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"eksik sütun(lar): {sorted(missing)}")
        if extra:
            parts.append(f"fazla sütun(lar): {sorted(extra)}")
        raise ValueError(f"{label}: şema uyuşmazlığı -- {', '.join(parts)}")


def build_gold(series_ids: Iterable[str]) -> tuple[Path, Path]:
    """Assemble silver -> observations.parquet + series_catalog.parquet.

    series_ids may mix "evds:" and "bddk:" prefixes -- each source's silver
    + catalog builder runs independently and the results are concatenated.
    """
    series_ids = list(series_ids)
    GOLD_ROOT.mkdir(parents=True, exist_ok=True)

    evds_ids = [sid for sid in series_ids if sid.startswith("evds:")]
    bddk_ids = [sid for sid in series_ids if sid.startswith("bddk:")]

    obs_frames = []
    catalog_frames = []
    dropped_duplicates = 0
    if evds_ids:
        obs, dropped = build_silver_observations(evds_ids, source="evds")
        obs_frames.append(obs)
        dropped_duplicates += dropped
        catalog_frames.append(build_series_catalog(evds_ids, source="evds"))
    if bddk_ids:
        obs, dropped = build_bddk_silver_observations(bddk_ids)
        obs_frames.append(obs)
        dropped_duplicates += dropped
        catalog_frames.append(build_bddk_series_catalog(bddk_ids))

    observations = (
        pd.concat(obs_frames, ignore_index=True)
        if obs_frames
        else pd.DataFrame(columns=OBSERVATIONS_COLUMNS)
    )
    _validate_columns(observations, OBSERVATIONS_COLUMNS, label="observations")
    observations = observations[OBSERVATIONS_COLUMNS]
    observations.to_parquet(OBSERVATIONS_PATH, index=False)

    catalog = (
        pd.concat(catalog_frames, ignore_index=True)
        if catalog_frames
        else pd.DataFrame(columns=SERIES_CATALOG_COLUMNS)
    )
    _validate_columns(catalog, SERIES_CATALOG_COLUMNS, label="series_catalog")
    catalog = catalog[SERIES_CATALOG_COLUMNS]
    catalog.to_parquet(SERIES_CATALOG_PATH, index=False)

    DEDUP_STATS_PATH.write_text(
        json.dumps({"dropped_duplicates": dropped_duplicates}), encoding="utf-8"
    )

    if GOLD_DB_PATH.exists():
        GOLD_DB_PATH.unlink()  # views embed the parquet path; rebuild on every gold run

    return OBSERVATIONS_PATH, SERIES_CATALOG_PATH


def _ensure_tables(db_path: Path = GOLD_DB_PATH) -> None:
    """(Re)materialize the two parquet files as native tables inside the
    DuckDB file itself, not as views over external parquet paths.

    A view over read_parquet(...) would need external file access at query
    time -- which is exactly what enable_external_access=false forbids on
    the read-only connection `connect()` hands out. Copying the bytes into
    the .duckdb file at build time (this writable connection, default
    external access) means later read-only queries never need to touch the
    filesystem beyond that one file.
    """
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE OR REPLACE TABLE observations AS "
            f"SELECT * FROM read_parquet('{OBSERVATIONS_PATH.as_posix()}')"
        )
        con.execute(
            "CREATE OR REPLACE TABLE series_catalog AS "
            f"SELECT * FROM read_parquet('{SERIES_CATALOG_PATH.as_posix()}')"
        )
    finally:
        con.close()


def connect() -> duckdb.DuckDBPyConnection:
    """Read-only DuckDB connection over the `observations` / `series_catalog`
    tables.

    enable_external_access=false is the actual control (CLAUDE.md): even a
    SELECT-only query can otherwise reach read_csv('/etc/passwd'), ATTACH,
    or INSTALL/LOAD httpfs and exfiltrate data -- a keyword blacklist alone
    does not stop that. Because the two tables are materialized inside
    GOLD_DB_PATH (see _ensure_tables), ordinary SELECTs against them still
    work with external access fully disabled.
    """
    if not GOLD_DB_PATH.exists():
        _ensure_tables()
    return duckdb.connect(
        str(GOLD_DB_PATH), read_only=True, config={"enable_external_access": "false"}
    )


_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _normalize_start(value: str) -> str:
    """ "YYYY-MM" -> that month's first day; "YYYY-MM-DD" passes through."""
    if _YEAR_MONTH_RE.match(value):
        return f"{value}-01"
    return value


def _normalize_end(value: str) -> str:
    """ "YYYY-MM" -> that month's LAST day (not the 1st -- docs/contracts.md's
    `get(..., end="2025-12")` must keep all of December, not drop it: DuckDB
    parses "2025-12" as 2025-12-01, so `period <= "2025-12"` would silently
    exclude every day after the 1st). "YYYY-MM-DD" passes through."""
    if _YEAR_MONTH_RE.match(value):
        year, month = (int(part) for part in value.split("-"))
        last_day = calendar.monthrange(year, month)[1]
        return f"{value}-{last_day:02d}"
    return value


def get(
    series_ids: Iterable[str], start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Long observations for series_ids, optionally restricted to [start, end]
    (period). `start`/`end` accept "YYYY-MM" (docs/contracts.md's signature)
    as well as "YYYY-MM-DD" -- a bare month is normalized to that month's
    first/last day respectively before hitting DuckDB, which otherwise
    parses "YYYY-MM" as day 1 and (for `end`) silently drops the rest of
    that month.
    """
    ids = list(series_ids)
    if not ids:
        return pd.DataFrame(columns=OBSERVATIONS_COLUMNS)

    placeholders = ", ".join(["?"] * len(ids))
    clauses = [f"series_id IN ({placeholders})"]
    params: list = list(ids)
    if start is not None:
        clauses.append("period >= ?")
        params.append(_normalize_start(start))
    if end is not None:
        clauses.append("period <= ?")
        params.append(_normalize_end(end))
    where_sql = " AND ".join(clauses)

    con = connect()
    try:
        return con.execute(
            f"SELECT * FROM observations WHERE {where_sql} ORDER BY series_id, period",
            params,
        ).df()
    finally:
        con.close()


def _series_freqs(series_ids: list[str]) -> dict[str, str]:
    """freq per series_id from series_catalog, for wide()'s mixed-frequency
    diagnostic. Empty dict if series_ids is empty; callers should not fail
    just because this diagnostic couldn't run."""
    if not series_ids:
        return {}
    placeholders = ", ".join(["?"] * len(series_ids))
    con = connect()
    try:
        rows = con.execute(
            f"SELECT series_id, freq FROM series_catalog WHERE series_id IN ({placeholders})",
            list(series_ids),
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


def wide(
    series_ids: Iterable[str], start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """get(), pivoted to one column per series_id, indexed by period.

    Series with different native frequencies rarely share exact period
    values (a Monday-anchored weekly series vs. a 1st-of-month monthly
    series) -- pivoting them together produces a table where almost every
    row has only one column populated, unusable for a joint read even
    though it succeeds without error. Measured:
    wide(["evds:TP.KTF12", "evds:TP.TUKFIY2025.GENEL"], "2021-01",
    "2025-12") -> 311 rows, no row with both columns filled (weekly vs.
    monthly). The shape returned here is unchanged (other callers may
    depend on it); a mismatch and the per-column fill rate are logged
    instead so the problem is visible rather than a silent unusable table.
    """
    ids = list(series_ids)
    long = get(ids, start, end)
    if long.empty:
        return long

    result = long.pivot(
        index="period", columns="series_id", values="value"
    ).reset_index()

    value_cols = [c for c in result.columns if c != "period"]
    if len(value_cols) > 1:
        freqs = _series_freqs(value_cols)
        both_filled = int(result[value_cols].notna().all(axis=1).sum())
        fill_counts = {c: int(result[c].notna().sum()) for c in value_cols}
        if len(set(freqs.values()) - {None}) > 1:
            logger.warning(
                "wide(): istenen seriler farklı frekanslarda (%s) -- %d/%d "
                "satırda tüm seriler birlikte dolu. Sütun başına dolu satır: %s",
                freqs,
                both_filled,
                len(result),
                fill_counts,
            )
        elif both_filled < len(result):
            logger.info(
                "wide(): %d/%d satırda tüm seriler birlikte dolu. Sütun "
                "başına dolu satır: %s",
                both_filled,
                len(result),
                fill_counts,
            )
    return result


def run_sql(
    sql: str, params: Iterable | None = None, *, limit: int = 1000
) -> pd.DataFrame:
    """Execute a read-only SELECT (or WITH ... SELECT) against the gold views.

    Only a single statement is accepted (no ";"-chaining). The query is
    always wrapped as `SELECT * FROM (<query>) LIMIT n` -- unconditionally,
    never based on checking whether "limit" appears in the text: that
    substring check used to false-positive on a literal like
    `WHERE unit <> 'limitless'` and skip appending LIMIT entirely, letting
    the query return unbounded (measured). The wrap works whether or not
    the inner query has its own LIMIT, and whether or not it's a WITH/CTE
    (verified against DuckDB). This is defense in depth, not the real
    control -- `connect()`'s enable_external_access=false is what actually
    stops file-read/exfiltration tricks (CLAUDE.md); that stays untouched.
    """
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("run_sql: birden fazla ifade (;) kabul edilmez")
    if not stripped.lower().startswith(("select", "with")):
        raise ValueError("run_sql: sadece SELECT (veya WITH ... SELECT) kabul edilir")

    wrapped = f"SELECT * FROM ({stripped}) AS _run_sql_limited LIMIT {limit}"

    con = connect()
    try:
        return con.execute(wrapped, list(params) if params else []).df()
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ids = default_series_ids()
    obs_path, catalog_path = build_gold(ids)
    logger.info(
        "gold yazıldı: %s, %s (%d seri: evds=%d bddk=%d)",
        obs_path,
        catalog_path,
        len(ids),
        sum(1 for i in ids if i.startswith("evds:")),
        sum(1 for i in ids if i.startswith("bddk:")),
    )
