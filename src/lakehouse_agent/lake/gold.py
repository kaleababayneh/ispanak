"""Gold layer: observations.parquet + series_catalog.parquet + a read-only
DuckDB view over both.

Column names/order here are the CLAUDE.md contract other tracks code
against -- do not rename or reorder without updating docs/contracts.md and
asking first.
"""

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
)

BDDK_METADATA_URL = "https://www.bddk.org.tr/BultenAylik/"

GOLD_ROOT = Path("data/gold")
OBSERVATIONS_PATH = GOLD_ROOT / "observations.parquet"
SERIES_CATALOG_PATH = GOLD_ROOT / "series_catalog.parquet"
GOLD_DB_PATH = GOLD_ROOT / "catalog.duckdb"

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
                "source": source,
                "freq": row.get("FREQUENCY_STR"),
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
                "freq": "AYLIK",
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
    if evds_ids:
        obs_frames.append(build_silver_observations(evds_ids, source="evds"))
        catalog_frames.append(build_series_catalog(evds_ids, source="evds"))
    if bddk_ids:
        obs_frames.append(build_bddk_silver_observations(bddk_ids))
        catalog_frames.append(build_bddk_series_catalog(bddk_ids))

    observations = (
        pd.concat(obs_frames, ignore_index=True)
        if obs_frames
        else pd.DataFrame(columns=OBSERVATIONS_COLUMNS)
    )
    observations = observations.reindex(columns=OBSERVATIONS_COLUMNS)
    observations.to_parquet(OBSERVATIONS_PATH, index=False)

    catalog = (
        pd.concat(catalog_frames, ignore_index=True)
        if catalog_frames
        else pd.DataFrame(columns=SERIES_CATALOG_COLUMNS)
    )
    catalog.to_parquet(SERIES_CATALOG_PATH, index=False)

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


def get(
    series_ids: Iterable[str], start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Long observations for series_ids, optionally restricted to [start, end] (period)."""
    ids = list(series_ids)
    if not ids:
        return pd.DataFrame(columns=OBSERVATIONS_COLUMNS)

    placeholders = ", ".join(["?"] * len(ids))
    clauses = [f"series_id IN ({placeholders})"]
    params: list = list(ids)
    if start is not None:
        clauses.append("period >= ?")
        params.append(start)
    if end is not None:
        clauses.append("period <= ?")
        params.append(end)
    where_sql = " AND ".join(clauses)

    con = connect()
    try:
        return con.execute(
            f"SELECT * FROM observations WHERE {where_sql} ORDER BY series_id, period",
            params,
        ).df()
    finally:
        con.close()


def wide(
    series_ids: Iterable[str], start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """get(), pivoted to one column per series_id, indexed by period."""
    long = get(series_ids, start, end)
    if long.empty:
        return long
    return long.pivot(index="period", columns="series_id", values="value").reset_index()


def run_sql(
    sql: str, params: Iterable | None = None, *, limit: int = 1000
) -> pd.DataFrame:
    """Execute a read-only SELECT against the gold views.

    Only a single SELECT statement is accepted (no ";"-chaining); a missing
    LIMIT is appended. This check is defense in depth, not the real
    control -- `connect()`'s enable_external_access=false is what actually
    stops file-read/exfiltration tricks (CLAUDE.md).
    """
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("run_sql: birden fazla ifade (;) kabul edilmez")
    if not stripped.lower().startswith("select"):
        raise ValueError("run_sql: sadece SELECT kabul edilir")
    if "limit" not in stripped.lower():
        stripped = f"{stripped} LIMIT {limit}"

    con = connect()
    try:
        return con.execute(stripped, list(params) if params else []).df()
    finally:
        con.close()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

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
