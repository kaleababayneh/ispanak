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

from lakehouse_agent.ingest.evds import CATALOG_PATH
from lakehouse_agent.lake.silver import build_silver_observations

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


def build_gold(series_ids: Iterable[str], *, source: str = "evds") -> tuple[Path, Path]:
    """Assemble silver -> observations.parquet + series_catalog.parquet."""
    series_ids = list(series_ids)
    GOLD_ROOT.mkdir(parents=True, exist_ok=True)

    observations = build_silver_observations(series_ids, source=source)
    observations = observations.reindex(columns=OBSERVATIONS_COLUMNS)
    observations.to_parquet(OBSERVATIONS_PATH, index=False)

    catalog = build_series_catalog(series_ids, source=source)
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
