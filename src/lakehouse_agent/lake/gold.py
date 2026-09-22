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
import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse_agent.analysis import transforms
from lakehouse_agent.ingest.evds import CATALOG_PATH
from lakehouse_agent.lake import bddk
from lakehouse_agent.lake.classify import EVDS_AGG_METHODS, evds_kind, evds_unit
from lakehouse_agent.lake.silver import iter_evds_silver, map_freq

logger = logging.getLogger(__name__)

GOLD_ROOT = Path("data/gold")
OBSERVATIONS_PATH = GOLD_ROOT / "observations.parquet"
SERIES_CATALOG_PATH = GOLD_ROOT / "series_catalog.parquet"
GOLD_DB_PATH = GOLD_ROOT / "catalog.duckdb"
DEDUP_STATS_PATH = GOLD_ROOT / "dedup_stats.json"
#: Why each series has the kind and unit it has (lake/classify.py): the
#: audit trail behind docs/kind_review.md. Not part of the data contract.
CLASSIFICATION_PATH = GOLD_ROOT / "classification.parquet"

# Project scope ends here (CLAUDE.md: 2021-01 -- 2026-06). EVDS ingest
# fetches whole calendar years, so the 2026 chunk carries every month
# published so far, while BDDK already stops at --until. The cut happens in
# gold, never in bronze: bronze keeps whatever the source returned.
GOLD_UNTIL = "2026-06"
GOLD_SINCE = "2021-01"

#: EVDS frequencies the contract has no letter for (22 Sep decision): 224
#: semi-monthly and 65 semi-annual series, left out of the pool and listed
#: in docs/data_dictionary.md. Bronze keeps them.
EXCLUDED_FREQUENCY_STRS = frozenset({"AYDA İKİ KEZ", "ALTI AYLIK"})

OBSERVATIONS_COLUMNS = [
    "series_id",
    "period",
    "freq",
    "value",
    "unit",
    "source",
    "fetched_at",
    "obs_date",
]

#: The observations file's physical schema. Written in pieces (the pool is
#: millions of rows), so it is fixed up front rather than inferred per piece:
#: a group whose units are all empty would otherwise make `unit` a null
#: column in one piece and a string column in the next.
OBSERVATIONS_SCHEMA = pa.schema(
    [
        ("series_id", pa.string()),
        ("period", pa.date32()),
        ("freq", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("source", pa.string()),
        ("fetched_at", pa.string()),
        ("obs_date", pa.date32()),
    ]
)
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

# Senaryo serileri (16 Eylül entegrasyon), elle eşlenmiş. Havuzun geri kalanı
# lake/classify.py kurallarıyla sınıflanıyor; buradaki değerler kuralların
# önüne geçer ve kuralların bu seriler için aynı sonucu verdiği test ediliyor
# (tests/test_classify.py).
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


def evds_catalog_rows(series_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Bronze catalog rows for the EVDS pool, classified.

    Every series whose frequency the contract can name (see
    EXCLUDED_FREQUENCY_STRS), or only `series_ids` when given ("evds:"
    prefixed). Adds `unit`/`unit_source`, `kind`/`kind_reason`, `agg_method`
    and `cum_type`: the rules of lake/classify.py, with SCENARIO_SERIES
    taking precedence. EVDS series are never year-to-date cumulative
    ("KÜMÜLATİF" is EVDS's aggregation hint, docs/data_dictionary.md), so
    `cum_type` is "none".
    """
    catalog = pd.read_parquet(CATALOG_PATH)
    catalog = catalog[~catalog["FREQUENCY_STR"].isin(EXCLUDED_FREQUENCY_STRS)]
    if series_ids is not None:
        wanted = {sid.split(":", 1)[1] for sid in series_ids if sid.startswith("evds:")}
        catalog = catalog[catalog["SERIE_CODE"].isin(wanted)]
    catalog = catalog.drop_duplicates("SERIE_CODE").reset_index(drop=True)

    units = [
        evds_unit(birimi, name)
        for birimi, name in zip(
            catalog.get("BIRIMI", pd.Series(None, index=catalog.index)),
            catalog.get("SERIE_NAME", pd.Series(None, index=catalog.index)),
            strict=True,
        )
    ]
    catalog["unit"] = [unit for unit, _ in units]
    catalog["unit_source"] = [source for _, source in units]
    agg = catalog.get("DEFAULT_AGG_METHOD", pd.Series(None, index=catalog.index))
    catalog["agg_method"] = [a if a in EVDS_AGG_METHODS else None for a in agg]

    kinds = [
        evds_kind(unit, name, method)
        for unit, name, method in zip(
            catalog["unit"],
            catalog.get("SERIE_NAME", pd.Series(None, index=catalog.index)),
            catalog["agg_method"],
            strict=True,
        )
    ]
    catalog["kind"] = [kind for kind, _ in kinds]
    catalog["kind_reason"] = [reason for _, reason in kinds]
    catalog["cum_type"] = "none"

    for position, code in enumerate(catalog["SERIE_CODE"]):
        override = SCENARIO_SERIES.get(f"evds:{code}")
        if override is None:
            continue
        for column in ("kind", "agg_method", "cum_type"):
            catalog.at[position, column] = override[column]
        catalog.at[position, "kind_reason"] = "elle eslendi (SCENARIO_SERIES)"
    return catalog


def build_series_catalog(
    series_ids: Iterable[str], *, source: str = "evds"
) -> pd.DataFrame:
    """series_catalog.parquet rows for the given EVDS series_ids: names,
    frequency, dates, topic and tags from the EVDS bronze catalog; unit,
    kind, agg_method and cum_type from `evds_catalog_rows`.
    """
    return _catalog_frame(evds_catalog_rows(list(series_ids)), source=source)


def _catalog_frame(rows: pd.DataFrame, *, source: str = "evds") -> pd.DataFrame:
    def column(name: str) -> pd.Series:
        return rows[name] if name in rows.columns else pd.Series(None, index=rows.index)

    frame = pd.DataFrame(
        {
            "series_id": f"{source}:" + rows["SERIE_CODE"],
            "name_tr": column("SERIE_NAME"),
            "name_en": column("SERIE_NAME_ENG"),
            "source": source.upper(),
            "freq": column("FREQUENCY_STR").map(map_freq),
            "unit": rows["unit"],
            "kind": rows["kind"],
            "agg_method": rows["agg_method"],
            "cum_type": rows["cum_type"],
            "start_date": column("START_DATE"),
            "end_date": column("END_DATE"),
            "topic": column("TOPIC_TITLE_TR"),
            "tags": column("TAG"),
            "metadata_url": column("METADATA_LINK"),
            "notes": column("NOTE"),
        }
    )
    return frame[SERIES_CATALOG_COLUMNS].reset_index(drop=True)


def default_series_ids() -> list[str]:
    """The scenario's series: the EVDS SCENARIO_SERIES and the BDDK ids the
    scenario has used since 17 Sep (ingest/bddk_tables.yaml aliases).
    `build_gold` with no ids builds the whole pool instead."""
    _tables, aliases = bddk.load_tables()
    return list(SCENARIO_SERIES) + sorted(set(aliases.values()))


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


def _clip_window(
    observations: pd.DataFrame, since: str | None, until: str | None
) -> pd.DataFrame:
    """Keep rows observed inside [since, until] ("YYYY-MM" or "YYYY-MM-DD").

    On `obs_date`, the date the source gave the value: the project's window
    is 2021-01 - 2026-06 of observations. A week whose Monday is 29 June 2026
    but whose Friday is 3 July is July's, and goes; the week of Monday 28
    December 2020, observed on 1 January 2021, is January's, and stays.
    The period column is filtered, not converted.
    """
    observed = pd.to_datetime(observations["obs_date"])
    keep = pd.Series(True, index=observations.index)
    if since is not None:
        keep &= observed >= pd.Timestamp(_normalize_start(since))
    if until is not None:
        keep &= observed <= pd.Timestamp(_normalize_end(until))
    return observations[keep].reset_index(drop=True)


def _with_values(observations: pd.DataFrame) -> pd.DataFrame:
    """Only the series with at least one value in the window: an all-empty
    series is a catalog row promising data the pool does not have."""
    has_value = observations.groupby("series_id")["value"].transform(
        lambda values: values.notna().any()
    )
    return observations[has_value.astype(bool)]


def build_gold(
    series_ids: Iterable[str] | None = None,
    *,
    since: str | None = GOLD_SINCE,
    until: str | None = GOLD_UNTIL,
) -> tuple[Path, Path]:
    """Assemble silver -> observations.parquet + series_catalog.parquet.

    With no `series_ids`, the whole pool: every EVDS series the bronze
    catalog lists in a contract frequency, and every series of the BDDK
    monthly and weekly bulletins and of FinTürk (lake/bddk.py).
    Given ids may mix "evds:" and "bddk:" prefixes. Observations outside
    [since, until] (by observation date, see `_clip_window`) are dropped,
    and so is a series left with no value at all; None disables a bound.

    Written one EVDS datagroup at a time with a fixed schema, zstd
    compressed: the pool is millions of rows. Each piece is checked against
    the contract columns before it is written. series_catalog's
    start_date/end_date are left alone: they describe the source series' own
    coverage, not what gold holds (the tools read that from observations).
    """
    ids = None if series_ids is None else list(series_ids)
    GOLD_ROOT.mkdir(parents=True, exist_ok=True)

    evds_rows = evds_catalog_rows(ids)
    if ids is not None and not any(sid.startswith("evds:") for sid in ids):
        evds_rows = evds_rows.iloc[0:0]
    wanted_bddk = (
        None if ids is None else {sid for sid in ids if sid.startswith("bddk:")}
    )

    written: set[str] = set()
    dropped_duplicates = 0
    partial = OBSERVATIONS_PATH.with_suffix(".parquet.partial")
    writer = pq.ParquetWriter(partial, OBSERVATIONS_SCHEMA, compression="zstd")
    try:

        def write(frame: pd.DataFrame) -> None:
            _validate_columns(frame, OBSERVATIONS_COLUMNS, label="observations")
            frame = _with_values(
                _clip_window(frame[OBSERVATIONS_COLUMNS], since, until)
            )
            if frame.empty:
                return
            written.update(frame["series_id"].unique())
            writer.write_table(
                pa.Table.from_pandas(
                    frame, schema=OBSERVATIONS_SCHEMA, preserve_index=False
                )
            )

        units = dict(zip(evds_rows["SERIE_CODE"], evds_rows["unit"], strict=True))
        for frame, dropped in iter_evds_silver(evds_rows, units=units):
            dropped_duplicates += dropped
            write(frame)
        bddk_catalogs = []
        if wanted_bddk is None or wanted_bddk:
            for build in (bddk.build_monthly, bddk.build_weekly, bddk.build_finturk):
                frame, catalog_part, dropped = build()
                dropped_duplicates += dropped
                if wanted_bddk is not None:
                    frame = frame[frame["series_id"].isin(wanted_bddk)]
                if not frame.empty:
                    write(frame)
                    bddk_catalogs.append(catalog_part)
    finally:
        writer.close()
    partial.replace(OBSERVATIONS_PATH)

    evds_catalog = _catalog_frame(evds_rows)
    frames = [evds_catalog[evds_catalog["series_id"].isin(written)]]
    for bddk_catalog in bddk_catalogs:
        frames.append(bddk_catalog[bddk_catalog["series_id"].isin(written)])
    catalog = pd.concat(frames, ignore_index=True)
    _validate_columns(catalog, SERIES_CATALOG_COLUMNS, label="series_catalog")
    catalog = catalog[SERIES_CATALOG_COLUMNS].sort_values("series_id")
    catalog.to_parquet(SERIES_CATALOG_PATH, index=False)

    classification = pd.DataFrame(
        {
            "series_id": "evds:" + evds_rows["SERIE_CODE"],
            "datagroup": evds_rows.get("DATAGROUP_CODE"),
            "kind": evds_rows["kind"],
            "kind_reason": evds_rows["kind_reason"],
            "unit": evds_rows["unit"],
            "unit_source": evds_rows["unit_source"],
            "agg_method": evds_rows["agg_method"],
        }
    )
    classification = classification[classification["series_id"].isin(written)]
    classification.to_parquet(CLASSIFICATION_PATH, index=False)

    DEDUP_STATS_PATH.write_text(
        json.dumps({"dropped_duplicates": dropped_duplicates}), encoding="utf-8"
    )

    if GOLD_DB_PATH.exists():
        GOLD_DB_PATH.unlink()  # views embed the parquet path; rebuild on every gold run

    return OBSERVATIONS_PATH, SERIES_CATALOG_PATH


def _ensure_tables(db_path: Path | None = None) -> None:
    """(Re)materialize the two parquet files as native tables inside the
    DuckDB file itself, not as views over external parquet paths.

    A view over read_parquet(...) would need external file access at query
    time -- which is exactly what enable_external_access=false forbids on
    the read-only connection `connect()` hands out. Copying the bytes into
    the .duckdb file at build time (this writable connection, default
    external access) means later read-only queries never need to touch the
    filesystem beyond that one file.
    """
    # Resolved at call time: a default bound at import would ignore a test's
    # (or a caller's) change of GOLD_DB_PATH.
    con = duckdb.connect(str(db_path or GOLD_DB_PATH))
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


def _stale(db_path: Path) -> bool:
    """Missing, or older than a parquet file it copies.

    The file is derived and gitignored. Built only when missing, it kept
    serving the old tables after a pull that updated the committed parquet --
    and the adapter, which reads the catalog parquet directly, then disagreed
    with it about which series exist.
    """
    if not db_path.exists():
        return True
    built = db_path.stat().st_mtime
    return any(
        source.exists() and source.stat().st_mtime > built
        for source in (OBSERVATIONS_PATH, SERIES_CATALOG_PATH)
    )


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
    if _stale(GOLD_DB_PATH):
        try:
            _ensure_tables()
        except Exception:
            # Rebuilding opens the same file for writing, and DuckDB refuses
            # that while a read-only connection is open on it in the same
            # process (measured, duckdb 1.5.5, in both orders) -- a tool that
            # overran its step timeout and is still reading in its thread is
            # enough. An out-of-date copy of the tables answers the query;
            # failing the call does not. If the file is missing altogether
            # the `connect` below raises, which is the honest outcome.
            if not GOLD_DB_PATH.exists():
                raise
            logger.warning(
                "gold: %s yenilenemedi, eski kopya kullaniliyor", GOLD_DB_PATH
            )
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
    by observation date (`obs_date`): "2025-04" is what was observed in
    April, so the week of Monday 31 March, observed Friday 4 April, is in
    it. For every series but a weekly one `obs_date` equals `period`.
    `start`/`end` accept "YYYY-MM" (docs/contracts.md's signature)
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
        clauses.append("obs_date >= ?")
        params.append(_normalize_start(start))
    if end is not None:
        clauses.append("obs_date <= ?")
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


def _series_methods(series_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """(kind, agg_method) per series_id from series_catalog, for wide()'s
    per-series to_freq aggregation: agg_method decides, kind is the
    fallback. Empty dict if series_ids is empty."""
    if not series_ids:
        return {}
    placeholders = ", ".join(["?"] * len(series_ids))
    con = connect()
    try:
        rows = con.execute(
            "SELECT series_id, kind, agg_method FROM series_catalog "
            f"WHERE series_id IN ({placeholders})",
            list(series_ids),
        ).fetchall()
    finally:
        con.close()
    return {sid: (kind, agg) for sid, kind, agg in rows}


def _pivot_raw(long: pd.DataFrame) -> pd.DataFrame:
    """Legacy wide() shape (freq=None): pivot on exact period match, no
    alignment. Series with different native frequencies rarely share exact
    period values (a Monday-anchored weekly series vs. a 1st-of-month
    monthly series) -- pivoting them together produces a table where almost
    every row has only one column populated, unusable for a joint read even
    though it succeeds without error. Measured:
    wide(["evds:TP.KTF12", "evds:TP.TUKFIY2025.GENEL"], "2021-01",
    "2025-12") -> 311 rows, no row with both columns filled (weekly vs.
    monthly). A mismatch and the per-column fill rate are logged so the
    problem is visible rather than a silent unusable table.
    """
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


def wide(
    series_ids: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    freq: str | None = "M",
) -> pd.DataFrame:
    """get(), pivoted to one column per series_id, indexed by period.

    `freq` (gold contract Freq: D|W|M|Q|A, default "M") aligns every
    requested series to that common frequency first, via
    `analysis.transforms.to_freq` with each series' own `agg_method` from
    series_catalog (its `kind` when there is none), bucketing by
    observation date -- resolving the mismatched-frequency problem below by
    construction instead of merely reporting it. It never goes finer: a
    series coarser than `freq` is left at its own periods (the tools refuse
    that request before it gets here). `freq=None` skips alignment entirely
    and pivots on exact period match (legacy behavior, kept for backward
    compatibility) -- see `_pivot_raw` for why that silently produces a
    mostly-empty table when series have different native frequencies.
    """
    ids = list(series_ids)
    long = get(ids, start, end)
    if long.empty:
        return long

    if freq is None:
        return _pivot_raw(long)

    methods = _series_methods(ids)
    aligned: dict[str, pd.DataFrame] = {}
    for sid in ids:
        sub = long.loc[long["series_id"] == sid, ["period", "obs_date", "value"]]
        if sub.empty:
            continue
        kind, agg_method = methods.get(sid, (None, None))
        if kind is None and agg_method is None:
            logger.warning(
                "wide(): %s için series_catalog'da kind/agg_method yok, "
                "%s frekansına hizalanmadı",
                sid,
                freq,
            )
            aligned[sid] = sub[["period", "value"]]
            continue
        aligned[sid], _formula = transforms.to_freq(
            sub, kind, freq, agg_method=agg_method
        )

    if not aligned:
        return pd.DataFrame(columns=["period", *ids])

    result, _formula = transforms.align_join(aligned, how="outer")
    return result


def coverage() -> pd.DataFrame:
    """(series_id, first, last): the first and last observation date holding
    a value, per pool series.

    Not `run_sql`: that is the model's door and always wraps a query in
    `LIMIT 1000`, which silently cut this aggregate to the first thousand
    series -- every candidate past them was then offered with no coverage at
    all (measured 22 Sep, 45k series). The guard stays where it belongs; the
    lake's own reads go through named queries like this one.
    """
    con = connect()
    try:
        return con.execute(
            "SELECT series_id, MIN(obs_date) AS first, MAX(obs_date) AS last "
            "FROM observations WHERE value IS NOT NULL GROUP BY series_id"
        ).df()
    finally:
        con.close()


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

    from lakehouse_agent.lake.classify import render_kind_review

    obs_path, catalog_path = build_gold()
    catalog = pd.read_parquet(catalog_path)
    review = Path("docs/kind_review.md")
    review.write_text(
        render_kind_review(pd.read_parquet(CLASSIFICATION_PATH), catalog),
        encoding="utf-8",
    )
    logger.info(
        "gold yazıldı: %s (%.1f MB), %s (%d seri: %s)",
        obs_path,
        obs_path.stat().st_size / 1e6,
        catalog_path,
        len(catalog),
        catalog["source"].value_counts().to_dict(),
    )
