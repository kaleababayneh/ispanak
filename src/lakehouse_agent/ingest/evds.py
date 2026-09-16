"""EVDS v3 (TCMB) ingest client: metadata catalog build.

Bronze layer only: no aggregation, no unit conversion, no NaN handling.
Bulk per-datagroup-per-year data fetch (with truncation detection) is a
separate follow-up piece and is not implemented here.
"""

import json
import logging
import random
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

from lakehouse_agent.config import Settings

logger = logging.getLogger(__name__)

BASE_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"

# Path-style segments, verified 12-13 Sep 2026 with live requests (200 +
# application/json). Metadata endpoints take the form "<name>/<params>";
# there is no "?" anywhere and params are "&"-joined.
CATEGORIES_PATH = "categories/type=json"
DATAGROUPS_PATH = "datagroups/mode=0&type=json"
SERIELIST_PATH_TMPL = "serieList/code={code}&type=json"
# Bulk series data for one datagroup. Deliberately no aggregationTypes,
# formulas or frequency params -- bronze stores the raw original frequency,
# frequency conversion happens in silver, based on `kind` (CLAUDE.md #4/#7).
DATAGROUP_DATA_PATH_TMPL = "datagroup={code}&startDate={start}&endDate={end}&type=json"

RATE_LIMIT_RANGE = (0.3, 0.5)  # seconds, slept before every single attempt
MAX_RETRIES = 3
RETRYABLE_STATUS = {
    429,
    500,
    502,
    503,
    504,
}  # 403 excluded — retrying an auth error is pointless
REQUEST_TIMEOUT = 30

BRONZE_ROOT = Path("data/bronze/evds")
DATAGROUPS_DATA_ROOT = BRONZE_ROOT / "datagroups"  # <code>/<year>.parquet
CATALOG_CACHE_ROOT = BRONZE_ROOT / "_catalog_cache"
CATEGORIES_SNAPSHOT_PATH = BRONZE_ROOT / "categories.parquet"
DATAGROUPS_SNAPSHOT_PATH = BRONZE_ROOT / "datagroups.parquet"
CATALOG_PATH = BRONZE_ROOT / "catalog.parquet"
FETCH_LOG_JSONL_PATH = BRONZE_ROOT / "fetch_log.jsonl"  # true append target
FETCH_LOG_PARQUET_PATH = BRONZE_ROOT / "fetch_log.parquet"  # compacted output


class EvdsRedirectError(RuntimeError):
    """Raised when EVDS responds with a 3xx status. Redirects are never followed."""


class EvdsAuthError(RuntimeError):
    """Raised on 403 Forbidden — an invalid or missing API key. Never retried."""


class EvdsRequestError(RuntimeError):
    """Raised when a request fails (non-2xx, non-3xx, non-403) or exhausts all retries."""


@dataclass(frozen=True)
class ApiCall:
    url: str
    params: str
    http_status: int
    row_count: int | None
    fetched_at: str
    outcome: str


def build_url(path: str) -> str:
    """Join BASE_URL with a pre-formatted path-style segment (e.g. "categories/type=json").

    Deliberately does not use requests' params= kwarg, which would produce a
    `?key=value` query string — EVDS v3 does not accept that form and the
    parameters must be built into `path` by the caller instead.
    """
    return BASE_URL + path


def redact_url(url: str) -> str:
    """Mask any `key=...` occurrence in a URL, for logging/storage only.

    Never applied to the actual outgoing request — the key travels in the
    HTTP header, never in the URL, but any URL that is logged or persisted
    (e.g. in fetch_log) must never be able to leak it either.
    """
    return re.sub(r"key=[^&]*", "key=***", url)


def _log_call(call: ApiCall) -> None:
    """Append one JSON line to fetch_log.jsonl.

    Parquet has no append; rewriting the whole parquet file on every one of
    several thousand calls would be O(n^2). Real compaction to
    fetch_log.parquet happens once, via compact_fetch_log().
    """
    BRONZE_ROOT.mkdir(parents=True, exist_ok=True)
    with FETCH_LOG_JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(call), ensure_ascii=False) + "\n")


def compact_fetch_log() -> Path:
    """Rebuild fetch_log.parquet from fetch_log.jsonl.

    Call once at the end of a catalog build or bulk fetch run — not after
    every individual call.
    """
    if not FETCH_LOG_JSONL_PATH.exists():
        return FETCH_LOG_PARQUET_PATH
    lines = FETCH_LOG_JSONL_PATH.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    pd.DataFrame(rows).to_parquet(FETCH_LOG_PARQUET_PATH, index=False)
    return FETCH_LOG_PARQUET_PATH


def _row_count(payload: list | dict) -> int | None:
    """Metadata endpoints (categories/datagroups/serieList) return a bare
    list. The bulk data endpoint ("datagroup=...") returns
    {"totalCount": N, "items": [...]} instead -- measured 16 Sep 2026,
    not documented in the official guide. Handle both.
    """
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return len(payload["items"])
    return None


def _extract_rows(payload: list | dict) -> list:
    """Unwrap the bulk data endpoint's {"totalCount", "items"} envelope; a
    bare list (metadata endpoints) passes through unchanged."""
    if isinstance(payload, dict):
        return payload.get("items", [])
    return payload


def _authenticated_get(path: str, settings: Settings) -> list | dict:
    """Single authenticated GET against EVDS v3. Returns the parsed JSON payload.

    - auth: `key` HTTP header (never the URL)
    - allow_redirects=False; any 3xx -> EvdsRedirectError
    - 403 -> immediate EvdsAuthError, no retry (an auth error will not fix itself)
    - status in RETRYABLE_STATUS -> up to MAX_RETRIES attempts, exponential backoff
    - a random RATE_LIMIT_RANGE sleep happens before every attempt
    - every attempt (success or failure) is logged via _log_call
    """
    url = build_url(path)
    logged_url = redact_url(url)
    last_status: int | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(random.uniform(*RATE_LIMIT_RANGE))
        response = requests.get(
            url,
            headers=settings.evds_auth_header(),
            allow_redirects=False,
            timeout=REQUEST_TIMEOUT,
        )
        last_status = response.status_code
        fetched_at = datetime.now(UTC).isoformat()

        if 300 <= response.status_code < 400:
            _log_call(
                ApiCall(
                    logged_url, path, response.status_code, None, fetched_at, "redirect"
                )
            )
            raise EvdsRedirectError(
                f"EVDS redirected ({response.status_code}): {logged_url}"
            )

        if response.status_code == 403:
            _log_call(ApiCall(logged_url, path, 403, None, fetched_at, "auth_error"))
            raise EvdsAuthError(
                "EVDS anahtarı geçersiz veya eksik (Profilim > API Key Kopyala)."
            )

        if response.status_code == 200:
            payload = response.json()
            row_count = _row_count(payload)
            _log_call(ApiCall(logged_url, path, 200, row_count, fetched_at, "ok"))
            return payload

        if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            _log_call(
                ApiCall(
                    logged_url, path, response.status_code, None, fetched_at, "retry"
                )
            )
            time.sleep(2 ** (attempt - 1))
            continue

        _log_call(
            ApiCall(logged_url, path, response.status_code, None, fetched_at, "failed")
        )
        raise EvdsRequestError(
            f"EVDS request failed ({response.status_code}): {logged_url}"
        )

    raise EvdsRequestError(
        f"EVDS request failed after {MAX_RETRIES} attempts (last status {last_status})"
    )


def fetch_categories(settings: Settings) -> pd.DataFrame:
    return pd.DataFrame(_authenticated_get(CATEGORIES_PATH, settings))


def fetch_datagroups(settings: Settings) -> pd.DataFrame:
    return pd.DataFrame(_authenticated_get(DATAGROUPS_PATH, settings))


def fetch_serie_list(datagroup_code: str, settings: Settings) -> pd.DataFrame:
    path = SERIELIST_PATH_TMPL.format(code=datagroup_code)
    return pd.DataFrame(_authenticated_get(path, settings))


def build_catalog(settings: Settings, *, force: bool = False) -> Path:
    """Build data/bronze/evds/catalog.parquet: one row per series, joined
    with its datagroup and category metadata.

    Idempotent at every level: categories/datagroups snapshots, per-datagroup
    serieList caches, and the final catalog are each skipped if they already
    exist and force=False — so a crash partway through ~680 serieList calls
    does not force a full re-fetch on retry, and a second full run makes
    zero HTTP calls.
    """
    if CATALOG_PATH.exists() and not force:
        return CATALOG_PATH

    BRONZE_ROOT.mkdir(parents=True, exist_ok=True)
    CATALOG_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if CATEGORIES_SNAPSHOT_PATH.exists() and not force:
        categories = pd.read_parquet(CATEGORIES_SNAPSHOT_PATH)
    else:
        categories = fetch_categories(settings)
        categories.to_parquet(CATEGORIES_SNAPSHOT_PATH, index=False)

    if DATAGROUPS_SNAPSHOT_PATH.exists() and not force:
        datagroups = pd.read_parquet(DATAGROUPS_SNAPSHOT_PATH)
    else:
        datagroups = fetch_datagroups(settings)
        datagroups.to_parquet(DATAGROUPS_SNAPSHOT_PATH, index=False)

    datagroup_codes = datagroups["DATAGROUP_CODE"].tolist()
    for code in datagroup_codes:
        cache_file = CATALOG_CACHE_ROOT / f"{code}.parquet"
        if cache_file.exists() and not force:
            continue
        series = fetch_serie_list(code, settings)
        series.to_parquet(cache_file, index=False)

    series_frames = [
        pd.read_parquet(CATALOG_CACHE_ROOT / f"{code}.parquet")
        for code in datagroup_codes
    ]
    all_series = (
        pd.concat(series_frames, ignore_index=True) if series_frames else pd.DataFrame()
    )

    catalog = all_series.merge(
        datagroups, on="DATAGROUP_CODE", how="left", suffixes=("", "_dg")
    )
    catalog = catalog.merge(
        categories, on="CATEGORY_ID", how="left", suffixes=("", "_cat")
    )
    catalog.to_parquet(CATALOG_PATH, index=False)

    compact_fetch_log()
    return CATALOG_PATH


def build_column_rename_map(serie_codes: Iterable[str]) -> dict[str, str]:
    """{underscore_form: original_SERIE_CODE} for the given catalog codes.

    Built from an exact list of known codes rather than a blind "_" -> "."
    replace: a multi-segment code like TP.ISTIRAKBS.A1 becomes
    TP_ISTIRAKBS_A1, and only a lookup against the catalog's own SERIE_CODE
    values can invert that unambiguously.
    """
    return {code.replace(".", "_"): code for code in serie_codes}


def denormalize_response_columns(
    df: pd.DataFrame, serie_codes: Iterable[str]
) -> pd.DataFrame:
    """Rename EVDS response columns (dots turned into underscores, e.g.
    TP_ISTIRAKBS_A1) back to their catalog SERIE_CODE form (TP.ISTIRAKBS.A1).

    Columns not present in the rename map (e.g. "Tarih", "UNIXTIME") pass
    through untouched. This is the single place in the codebase this
    reversal happens — lake/silver.py (a later PR) must import and reuse it
    rather than reimplementing the logic.
    """
    rename_map = build_column_rename_map(serie_codes)
    return df.rename(columns=rename_map)


# --- Bulk data fetch: one datagroup x one year -------------------------------

_DAILY_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")  # günlük/işgünü/haftalık
_QUARTERLY_RE = re.compile(r"^(\d{4})-Q(\d)$")
_MONTHLY_RE = re.compile(r"^(\d{4})-(\d{1,2})$")  # sıfırsız ay: "2024-1"
_YEARLY_RE = re.compile(r"^(\d{4})$")


def _parse_tarih(value: str) -> date:
    """Parse one EVDS "Tarih" cell into a date.

    Format is detected by regex FIRST, then parsed -- trying strptime
    formats in sequence and falling through on exception is unsafe here:
    "2024" and "2024-1" both look like partial matches of each other's
    pattern. Formats measured 12-13 Sep 2026, one per frequency family:
    daily/işgünü/haftalık "01-01-2024" (DD-MM-YYYY), aylık "2024-1"
    (YYYY-M, no leading zero), üç aylık "2024-Q1", yıllık "2024".
    """
    value = value.strip()

    m = _DAILY_RE.match(value)
    if m:
        day, month, year = m.groups()
        return date(int(year), int(month), int(day))

    m = _QUARTERLY_RE.match(value)
    if m:
        year, quarter = m.groups()
        month = (int(quarter) - 1) * 3 + 1
        return date(int(year), month, 1)

    m = _MONTHLY_RE.match(value)
    if m:
        year, month = m.groups()
        return date(int(year), int(month), 1)

    m = _YEARLY_RE.match(value)
    if m:
        return date(int(value), 1, 1)

    raise ValueError(f"Tanınmayan EVDS Tarih biçimi: {value!r}")


def detect_truncation(
    df: pd.DataFrame,
    expected_start: date,
    *,
    date_column: str = "Tarih",
    tolerance_days: int = 10,
) -> bool:
    """True only when the response's first row starts well after the
    requested start -- the signature of the 1000-observation cap silently
    dropping the earliest rows (measured: 2021-2026 requested, 2023-12-20
    came back first).

    Deliberately checks the START only. A weekly/Friday-anchored group
    legitimately overruns the requested year at the END (2024 requested,
    last row 2025-01-03, measured) -- not truncation. An empty response is
    also not truncation (series not started yet that year) -- callers must
    check `df.empty` themselves before calling this, since an empty frame
    has no row to compare.
    """
    if df.empty:
        return False
    first_date = _parse_tarih(str(df[date_column].iloc[0]))
    return (first_date - expected_start).days > tolerance_days


@dataclass(frozen=True)
class YearFetchResult:
    datagroup_code: str
    year: int
    outcome: str  # "cached" | "ok" | "empty" | "truncated" | "failed"
    row_count: int | None
    path: Path | None


def _datagroup_year_path(datagroup_code: str, year: int) -> Path:
    return DATAGROUPS_DATA_ROOT / datagroup_code / f"{year}.parquet"


def fetch_datagroup_year(
    datagroup_code: str, year: int, settings: Settings, *, force: bool = False
) -> YearFetchResult:
    """Fetch one datagroup's raw (wide, bronze) data for one calendar year.

    Idempotent: skips the HTTP call entirely if the year's parquet already
    exists. A truncated or network-failed year never writes a parquet, so a
    retry with force=False will fetch it again rather than silently
    accepting partial data. A 403 (EvdsAuthError) is not caught here --
    it propagates and is meant to stop the whole bulk run.
    """
    out_path = _datagroup_year_path(datagroup_code, year)
    if out_path.exists() and not force:
        return YearFetchResult(datagroup_code, year, "cached", None, out_path)

    expected_start = date(year, 1, 1)
    path = DATAGROUP_DATA_PATH_TMPL.format(
        code=datagroup_code,
        start=expected_start.strftime("%d-%m-%Y"),
        end=date(year, 12, 31).strftime("%d-%m-%Y"),
    )

    try:
        payload = _authenticated_get(path, settings)
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        EvdsRequestError,
        EvdsRedirectError,
    ) as exc:
        logger.warning("EVDS fetch failed for %s/%s: %s", datagroup_code, year, exc)
        return YearFetchResult(datagroup_code, year, "failed", None, None)

    df = pd.DataFrame(_extract_rows(payload))
    if df.empty:
        return YearFetchResult(datagroup_code, year, "empty", 0, None)

    row_count = len(df)
    if row_count >= 1000:
        logger.warning(
            "%s/%s: %d satır döndü (1000 sınırına yakın/eşit) -- ilk tarih kontrol ediliyor",
            datagroup_code,
            year,
            row_count,
        )

    if detect_truncation(df, expected_start):
        logger.warning(
            "%s/%s: kesme tespit edildi (ilk satır %s, beklenen %s) -- parquet yazılmadı",
            datagroup_code,
            year,
            df["Tarih"].iloc[0],
            expected_start,
        )
        return YearFetchResult(datagroup_code, year, "truncated", row_count, None)

    df["_fetched_at"] = datetime.now(UTC).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return YearFetchResult(datagroup_code, year, "ok", row_count, out_path)


def fetch_datagroups_years(
    datagroup_codes: Iterable[str],
    years: Iterable[int],
    settings: Settings,
    *,
    force: bool = False,
) -> list[YearFetchResult]:
    """Fetch every (datagroup, year) pair. Logs progress every 10 groups and
    keeps going past a "failed"/"truncated"/"empty" outcome -- the caller
    (ingest/run.py) reports failures at the end. A 403 still propagates and
    stops the whole run immediately.
    """
    codes = list(datagroup_codes)
    year_list = list(years)
    results: list[YearFetchResult] = []

    for i, code in enumerate(codes, start=1):
        for year in year_list:
            results.append(fetch_datagroup_year(code, year, settings, force=force))
        if i % 10 == 0:
            logger.info("EVDS toplu çekim: %d/%d veri grubu tamamlandı", i, len(codes))

    compact_fetch_log()
    return results
