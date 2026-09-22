"""EVDS v3 (TCMB) ingest client: metadata catalog and bulk data fetch.

Bronze layer only: no aggregation, no unit conversion, no NaN handling.
Data is fetched one datagroup x one calendar year at a time (the 1000-row
cap), with truncation detection; a year that times out is fetched again in
quarters, and a quarter that times out in months (`fetch_datagroup_year`).
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


class EvdsTimeoutError(EvdsRequestError):
    """The server did not answer within REQUEST_TIMEOUT. Not retried as is:
    the previous full run lost 34 datagroup-years to timeouts that a second
    identical round did not recover (large daily groups), so the caller asks
    for a smaller window instead."""


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
        try:
            response = requests.get(
                url,
                headers=settings.evds_auth_header(),
                allow_redirects=False,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.Timeout as exc:
            # Logged too: a call that got no answer is still a call, and the
            # log is what says which windows were lost and why. Status 0 =
            # no HTTP response at all.
            _log_call(
                ApiCall(
                    logged_url, path, 0, None, datetime.now(UTC).isoformat(), "timeout"
                )
            )
            raise EvdsTimeoutError(f"EVDS zaman asimi: {logged_url}") from exc
        except requests.exceptions.ConnectionError as exc:
            _log_call(
                ApiCall(
                    logged_url,
                    path,
                    0,
                    None,
                    datetime.now(UTC).isoformat(),
                    "connection_error",
                )
            )
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            raise EvdsRequestError(f"EVDS baglanti hatasi: {logged_url}") from exc
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
    total_count: int | None = None,
) -> bool:
    """True only when the response is actually missing rows the server had
    -- never when a series simply started partway through the requested
    year. Measured regression: TP.HPBITABLO6.3 legitimately starts
    2024-06-01; the old "first row far after expected_start" heuristic
    alone flagged that whole year as truncated and its parquet was never
    written, silently losing every real observation that year.

    Two independent signals, either is sufficient:
    - `total_count` (the bulk endpoint's own "totalCount" field, see
      `_row_count`) says more observations exist than were actually
      returned -- the server's own word for it, checked first regardless of
      row count.
    - Fallback heuristic, only once row_count actually hits the
      1000-observation cap AND the first row starts well after
      `expected_start` -- the signature measured on the real cap incident
      (2021-2026 requested, 2023-12-20 came back first). Below 1000 rows, a
      late first date just means the series started late that year, not
      truncation.

    Deliberately checks the START only (not `total_count`'s own end-of-year
    implications): a weekly/Friday-anchored group legitimately overruns the
    requested year at the END (2024 requested, last row 2025-01-03,
    measured) -- not truncation. An empty response is also not truncation
    (series not started yet that year) -- callers must check `df.empty`
    themselves before calling this, since an empty frame has no row to
    compare.
    """
    if df.empty:
        return False
    row_count = len(df)
    if total_count is not None and total_count > row_count:
        return True
    if row_count < 1000:
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


@dataclass(frozen=True)
class _Window:
    """One request's date range and the bronze file its answer goes to."""

    label: str  # "2024", "2024-Q1" or "2024-03": the file stem
    start: date
    end: date

    def split(self) -> list["_Window"]:
        """A year into quarters, a quarter into months; a month is final."""
        if len(self.label) == 4:
            return [
                _Window(
                    f"{self.label}-Q{q}",
                    date(self.start.year, 3 * q - 2, 1),
                    _month_end(self.start.year, 3 * q),
                )
                for q in range(1, 5)
            ]
        if "-Q" in self.label:
            return [
                _Window(
                    f"{self.start.year}-{m:02d}",
                    date(self.start.year, m, 1),
                    _month_end(self.start.year, m),
                )
                for m in range(self.start.month, self.start.month + 3)
            ]
        return []


def _month_end(year: int, month: int) -> date:
    following = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return date.fromordinal(following.toordinal() - 1)


def _datagroup_year_path(datagroup_code: str, year: int) -> Path:
    return DATAGROUPS_DATA_ROOT / datagroup_code / f"{year}.parquet"


def _window_path(datagroup_code: str, label: str) -> Path:
    return DATAGROUPS_DATA_ROOT / datagroup_code / f"{label}.parquet"


def _empty_marker(datagroup_code: str, label: str) -> Path:
    """Records that a window came back empty, so a second run makes no call
    for it (CLAUDE.md #6). Not a parquet: silver reads `*.parquet` only."""
    return DATAGROUPS_DATA_ROOT / datagroup_code / f"{label}.empty.json"


def _window_done(datagroup_code: str, window: _Window) -> bool:
    if _window_path(datagroup_code, window.label).exists():
        return True
    if _empty_marker(datagroup_code, window.label).exists():
        return True
    parts = window.split()
    return bool(parts) and all(_window_done(datagroup_code, part) for part in parts)


def _has_parts(datagroup_code: str, window: _Window) -> bool:
    """A window already being fetched piece by piece stays that way, so a
    rerun never mixes a whole-year file with some of its quarters."""
    return any(
        _window_path(datagroup_code, part.label).exists()
        or _empty_marker(datagroup_code, part.label).exists()
        or _has_parts(datagroup_code, part)
        for part in window.split()
    )


def fetch_datagroup_year(
    datagroup_code: str, year: int, settings: Settings, *, force: bool = False
) -> YearFetchResult:
    """Fetch one datagroup's raw (wide, bronze) data for one calendar year.

    Idempotent: skips the HTTP call entirely if the year is already on disk
    -- as one file, as an empty marker, or as complete quarter/month parts.
    A year that times out is fetched again as quarters, a quarter as months;
    each answer is stored as it came, in its own file. A truncated or
    failed window never writes a parquet, so a rerun with force=False
    fetches it again rather than silently accepting partial data. A 403
    (EvdsAuthError) is not caught here -- it propagates and is meant to stop
    the whole bulk run.
    """
    year_window = _Window(str(year), date(year, 1, 1), date(year, 12, 31))
    out_path = _datagroup_year_path(datagroup_code, year)
    if not force and _window_done(datagroup_code, year_window):
        return YearFetchResult(datagroup_code, year, "cached", None, out_path)

    outcomes = _fetch_window(
        datagroup_code,
        year_window,
        settings,
        force=force,
        split_first=_has_parts(datagroup_code, year_window),
    )
    rows = sum(count for _, count in outcomes if count)
    kinds = {outcome for outcome, _ in outcomes}
    for outcome in ("truncated", "failed"):
        if outcome in kinds:
            return YearFetchResult(datagroup_code, year, outcome, rows or None, None)
    if kinds <= {"empty", "cached"} and "empty" in kinds and not rows:
        return YearFetchResult(datagroup_code, year, "empty", 0, None)
    return YearFetchResult(datagroup_code, year, "ok", rows, out_path)


def _fetch_window(
    datagroup_code: str,
    window: _Window,
    settings: Settings,
    *,
    force: bool,
    split_first: bool = False,
) -> list[tuple[str, int | None]]:
    """Fetch `window`, splitting it on a timeout. Returns (outcome, rows)
    per request made or skipped."""
    if not force and (
        _window_path(datagroup_code, window.label).exists()
        or _empty_marker(datagroup_code, window.label).exists()
    ):
        return [("cached", None)]
    if split_first and window.split():
        return [
            outcome
            for part in window.split()
            for outcome in _fetch_window(
                datagroup_code,
                part,
                settings,
                force=force,
                split_first=_has_parts(datagroup_code, part),
            )
        ]

    path = DATAGROUP_DATA_PATH_TMPL.format(
        code=datagroup_code,
        start=window.start.strftime("%d-%m-%Y"),
        end=window.end.strftime("%d-%m-%Y"),
    )
    try:
        payload = _authenticated_get(path, settings)
    except EvdsTimeoutError:
        parts = window.split()
        if not parts:
            logger.warning(
                "EVDS %s/%s: ay parcasi da zaman asimina ugradi",
                datagroup_code,
                window.label,
            )
            return [("failed", None)]
        logger.info(
            "EVDS %s/%s zaman asimi; %d parcaya bolunuyor",
            datagroup_code,
            window.label,
            len(parts),
        )
        return [
            outcome
            for part in parts
            for outcome in _fetch_window(datagroup_code, part, settings, force=force)
        ]
    except (
        requests.exceptions.ConnectionError,
        EvdsRequestError,
        EvdsRedirectError,
    ) as exc:
        logger.warning(
            "EVDS fetch failed for %s/%s: %s", datagroup_code, window.label, exc
        )
        return [("failed", None)]

    df = pd.DataFrame(_extract_rows(payload))
    fetched_at = datetime.now(UTC).isoformat()
    if df.empty:
        marker = _empty_marker(datagroup_code, window.label)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"params": path, "fetched_at": fetched_at, "rows": 0}),
            encoding="utf-8",
        )
        return [("empty", 0)]

    row_count = len(df)
    total_count = payload.get("totalCount") if isinstance(payload, dict) else None
    if row_count >= 1000:
        logger.warning(
            "%s/%s: %d satır döndü (1000 sınırına yakın/eşit) -- ilk tarih kontrol ediliyor",
            datagroup_code,
            window.label,
            row_count,
        )

    if detect_truncation(df, window.start, total_count=total_count):
        logger.warning(
            "%s/%s: kesme tespit edildi (ilk satır %s, beklenen %s) -- parquet yazılmadı",
            datagroup_code,
            window.label,
            df["Tarih"].iloc[0],
            window.start,
        )
        return [("truncated", row_count)]

    df["_fetched_at"] = fetched_at
    out_path = _window_path(datagroup_code, window.label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return [("ok", row_count)]


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
