"""BDDK Aylık Bülten ingest client (plan 1.2).

Bronze layer only: the exact `cell` array from `BasitRaporGetir` is stored as
a JSON string, untouched. Selecting a value by `cell_index` and any Turkish
number-format conversion happens in lake/silver.py, not here.

Verified 17 Sep 2026 with live requests:
- POST https://www.bddk.org.tr/BultenAylik/tr/Home/BasitRaporGetir
- form body: tabloNo, yil, ay, paraBirimi=TL, taraf=10001 (Sektör)
- no cookie needed
- BDDK's server does not send its intermediate certificate (a 1-cert chain);
  certs/bddk_chain.pem (leaf + GlobalSign intermediate + system CA) must be
  passed as `verify=`. `verify=False` is never acceptable (MITM exposure).
- response: {"success":..., "Json":{"data":{"rows":[{"cell":[...]}]}}}
  cell = [taraf, sira, ad, stil, TP, YP, TOPLAM]
"""

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

BASE_URL = "https://www.bddk.org.tr/BultenAylik/tr/Home/BasitRaporGetir"
CERT_PATH = Path("certs/bddk_chain.pem")
HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.bddk.org.tr/BultenAylik/",
    "Origin": "https://www.bddk.org.tr",
}
TARAF_SEKTOR = 10001
PARA_BIRIMI_TL = "TL"

RATE_LIMIT_RANGE = (0.3, 0.5)  # seconds, slept before every single attempt
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = 30

ROWS_YAML_PATH = Path("ingest/bddk_rows.yaml")
BRONZE_ROOT = Path("data/bronze/bddk_aylik")
RAW_ROOT = BRONZE_ROOT / "raw"

# cell = [taraf, sira, ad, stil, TP, YP, TOPLAM] -- documents what cell_index
# refers to; indexing itself happens in lake/silver.py.
CELL_FIELDS = ["taraf", "sira", "ad", "stil", "TP", "YP", "TOPLAM"]

BRONZE_MONTH_COLUMNS = [
    "yil",
    "ay",
    "tablo_no",
    "satir_etiketi",
    "series_id",
    "cell_json",
    "url",
    "http_status",
    "fetched_at",
]


class BddkRowMismatchError(RuntimeError):
    """Raised when an expected satir_etiketi is missing from a fetched table.

    This means the bulletin's row layout changed -- never silently skipped,
    the whole fetch must stop so a human updates bddk_rows.yaml.
    """


class BddkRequestError(RuntimeError):
    """Raised when a BDDK request fails (non-2xx) or exhausts all retries."""


@dataclass(frozen=True)
class BddkRow:
    tablo_no: int
    satir_etiketi: str
    series_id: str
    cell_index: int
    kind: str
    unit: str
    cum_type: str


def load_row_definitions(path: Path = ROWS_YAML_PATH) -> list[BddkRow]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [BddkRow(**row) for row in raw]


def find_row(payload: dict, satir_etiketi: str) -> list:
    """Find the row whose `ad` field (cell[2]) matches `satir_etiketi`.

    Raises BddkRowMismatchError if not found -- the bulletin's row layout may
    have changed since bddk_rows.yaml was written.
    """
    rows = payload.get("Json", {}).get("data", {}).get("rows", [])
    for row in rows:
        cell = row.get("cell", [])
        if len(cell) > 2 and str(cell[2]).strip() == satir_etiketi:
            return cell
    raise BddkRowMismatchError(
        f"'{satir_etiketi}' satırı bulunamadı (tablo düzeni değişmiş olabilir)"
    )


def _post_report(tablo_no: int, year: int, month: int) -> dict:
    """Single POST to BasitRaporGetir, with retry/backoff on 429/5xx.

    No auth header -- BDDK's public bulletin form does not require a key.
    """
    data = {
        "tabloNo": tablo_no,
        "yil": year,
        "ay": month,
        "paraBirimi": PARA_BIRIMI_TL,
        "taraf": TARAF_SEKTOR,
    }
    last_status: int | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(random.uniform(*RATE_LIMIT_RANGE))
        response = requests.post(
            BASE_URL,
            data=data,
            headers=HEADERS,
            verify=str(CERT_PATH),
            timeout=REQUEST_TIMEOUT,
        )
        last_status = response.status_code

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))
            continue

        raise BddkRequestError(
            f"BDDK isteği başarısız (tabloNo={tablo_no}, {year}-{month:02d}, "
            f"status={response.status_code})"
        )

    raise BddkRequestError(
        f"BDDK isteği {MAX_RETRIES} denemeden sonra başarısız "
        f"(tabloNo={tablo_no}, {year}-{month:02d}, son status={last_status})"
    )


@dataclass(frozen=True)
class BddkMonthResult:
    year: int
    month: int
    outcome: str  # "cached" | "ok" | "failed"
    row_count: int
    path: Path | None


def _month_path(year: int, month: int) -> Path:
    return BRONZE_ROOT / f"{year}-{month:02d}.parquet"


def _raw_path(year: int, month: int) -> Path:
    return RAW_ROOT / f"{year}-{month:02d}.json"


def fetch_month(
    year: int, month: int, row_defs: list[BddkRow], *, force: bool = False
) -> BddkMonthResult:
    """Fetch every distinct tabloNo referenced by `row_defs` for one
    (year, month) and write the found rows to bronze.

    Idempotent: skips entirely (no HTTP) if the month's parquet already
    exists and force=False. A BddkRowMismatchError propagates uncaught --
    the bulletin layout changed and nothing is written for this month. A
    network/request error is caught, logged, and returned as "failed" so
    the caller can move on to the next month.
    """
    out_path = _month_path(year, month)
    if out_path.exists() and not force:
        return BddkMonthResult(year, month, "cached", 0, out_path)

    by_tablo: dict[int, list[BddkRow]] = defaultdict(list)
    for row_def in row_defs:
        by_tablo[row_def.tablo_no].append(row_def)

    raw_payloads: dict[str, dict] = {}
    rows_out: list[dict] = []
    fetched_at = datetime.now(UTC).isoformat()

    for tablo_no, defs in by_tablo.items():
        try:
            payload = _post_report(tablo_no, year, month)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            BddkRequestError,
        ) as exc:
            logger.warning(
                "BDDK fetch failed tabloNo=%s %d-%02d: %s", tablo_no, year, month, exc
            )
            return BddkMonthResult(year, month, "failed", 0, None)

        raw_payloads[str(tablo_no)] = payload

        for row_def in defs:
            cell = find_row(payload, row_def.satir_etiketi)  # raises, not caught
            rows_out.append(
                {
                    "yil": year,
                    "ay": month,
                    "tablo_no": tablo_no,
                    "satir_etiketi": row_def.satir_etiketi,
                    "series_id": row_def.series_id,
                    "cell_json": json.dumps(cell, ensure_ascii=False),
                    "url": BASE_URL,
                    "http_status": 200,
                    "fetched_at": fetched_at,
                }
            )

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    _raw_path(year, month).write_text(
        json.dumps(raw_payloads, ensure_ascii=False), encoding="utf-8"
    )

    df = pd.DataFrame(rows_out, columns=BRONZE_MONTH_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return BddkMonthResult(year, month, "ok", len(df), out_path)


def _month_range(since: str, until: str) -> list[tuple[int, int]]:
    """ "YYYY-MM" .. "YYYY-MM" (inclusive) -> [(year, month), ...]."""
    since_year, since_month = (int(p) for p in since.split("-"))
    until_year, until_month = (int(p) for p in until.split("-"))
    months = []
    year, month = since_year, since_month
    while (year, month) <= (until_year, until_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def fetch_months(
    since: str, until: str, row_defs: list[BddkRow], *, force: bool = False
) -> list[BddkMonthResult]:
    """Fetch every (year, month) in ["since", "until"] (inclusive, "YYYY-MM").

    Keeps going past a "failed" outcome; a BddkRowMismatchError still
    propagates and stops the whole run.
    """
    results = []
    for year, month in _month_range(since, until):
        results.append(fetch_month(year, month, row_defs, force=force))
    return results
