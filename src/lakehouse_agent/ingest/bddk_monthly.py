"""BDDK Aylık Bülten ingest client (plan 1.2).

Bronze layer only: every table of the bulletin, every month, stored as the
whole response with what was asked and when (`tables/tNN/YYYY-MM.json`).
Picking values out, reading units and Turkish number-format conversion
happen in lake/bddk.py, not here.

Verified 17 Sep 2026 with live requests:
- POST https://www.bddk.org.tr/BultenAylik/tr/Home/BasitRaporGetir
- form body: tabloNo, yil, ay, paraBirimi=TL, taraf=10001 (Sektör)
- no cookie needed
- BDDK's server does not send its intermediate certificate (a 1-cert chain);
  certs/bddk_chain.pem (leaf + GlobalSign intermediate + system CA) must be
  passed as `verify=`. `verify=False` is never acceptable (MITM exposure).
- response: {"success":..., "Json":{"colNames": [...], "caption": ...,
  "uyari": ..., "data":{"rows":[{"cell":[...]}]}}}; cell = [taraf, sira, ad,
  stil, value columns...], the value columns named by colNames[4:].
22 Sep 2026: tables 1..17 (the page's tabloNo list), all fetched; before
that only table 4 was, three rows of it, picked by label from a yaml.
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

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

BRONZE_ROOT = Path("data/bronze/bddk_aylik")
#: One JSON file per table x month, the whole response: tables/tNN/YYYY-MM.json.
TABLES_ROOT = BRONZE_ROOT / "tables"

#: The bulletin's tables, as its page lists them (measured 22 Sep 2026: the
#: tabloNo <select> on https://www.bddk.org.tr/BultenAylik/ runs 1..17, from
#: "Bilanço" to "Yurt Dışı Şube Rasyoları").
ALL_TABLES = tuple(range(1, 18))


class BddkRequestError(RuntimeError):
    """Raised when a BDDK request fails (non-2xx) or exhausts all retries."""


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


# --- whole tables (22 Sep 2026) ------------------------------------------


@dataclass(frozen=True)
class BddkTableResult:
    tablo_no: int
    year: int
    month: int
    outcome: str  # "cached" | "ok" | "failed"
    row_count: int
    path: Path | None


def table_path(tablo_no: int, year: int, month: int) -> Path:
    return TABLES_ROOT / f"t{tablo_no:02d}" / f"{year}-{month:02d}.json"


def fetch_table_month(
    tablo_no: int, year: int, month: int, *, force: bool = False
) -> BddkTableResult:
    """Fetch one table of the bulletin for one month and store the response
    whole, with what was asked and when (bronze: nothing is picked out or
    converted here). Skips the call if the file exists. A network or
    request error is logged and returned as "failed"; nothing is written.
    """
    out = table_path(tablo_no, year, month)
    if out.exists() and not force:
        return BddkTableResult(tablo_no, year, month, "cached", 0, out)
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
        return BddkTableResult(tablo_no, year, month, "failed", 0, None)

    rows = payload.get("Json", {}).get("data", {}).get("rows", []) or []
    record = {
        "url": BASE_URL,
        "params": {
            "tabloNo": tablo_no,
            "yil": year,
            "ay": month,
            "paraBirimi": PARA_BIRIMI_TL,
            "taraf": TARAF_SEKTOR,
        },
        "http_status": 200,
        "fetched_at": datetime.now(UTC).isoformat(),
        "payload": payload,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return BddkTableResult(tablo_no, year, month, "ok", len(rows), out)


def fetch_tables(
    since: str,
    until: str,
    tables: tuple[int, ...] = ALL_TABLES,
    *,
    force: bool = False,
) -> list[BddkTableResult]:
    """Every table x every month in ["since", "until"] (inclusive, "YYYY-MM"),
    carrying on past a failure; the caller reports failures at the end."""
    results = []
    for year, month in _month_range(since, until):
        for tablo_no in tables:
            results.append(fetch_table_month(tablo_no, year, month, force=force))
        logger.info("BDDK aylik: %d-%02d tamam", year, month)
    return results
