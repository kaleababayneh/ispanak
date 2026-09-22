"""BDDK FinTürk (İllere Göre Seçilmiş Bankacılık Sektörü Verileri) ingest
client (plan 1.3).

Bronze layer only: the whole response per table x quarter, with what was
asked and when (`tNN/YYYY-M.json`). Values, units and series ids are read in
lake/bddk.py.

Measured 22 Sep 2026 (https://www.bddk.org.tr/BultenFinTurk):
- POST /BultenFinturk/tr/Home/VeriGetir, form body tabloNo, donem ("2021-3",
  quarterly: months 3/6/9/12), tarafList[0]=10001 (sector),
  sehirList[0]=HEPSİ -- one call returns every province (82 rows, "YURT
  DIŞI" included);
- response like the monthly bulletin's: {"Json": {"colNames": [Eftodu, Yıl,
  Ay, Şehir, Grup, value columns...], "data": {"rows": [{"cell": [...]}]}}};
- seven tables, named with their unit: "Krediler (Bin TL)" .. "Altın
  Kredileri ve Altın Mevduatı (Bin TL)".
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from lakehouse_agent.ingest.bddk_monthly import CERT_PATH, TARAF_SEKTOR

logger = logging.getLogger(__name__)

URL = "https://www.bddk.org.tr/BultenFinturk/tr/Home/VeriGetir"
HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.bddk.org.tr/BultenFinTurk",
    "Origin": "https://www.bddk.org.tr",
}
ALL_PROVINCES = "HEPSİ"

#: The page's table list (ddlTablo), 22 Sep 2026.
TABLES = {
    1: "Krediler (Bin TL)",
    2: "Mevduat (Bin TL)",
    3: "Bireysel Bankacılık (Bin TL)",
    4: "Seçilmiş Sektörel Krediler (Bin TL)",
    5: "Oranlar (%)",
    6: "Şubeler (Adet) ve Nüfusa Göre Dağılım (TL)",
    7: "Altın Kredileri ve Altın Mevduatı (Bin TL)",
}

RATE_LIMIT_RANGE = (0.3, 0.5)
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = 60

BRONZE_ROOT = Path("data/bronze/bddk_finturk")


class BddkFinturkError(RuntimeError):
    """A FinTürk request failed or exhausted its retries."""


@dataclass(frozen=True)
class FinturkResult:
    tablo_no: int
    year: int
    month: int
    outcome: str  # "cached" | "ok" | "failed"
    row_count: int
    path: Path | None


def quarter_path(tablo_no: int, year: int, month: int) -> Path:
    return BRONZE_ROOT / f"t{tablo_no:02d}" / f"{year}-{month}.json"


def _post(tablo_no: int, year: int, month: int) -> dict:
    data = {
        "tabloNo": tablo_no,
        "donem": f"{year}-{month}",
        "tarafList[0]": TARAF_SEKTOR,
        "sehirList[0]": ALL_PROVINCES,
    }
    last: int | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(random.uniform(*RATE_LIMIT_RANGE))
        response = requests.post(
            URL,
            data=data,
            headers=HEADERS,
            verify=str(CERT_PATH),
            timeout=REQUEST_TIMEOUT,
        )
        last = response.status_code
        if response.status_code == 200:
            return response.json()
        if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))
            continue
        break
    raise BddkFinturkError(
        f"FinTürk isteği başarısız (tabloNo={tablo_no}, {year}-{month}, status={last})"
    )


def fetch_quarter(
    tablo_no: int, year: int, month: int, *, force: bool = False
) -> FinturkResult:
    """One table for one quarter (`month` 3/6/9/12), every province, stored
    whole. Skips the call if the file exists; a failure writes nothing."""
    out = quarter_path(tablo_no, year, month)
    if out.exists() and not force:
        return FinturkResult(tablo_no, year, month, "cached", 0, out)
    try:
        payload = _post(tablo_no, year, month)
    except (requests.RequestException, BddkFinturkError, ValueError) as exc:
        logger.warning("FinTürk t%s %d-%d: %s", tablo_no, year, month, exc)
        return FinturkResult(tablo_no, year, month, "failed", 0, None)

    rows = payload.get("Json", {}).get("data", {}).get("rows", []) or []
    record = {
        "url": URL,
        "params": {
            "tabloNo": tablo_no,
            "donem": f"{year}-{month}",
            "taraf": TARAF_SEKTOR,
            "sehir": ALL_PROVINCES,
        },
        "table": TABLES.get(tablo_no),
        "http_status": 200,
        "fetched_at": datetime.now(UTC).isoformat(),
        "payload": payload,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return FinturkResult(tablo_no, year, month, "ok", len(rows), out)


def quarters(since: str, until: str) -> list[tuple[int, int]]:
    """Quarter-end (year, month) pairs whose month is in ["since", "until"]."""
    first = (int(since[:4]), int(since[5:7]))
    last = (int(until[:4]), int(until[5:7]))
    return [
        (year, month)
        for year in range(first[0], last[0] + 1)
        for month in (3, 6, 9, 12)
        if first <= (year, month) <= last
    ]


def fetch_finturk(
    since: str, until: str, *, force: bool = False
) -> list[FinturkResult]:
    results = []
    for year, month in quarters(since, until):
        for tablo_no in TABLES:
            results.append(fetch_quarter(tablo_no, year, month, force=force))
        logger.info("FinTürk: %d-%d tamam", year, month)
    return results
