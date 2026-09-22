"""BDDK Haftalık Bülten ingest client (plan 1.3).

Bronze layer only. The weekly bulletin has no JSON report endpoint like the
monthly one: the page renders the chosen week's chosen table as HTML, and
the choice lives in the server-side session. Measured 22 Sep 2026:

- GET https://www.bddk.org.tr/BultenHaftalik/ lists the weeks in a
  <select name="donemId"> (value = period id, class "Yil-YYYY", text
  "Ocak/08 (1. Hafta)") and the tables as TabloDegistir('289'..'297').
- POST /BultenHaftalik/tr/Home/DonemDegistir {token, yil, donemId, para}
  switches the week; POST /BultenHaftalik/ {token, tabloId} the table. Each
  form carries an anti-forgery token, and the session cookie carries the
  choice. Sector (taraf 10001) is the default side.
- The table is <table id="TabloExcel2">: a caption row ("Sektör / Krediler
  ( 8 Ocak 2021 Cuma ) (Milyon TL)"), a header row (TP, YP, TOPLAM) and one
  row per item: sira, label, values in Turkish number format.
- The time-series endpoint (KiyaslamaJsonGetir) returns 13 weeks per call
  whatever is asked -- ~23 calls per series -- so it is not used.

What is stored per table x week is the table element exactly as it came
(the rest of the page is the same navigation and scripts every time), with
the request, the status and the time. Before writing, the caption is
checked against the week and table asked for: a session that silently
served another week must not end up filed under this one.
"""

from __future__ import annotations

import html
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from lakehouse_agent.ingest.bddk_monthly import CERT_PATH

logger = logging.getLogger(__name__)

SITE = "https://www.bddk.org.tr"
PAGE_URL = SITE + "/BultenHaftalik/"
DONEM_URL = SITE + "/BultenHaftalik/tr/Home/DonemDegistir"
HEADERS = {"Referer": PAGE_URL, "Origin": SITE}

RATE_LIMIT_RANGE = (0.3, 0.5)
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = 60

BRONZE_ROOT = Path("data/bronze/bddk_haftalik")
#: The page's list of weeks and tables, as parsed on the first run: a rerun
#: over the same window needs no call to learn what to look for.
LISTING_PATH = BRONZE_ROOT / "listing.json"

_MONTHS = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}
_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")

_WEEK_OPTION = re.compile(
    r'<option value="(\d+)" class="Yil-(\d{4}) YilDonem"[^>]*>\s*([^<]*)', re.S
)
_TABLE_ITEM = re.compile(
    r"TabloDegistir\('(\d+)'\)[^>]*>\s*<span class=\"text\">([^<]*)"
)
_FORM_TOKEN = re.compile(
    r'<form action="([^"]+)"[^>]*>\s*<input name="__RequestVerificationToken" '
    r'type="hidden" value="([^"]+)"'
)
_CAPTION_DATE = re.compile(r"\(\s*(\d{1,2})\s+(\w+)\s+(\d{4})\s+\w+\s*\)")


class BddkWeeklyError(RuntimeError):
    """The page did not come back as expected (no table, wrong week)."""


@dataclass(frozen=True)
class Week:
    donem_id: int
    day: date
    label: str


@dataclass(frozen=True)
class WeeklyResult:
    table_id: int
    day: date
    outcome: str  # "cached" | "ok" | "failed"
    path: Path | None


def _fold(text: str) -> str:
    return html.unescape(text).strip().translate(_FOLD).lower()


def parse_weeks(page: str) -> list[Week]:
    """Every week the page offers, dated from its label and its year."""
    weeks = []
    for value, year, text in _WEEK_OPTION.findall(page):
        label = " ".join(html.unescape(text).split())
        match = re.match(r"(\w+)/(\d{1,2})", _fold(label))
        if not match or match.group(1) not in _MONTHS:
            continue
        day = date(int(year), _MONTHS[match.group(1)], int(match.group(2)))
        weeks.append(Week(int(value), day, label))
    return sorted(weeks, key=lambda week: week.day)


def parse_tables(page: str) -> list[tuple[int, str]]:
    return [
        (int(table_id), " ".join(html.unescape(name).split()))
        for table_id, name in _TABLE_ITEM.findall(page)
    ]


def table_fragment(page: str) -> str:
    """The <table id="TabloExcel2"> element, verbatim."""
    marker = page.find('id="TabloExcel2"')
    start = page.rfind("<table", 0, marker) if marker >= 0 else -1
    end = page.find("</table>", start)
    if start < 0 or end < 0:
        raise BddkWeeklyError("sayfada TabloExcel2 tablosu yok")
    return page[start : end + len("</table>")]


def caption_day(fragment: str) -> date | None:
    """The date in the table's caption: "( 8 Ocak 2021 Cuma )"."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment[:2000]))
    match = _CAPTION_DATE.search(text)
    if not match or _fold(match.group(2)) not in _MONTHS:
        return None
    return date(
        int(match.group(3)), _MONTHS[_fold(match.group(2))], int(match.group(1))
    )


def week_path(table_id: int, day: date) -> Path:
    return BRONZE_ROOT / f"t{table_id}" / f"{day.isoformat()}.json"


class _Session:
    """A requests session that keeps the page's form tokens current."""

    def __init__(self) -> None:
        self.http = requests.Session()
        self.http.verify = str(CERT_PATH)
        self.http.headers.update(HEADERS)
        self.tokens: dict[str, str] = {}
        self.page = ""

    def _send(self, method: str, url: str, data: dict | None = None) -> str:
        last: int | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(random.uniform(*RATE_LIMIT_RANGE))
            response = self.http.request(
                method, url, data=data, timeout=REQUEST_TIMEOUT
            )
            last = response.status_code
            if response.status_code == 200:
                self.page = response.text
                self.tokens = dict(_FORM_TOKEN.findall(self.page))
                return self.page
            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            break
        raise BddkWeeklyError(
            f"BDDK haftalik istegi basarisiz ({method} {url}, {last})"
        )

    def open(self) -> str:
        return self._send("GET", PAGE_URL)

    def week(self, week: Week) -> str:
        token = self.tokens.get("/BultenHaftalik/tr/Home/DonemDegistir", "")
        data = {
            "__RequestVerificationToken": token,
            "yil": week.day.year,
            "donemId": week.donem_id,
            "para": "TL",
        }
        return self._send("POST", DONEM_URL, data)

    def table(self, table_id: int) -> str:
        token = self.tokens.get("/BultenHaftalik/", "")
        return self._send(
            "POST", PAGE_URL, {"__RequestVerificationToken": token, "tabloId": table_id}
        )


def fetch_weeks(since: str, until: str, *, force: bool = False) -> list[WeeklyResult]:
    """Every table of every week whose date is in ["since", "until"]
    (inclusive "YYYY-MM"). Skips what is stored; a week with every table
    stored makes no call at all (CLAUDE.md #6)."""
    first = date(int(since[:4]), int(since[5:7]), 1)
    last_year, last_month = int(until[:4]), int(until[5:7])
    after = date(last_year + (last_month == 12), last_month % 12 + 1, 1)

    session = _Session()
    all_weeks, tables = _listing(session, refresh=force)
    weeks = [week for week in all_weeks if first <= week.day < after]
    if not weeks or not tables:
        raise BddkWeeklyError("sayfada hafta ya da tablo listesi yok")
    logger.info("BDDK haftalik: %d hafta x %d tablo", len(weeks), len(tables))

    results: list[WeeklyResult] = []
    for index, week in enumerate(weeks, start=1):
        todo = [
            (tid, name)
            for tid, name in tables
            if force or not week_path(tid, week.day).exists()
        ]
        results += [
            WeeklyResult(tid, week.day, "cached", week_path(tid, week.day))
            for tid, _ in tables
            if (tid, _) not in todo
        ]
        if not todo:
            continue
        try:
            if not session.page:
                session.open()
            session.week(week)
        except (requests.RequestException, BddkWeeklyError) as exc:
            logger.warning("BDDK haftalik %s: hafta secilemedi: %s", week.day, exc)
            results += [WeeklyResult(tid, week.day, "failed", None) for tid, _ in todo]
            continue
        for table_id, name in todo:
            results.append(_fetch_table(session, week, table_id, name))
        if index % 10 == 0:
            logger.info("BDDK haftalik: %d/%d hafta", index, len(weeks))
    return results


def _listing(
    session: _Session, *, refresh: bool
) -> tuple[list[Week], list[tuple[int, str]]]:
    """Weeks and tables: from the stored listing, else from the page (and
    stored). A window reaching past the stored listing's last week asks
    again, since the bulletin adds a week every Friday."""
    if not refresh and LISTING_PATH.exists():
        stored = json.loads(LISTING_PATH.read_text(encoding="utf-8"))
        weeks = [
            Week(item["donem_id"], date.fromisoformat(item["day"]), item["label"])
            for item in stored["weeks"]
        ]
        return weeks, [(item["id"], item["name"]) for item in stored["tables"]]
    page = session.open()
    weeks, tables = parse_weeks(page), parse_tables(page)
    LISTING_PATH.parent.mkdir(parents=True, exist_ok=True)
    LISTING_PATH.write_text(
        json.dumps(
            {
                "url": PAGE_URL,
                "fetched_at": datetime.now(UTC).isoformat(),
                "weeks": [
                    {"donem_id": w.donem_id, "day": w.day.isoformat(), "label": w.label}
                    for w in weeks
                ],
                "tables": [{"id": tid, "name": name} for tid, name in tables],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return weeks, tables


def _fetch_table(
    session: _Session, week: Week, table_id: int, name: str
) -> WeeklyResult:
    try:
        page = session.table(table_id)
        fragment = table_fragment(page)
    except (requests.RequestException, BddkWeeklyError) as exc:
        logger.warning("BDDK haftalik %s t%s: %s", week.day, table_id, exc)
        return WeeklyResult(table_id, week.day, "failed", None)

    shown = caption_day(fragment)
    caption = _fold(re.sub(r"<[^>]+>", " ", fragment[:2000]))
    if shown != week.day or _fold(name) not in caption:
        logger.warning(
            "BDDK haftalik: %s t%s istendi, sayfa %s / baska tablo dondu -- yazilmadi",
            week.day,
            table_id,
            shown,
        )
        return WeeklyResult(table_id, week.day, "failed", None)

    record = {
        "url": PAGE_URL,
        "params": {
            "donemId": week.donem_id,
            "tabloId": table_id,
            "para": "TL",
            "taraf": 10001,
        },
        "week": {"label": week.label, "day": week.day.isoformat()},
        "table": {"id": table_id, "name": name},
        "http_status": 200,
        "fetched_at": datetime.now(UTC).isoformat(),
        "html": fragment,
    }
    path = week_path(table_id, week.day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return WeeklyResult(table_id, week.day, "ok", path)
