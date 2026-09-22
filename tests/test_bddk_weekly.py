"""Tests for ingest/bddk_weekly.py and ingest/bddk_finturk.py. No network."""

import json
from datetime import date

import pytest

from lakehouse_agent.ingest import bddk_finturk, bddk_weekly

PAGE = """
<form action="/BultenHaftalik/tr/Home/DonemDegistir" class="form-inline" method="post"><input name="__RequestVerificationToken" type="hidden" value="tok1" />
<select id="Donem" name="donemId">
<option value="368" class="Yil-2021 YilDonem" >Ocak/15 (2. Hafta)
</option>
<option value="367" class="Yil-2021 YilDonem" >Ocak/08 (1. Hafta)
</option>
<option value="366" class="Yil-2020 YilDonem" style="display:none;">Aral&#305;k/31 (53. Hafta)
</option>
</select></form>
<td onclick="TabloDegistir('289')" class="tabloListesiItem"> <span class="text">Krediler</span></td>
<td onclick="TabloDegistir('292')" class="tabloListesiItem"> <span class="text">Mevduat</span></td>
"""


def _table_page(name, day_text):
    return (
        '<form action="/BultenHaftalik/" method="post"><input name="__RequestVerificationToken" '
        'type="hidden" value="tok2" /></form>'
        '<table class="t" id="TabloExcel2"><thead><tr><th></th>'
        f"<th>Sekt&#246;r / {name} ( {day_text} <text>) (Milyon TL)</text></th>"
        "<th>TP</th></tr></thead><tr><td>1</td><td>X</td><td>1</td></tr></table>"
    )


@pytest.fixture(autouse=True)
def _in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def test_the_page_lists_weeks_with_their_dates_and_the_tables():
    weeks = bddk_weekly.parse_weeks(PAGE)

    assert [(w.donem_id, w.day) for w in weeks] == [
        (366, date(2020, 12, 31)),
        (367, date(2021, 1, 8)),
        (368, date(2021, 1, 15)),
    ]
    assert bddk_weekly.parse_tables(PAGE) == [(289, "Krediler"), (292, "Mevduat")]


def test_the_caption_date_is_read():
    fragment = bddk_weekly.table_fragment(_table_page("Krediler", "8 Ocak 2021 Cuma"))

    assert bddk_weekly.caption_day(fragment) == date(2021, 1, 8)


class _FakeSession:
    """Serves the table the session was last switched to, for the week it
    was switched to -- or, when `lie` is set, always 8 January."""

    def __init__(self, lie=False):
        self.page = ""
        self.lie = lie
        self.day = None
        self.calls = []

    def open(self):
        self.calls.append("open")
        self.page = PAGE
        return PAGE

    def week(self, week):
        self.calls.append(("week", week.day))
        self.day = week.day
        return self.page

    def table(self, table_id):
        self.calls.append(("table", table_id))
        name = {289: "Krediler", 292: "Mevduat"}[table_id]
        day = date(2021, 1, 8) if self.lie else self.day
        months = {1: "Ocak"}
        return _table_page(name, f"{day.day} {months[day.month]} {day.year} Cuma")


def test_each_table_week_is_stored_and_a_rerun_makes_no_call(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(bddk_weekly, "_Session", lambda: session)

    results = bddk_weekly.fetch_weeks("2021-01", "2021-01")

    assert sorted((r.table_id, r.day) for r in results if r.outcome == "ok") == [
        (289, date(2021, 1, 8)),
        (289, date(2021, 1, 15)),
        (292, date(2021, 1, 8)),
        (292, date(2021, 1, 15)),
    ]
    record = json.loads(bddk_weekly.week_path(289, date(2021, 1, 8)).read_text())
    assert record["html"].startswith("<table")
    assert record["params"]["donemId"] == 367

    session.calls.clear()
    again = bddk_weekly.fetch_weeks("2021-01", "2021-01")
    assert session.calls == []  # the listing is stored too
    assert {r.outcome for r in again} == {"cached"}


def test_a_page_for_another_week_is_not_filed_under_this_one(monkeypatch):
    """The choice lives in a server session; if it serves the wrong week, the
    caption says so and nothing is written."""
    monkeypatch.setattr(bddk_weekly, "_Session", lambda: _FakeSession(lie=True))

    results = bddk_weekly.fetch_weeks("2021-01", "2021-01")

    wrong = [r for r in results if r.day == date(2021, 1, 15)]
    assert {r.outcome for r in wrong} == {"failed"}
    assert not bddk_weekly.week_path(289, date(2021, 1, 15)).exists()


# --- FinTürk ---------------------------------------------------------------------


def test_finturk_quarters_are_the_quarter_end_months_in_the_window():
    assert bddk_finturk.quarters("2021-01", "2021-12") == [
        (2021, 3),
        (2021, 6),
        (2021, 9),
        (2021, 12),
    ]
    assert bddk_finturk.quarters("2026-01", "2026-06") == [(2026, 3), (2026, 6)]


def test_a_finturk_quarter_is_stored_whole_and_not_fetched_again(monkeypatch):
    payload = {"Json": {"colNames": [], "data": {"rows": [{"cell": [1]}]}}}
    calls = []
    monkeypatch.setattr(bddk_finturk, "_post", lambda *a: calls.append(a) or payload)

    first = bddk_finturk.fetch_quarter(1, 2021, 3)
    second = bddk_finturk.fetch_quarter(1, 2021, 3)

    assert (first.outcome, first.row_count, second.outcome) == ("ok", 1, "cached")
    assert len(calls) == 1
    record = json.loads(first.path.read_text(encoding="utf-8"))
    assert record["payload"] == payload
    assert record["params"]["sehir"] == "HEPSİ"
