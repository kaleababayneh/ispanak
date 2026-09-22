"""Tests for ingest/bddk_monthly.py. No network: `_post_report` is mocked."""

import json

import pytest

from lakehouse_agent.ingest import bddk_monthly

PAYLOAD = {
    "Json": {
        "colNames": ["", "", "", "BasitFont", "TP", "YP", "Toplam"],
        "caption": "Tüketici Kredileri (milyon TL), Dönem:2024/1",
        "data": {
            "rows": [
                {
                    "cell": [
                        "Sektör",
                        2,
                        "Tüketici Kredileri - Konut",
                        "",
                        594545,
                        62,
                        594607,
                    ]
                }
            ]
        },
    }
}


@pytest.fixture(autouse=True)
def _in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def test_a_table_month_is_stored_whole_with_what_was_asked(monkeypatch):
    """Bronze: the response as it came, nothing picked out or converted."""
    monkeypatch.setattr(bddk_monthly, "_post_report", lambda *_args: PAYLOAD)

    result = bddk_monthly.fetch_table_month(4, 2024, 1)

    assert (result.outcome, result.row_count) == ("ok", 1)
    record = json.loads(result.path.read_text(encoding="utf-8"))
    assert record["payload"] == PAYLOAD
    assert record["params"] == {
        "tabloNo": 4,
        "yil": 2024,
        "ay": 1,
        "paraBirimi": "TL",
        "taraf": 10001,
    }
    assert record["url"] == bddk_monthly.BASE_URL
    assert record["fetched_at"]


def test_a_stored_table_month_is_not_fetched_again(monkeypatch):
    """CLAUDE.md #6: the second run makes no call."""
    calls = []

    def post(*args):
        calls.append(args)
        return PAYLOAD

    monkeypatch.setattr(bddk_monthly, "_post_report", post)

    bddk_monthly.fetch_table_month(4, 2024, 1)
    second = bddk_monthly.fetch_table_month(4, 2024, 1)

    assert len(calls) == 1
    assert second.outcome == "cached"


def test_a_failed_request_writes_nothing(monkeypatch):
    def fail(*_args):
        raise bddk_monthly.BddkRequestError("503")

    monkeypatch.setattr(bddk_monthly, "_post_report", fail)

    result = bddk_monthly.fetch_table_month(4, 2024, 1)

    assert result.outcome == "failed"
    assert not bddk_monthly.table_path(4, 2024, 1).exists()


def test_fetch_tables_covers_every_table_and_month(monkeypatch):
    seen = []
    monkeypatch.setattr(
        bddk_monthly, "_post_report", lambda t, y, m: seen.append((t, y, m)) or PAYLOAD
    )

    results = bddk_monthly.fetch_tables("2024-12", "2025-01", tables=(1, 4))

    assert seen == [(1, 2024, 12), (4, 2024, 12), (1, 2025, 1), (4, 2025, 1)]
    assert all(r.outcome == "ok" for r in results)


def test_the_bulletin_has_seventeen_tables():
    """The page's tabloNo list, 22 Sep 2026: Bilanço .. Yurt Dışı Şube Rasyoları."""
    assert bddk_monthly.ALL_TABLES == tuple(range(1, 18))
