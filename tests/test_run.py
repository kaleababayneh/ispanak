"""Tests for ingest/run.py: --since/--until CLI validation. No network."""

import pytest

from lakehouse_agent.ingest import run


def test_parse_year_month_rejects_garbage_instead_of_a_bare_traceback():
    with pytest.raises(SystemExit) as exc:
        run.build_parser().parse_args(
            ["--source", "evds", "--all", "--since", "abc", "--until", "2021-01"]
        )
    assert exc.value.code == 2


def test_parse_year_month_rejects_invalid_month():
    # Old _year_range('2021-13', '2021-01') silently accepted the bad month.
    with pytest.raises(SystemExit) as exc:
        run.build_parser().parse_args(
            ["--source", "evds", "--all", "--since", "2021-13", "--until", "2021-01"]
        )
    assert exc.value.code == 2


def test_main_rejects_reversed_since_until_range(monkeypatch):
    """B'nin ölçtüğü sessiz kayıp: _year_range('2026-06', '2021-01') -> []
    -- ters aralık, hiçbir şey çekilmeden exit 0. Artık parser.error() ile
    reddedilmeli (exit 2), sessizce boş listeye düşmemeli."""
    monkeypatch.setattr(run, "load_dotenv", lambda *_a, **_k: None)

    with pytest.raises(SystemExit) as exc:
        run.main(
            ["--source", "evds", "--all", "--since", "2026-06", "--until", "2021-01"]
        )
    assert exc.value.code == 2


def test_year_range_still_works_for_a_valid_ascending_range():
    assert run._year_range("2021-01", "2021-12") == [2021]
    assert run._year_range("2021-06", "2023-02") == [2021, 2022, 2023]
