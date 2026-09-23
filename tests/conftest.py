"""Shared fixtures.

The important one is `_no_secrets_in_env`: it runs for every test and strips the
project's API keys out of the environment. A unit test that quietly reaches the
real MIA or EVDS would be slow, flaky and would burn quota we need for the demo,
so the suite is built so that it cannot happen by accident. Tests that need a
key set one explicitly.

The fake-layer fixtures import inside their own bodies rather than at module
level, so this file still collects cleanly in a tree where
`lakehouse_agent.testing` is not present yet.
"""

from pathlib import Path

import pandas as pd
import pytest

from lakehouse_agent.config import AgentSettings


@pytest.fixture(autouse=True)
def _no_secrets_in_env(monkeypatch):
    """No test may inherit a real key from the developer's shell."""
    for name in ("MIA_API_KEY", "EVDS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True, scope="session")
def _gold_db_outside_the_repo(tmp_path_factory):
    """Tests build their DuckDB copy of gold in a temporary directory.

    `gold.connect()` materialises the parquet files as tables inside
    `GOLD_DB_PATH`, which is relative to the working directory: every test
    that touches real gold was writing an 800 KB file into `data/gold/` in
    the working tree. It is gitignored, so `git status` never showed it and
    nobody noticed. The parquet files themselves are only ever read.

    Session-scoped, so the tables are built once for the whole suite rather
    than per test, and `raising=False` is not needed: the attribute exists.
    """
    from lakehouse_agent.lake import gold

    original = gold.GOLD_DB_PATH
    gold.GOLD_DB_PATH = tmp_path_factory.mktemp("gold") / "catalog.duckdb"
    yield
    gold.GOLD_DB_PATH = original


@pytest.fixture(autouse=True, scope="session")
def _no_pool_index_from_the_machine(tmp_path_factory):
    """No test may use the search index that happens to be on this machine.

    `data/pool_index` is derived and gitignored: it exists wherever someone
    ran the builder and nowhere else, CI included. A search behaves
    differently with and without it -- the keyword fallback takes the index's
    full-text order rather than walking the catalog -- so a test that reads
    whichever one is lying around passes for its author and fails in CI,
    which is exactly what happened on 22 Sep (two tests, green locally, red
    on the first push).

    Tests that are about the index build their own and pass `index_path`;
    this only moves the default out of reach. Session-scoped on purpose: a
    function-scoped patch is applied *after* a module-scoped fixture has
    already built its adapter, which is how the scenario's session kept
    reading the real index while every other test was insulated from it.
    """
    from lakehouse_agent.tools import lake_adapter

    missing = tmp_path_factory.mktemp("havuz") / "indeks-yok"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lake_adapter, "POOL_INDEX_PATH", missing)
        yield


@pytest.fixture
def agent_settings(monkeypatch, tmp_path) -> AgentSettings:
    """Settings with a throwaway key and a per-test data directory."""
    monkeypatch.setenv("MIA_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return AgentSettings()


@pytest.fixture
def fake_llm():
    from lakehouse_agent.testing.fake_llm import FakeLLM

    return FakeLLM()


@pytest.fixture
def fake_lake():
    from lakehouse_agent.testing.fake_lake import FakeLake

    return FakeLake()


@pytest.fixture
def scenario_wide(fake_lake) -> pd.DataFrame:
    """The scenario's four series, monthly, 2021-01 .. 2025-12: 60 rows."""
    from lakehouse_agent.testing.fake_lake import (
        CPI_SERIES,
        CREDIT_SERIES,
        HPI_SERIES,
        RATE_SERIES,
    )

    return fake_lake.wide(
        [CREDIT_SERIES, RATE_SERIES, CPI_SERIES, HPI_SERIES], "2021-01", "2025-12"
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def bddk_bronze(monkeypatch, tmp_path):
    """A working directory with the real BDDK table definitions and a writer
    for bronze table files in the shape `fetch_table_month` stores.

    `write(tablo_no, year, month, caption, columns, rows)`: `columns` are the
    value columns' names, `rows` are (label, style, values...) tuples.
    """
    import json
    import shutil

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ingest").mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "ingest" / "bddk_tables.yaml", tmp_path / "ingest")

    from lakehouse_agent.ingest import bddk_monthly

    def write(tablo_no, year, month, caption, columns, rows, fetched_at=None):
        payload = {
            "Json": {
                "colNames": ["", "", "", "BasitFont", *columns],
                "caption": caption,
                "uyari": "",
                "data": {
                    "rows": [
                        {"cell": ["Sektör", number, label, style, *values]}
                        for number, (label, style, *values) in enumerate(rows, start=1)
                    ]
                },
            }
        }
        record = {
            "url": bddk_monthly.BASE_URL,
            "params": {"tabloNo": tablo_no, "yil": year, "ay": month},
            "http_status": 200,
            "fetched_at": fetched_at or f"2026-09-22T00:00:{month:02d}+00:00",
            "payload": payload,
        }
        path = bddk_monthly.table_path(tablo_no, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    return write
