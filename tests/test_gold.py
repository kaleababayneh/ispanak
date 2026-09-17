"""Tests for lakehouse_agent.lake.gold. No network access -- builds tiny
bronze/catalog fixtures on disk under tmp_path and exercises the real
DuckDB engine (no client mocked) since the security properties (read-only,
enable_external_access=false) only mean something against the real engine.
"""

import json
from datetime import date

import duckdb
import pandas as pd
import pytest

from lakehouse_agent.ingest import bddk_monthly, evds
from lakehouse_agent.lake import gold


def _write_bronze_catalog_and_year(tmp_path):
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SERIE_CODE": "TP.KFE.TR",
                "DATAGROUP_CODE": "bie_kfe",
                "SERIE_NAME": "Konut Fiyat Endeksi (KFE)",
                "SERIE_NAME_ENG": "Housing Price Index",
                "FREQUENCY_STR": "AYLIK",
                "BIRIMI": None,
                "START_DATE": "01-01-2010",
                "END_DATE": "01-07-2026",
                "TOPIC_TITLE_TR": "Konut",
                "TAG": "konut,fiyat",
                "METADATA_LINK": "https://example/kfe",
                "NOTE": None,
            }
        ]
    ).to_parquet(evds.CATALOG_PATH, index=False)

    year_dir = evds.DATAGROUPS_DATA_ROOT / "bie_kfe"
    year_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Tarih": ["2024-1", "2024-2"],
            "TP_KFE_TR": ["150.0", "151.5"],
            "_fetched_at": ["2026-09-15T00:00:00+00:00", "2026-09-15T00:00:00+00:00"],
        }
    ).to_parquet(year_dir / "2024.parquet", index=False)


def test_build_series_catalog_maps_scenario_kind_and_agg_method(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)

    catalog = gold.build_series_catalog(["evds:TP.KFE.TR"])

    assert list(catalog.columns) == gold.SERIES_CATALOG_COLUMNS
    row = catalog.iloc[0]
    assert row["series_id"] == "evds:TP.KFE.TR"
    assert row["kind"] == "index"
    assert row["agg_method"] == "avg"
    assert row["cum_type"] == "none"
    assert row["name_tr"] == "Konut Fiyat Endeksi (KFE)"


def test_build_series_catalog_raises_for_series_without_scenario_mapping(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SERIE_CODE": "TP.UNMAPPED",
                "DATAGROUP_CODE": "bie_x",
                "FREQUENCY_STR": "AYLIK",
            }
        ]
    ).to_parquet(evds.CATALOG_PATH, index=False)

    with pytest.raises(KeyError):
        gold.build_series_catalog(["evds:TP.UNMAPPED"])


def _write_bddk_fixture(tmp_path):
    rows_yaml = tmp_path / "ingest" / "bddk_rows.yaml"
    rows_yaml.parent.mkdir(parents=True)
    rows_yaml.write_text(
        "- tablo_no: 4\n"
        '  satir_etiketi: "Tüketici Kredileri - Konut"\n'
        '  series_id: "bddk:aylik:tuketici:konut:TL:sektor"\n'
        "  cell_index: 4\n"
        "  kind: stock\n"
        '  unit: "milyon TL"\n'
        "  cum_type: none\n",
        encoding="utf-8",
    )

    bronze_dir = tmp_path / "data" / "bronze" / "bddk_aylik"
    bronze_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "yil": 2024,
                "ay": 1,
                "tablo_no": 4,
                "satir_etiketi": "Tüketici Kredileri - Konut",
                "series_id": "bddk:aylik:tuketici:konut:TL:sektor",
                "cell_json": json.dumps(
                    ["Sektör", 2, "Tüketici Kredileri - Konut", "", 594545, 62, 594607]
                ),
                "url": bddk_monthly.BASE_URL,
                "http_status": 200,
                "fetched_at": "2026-09-17T00:00:00+00:00",
            }
        ]
    ).to_parquet(bronze_dir / "2024-01.parquet", index=False)


def test_build_bddk_series_catalog_maps_yaml_fields(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bddk_fixture(tmp_path)

    catalog = gold.build_bddk_series_catalog(["bddk:aylik:tuketici:konut:TL:sektor"])

    assert list(catalog.columns) == gold.SERIES_CATALOG_COLUMNS
    row = catalog.iloc[0]
    assert row["name_tr"] == "Tüketici Kredileri - Konut"
    assert row["source"] == "BDDK-Aylik"
    assert row["freq"] == "AYLIK"
    assert row["unit"] == "milyon TL"
    assert row["kind"] == "stock"
    assert row["agg_method"] == "last"
    assert row["cum_type"] == "none"
    assert row["metadata_url"] == "https://www.bddk.org.tr/BultenAylik/"


def test_build_bddk_series_catalog_raises_for_series_without_yaml_mapping(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    _write_bddk_fixture(tmp_path)

    with pytest.raises(KeyError):
        gold.build_bddk_series_catalog(["bddk:aylik:hayali"])


def test_build_gold_mixes_evds_and_bddk_sources(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    _write_bddk_fixture(tmp_path)

    obs_path, catalog_path = gold.build_gold(
        ["evds:TP.KFE.TR", "bddk:aylik:tuketici:konut:TL:sektor"]
    )

    observations = pd.read_parquet(obs_path)
    assert set(observations["series_id"]) == {
        "evds:TP.KFE.TR",
        "bddk:aylik:tuketici:konut:TL:sektor",
    }
    assert set(observations["source"]) == {"evds", "BDDK-Aylik"}

    catalog = pd.read_parquet(catalog_path)
    assert set(catalog["series_id"]) == {
        "evds:TP.KFE.TR",
        "bddk:aylik:tuketici:konut:TL:sektor",
    }


def test_build_gold_writes_both_parquet_files_matching_contract_columns(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)

    obs_path, catalog_path = gold.build_gold(["evds:TP.KFE.TR"])

    observations = pd.read_parquet(obs_path)
    assert list(observations.columns) == gold.OBSERVATIONS_COLUMNS
    assert list(observations["period"]) == [date(2024, 1, 1), date(2024, 2, 1)]

    catalog = pd.read_parquet(catalog_path)
    assert list(catalog.columns) == gold.SERIES_CATALOG_COLUMNS


def test_get_filters_by_series_id_and_period_range(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    result = gold.get(["evds:TP.KFE.TR"], start="2024-02-01")

    assert len(result) == 1
    assert result["value"].iloc[0] == 151.5


def test_wide_pivots_one_column_per_series(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    result = gold.wide(["evds:TP.KFE.TR"])

    assert "evds:TP.KFE.TR" in result.columns
    assert list(result["evds:TP.KFE.TR"]) == [150.0, 151.5]


def test_run_sql_rejects_non_select(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    with pytest.raises(ValueError):
        gold.run_sql("DELETE FROM observations")


def test_run_sql_rejects_statement_chaining(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    with pytest.raises(ValueError):
        gold.run_sql("SELECT * FROM observations; DROP TABLE observations")


def test_run_sql_adds_missing_limit_and_returns_rows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    result = gold.run_sql("SELECT * FROM observations", limit=1)

    assert len(result) == 1


def test_connect_blocks_local_file_read_via_read_csv(monkeypatch, tmp_path):
    """The engine-level control CLAUDE.md calls out: even though run_sql
    only accepts SELECT, a SELECT alone can call read_csv() -- that must be
    blocked by enable_external_access=false, not by a keyword blacklist."""
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    secret = tmp_path / "secret.csv"
    secret.write_text("a,b\n1,2\n", encoding="utf-8")

    con = gold.connect()
    try:
        with pytest.raises(duckdb.Error):
            con.execute(f"SELECT * FROM read_csv('{secret.as_posix()}')")
    finally:
        con.close()


def test_connect_is_read_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    con = gold.connect()
    try:
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE t (x INT)")
    finally:
        con.close()
