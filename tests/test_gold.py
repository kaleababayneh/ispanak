"""Tests for lakehouse_agent.lake.gold. No network access -- builds tiny
bronze/catalog fixtures on disk under tmp_path and exercises the real
DuckDB engine (no client mocked) since the security properties (read-only,
enable_external_access=false) only mean something against the real engine.
"""

import json
import logging
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
    assert row["freq"] == "M"
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
    assert set(observations["source"]) == {"EVDS", "BDDK-Aylik"}

    catalog = pd.read_parquet(catalog_path)
    assert set(catalog["series_id"]) == {
        "evds:TP.KFE.TR",
        "bddk:aylik:tuketici:konut:TL:sektor",
    }


def test_build_gold_freq_values_match_contracts_freq_literal(monkeypatch, tmp_path):
    """B'nin Pydantic Freq Literal'i D|W|M|Q|A (contracts.py) -- build_gold()
    çıktısı hiçbir zaman ham EVDS/BDDK dizgesi ("AYLIK" gibi) taşımamalı."""
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    _write_bddk_fixture(tmp_path)

    obs_path, catalog_path = gold.build_gold(
        ["evds:TP.KFE.TR", "bddk:aylik:tuketici:konut:TL:sektor"]
    )

    from typing import get_args

    from lakehouse_agent.contracts import Freq

    allowed = set(get_args(Freq))
    observations = pd.read_parquet(obs_path)
    catalog = pd.read_parquet(catalog_path)

    assert set(observations["freq"]).issubset(allowed)
    assert set(catalog["freq"]).issubset(allowed)


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


def test_build_gold_raises_on_schema_mismatch_instead_of_silent_nan_column(
    monkeypatch, tmp_path
):
    """Regression: reindex(columns=...) silently turns a renamed/missing
    source column into an all-NaN column instead of raising. If silver ever
    wrote "frekans" instead of "freq", the old code produced an all-NaN
    freq column with no exception at all."""
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)

    from lakehouse_agent.lake import silver as silver_module

    def broken_build_silver_observations(series_ids, *, source="evds"):
        real, dropped = silver_module.build_silver_observations(
            series_ids, source=source
        )
        return real.rename(columns={"freq": "frekans"}), dropped

    monkeypatch.setattr(
        gold, "build_silver_observations", broken_build_silver_observations
    )

    with pytest.raises(ValueError, match="freq"):
        gold.build_gold(["evds:TP.KFE.TR"])


def test_build_gold_writes_dedup_stats_with_real_dropped_count(monkeypatch, tmp_path):
    """Regression: silver's own dedup silently absorbed duplicate
    (series_id, period) rows before gold ever saw them, so quality.py's
    gold-side duplicate count was structurally always 0. build_gold now
    persists the real dropped count to a sidecar file."""
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    # Same (series_id, period) fetched twice (two overlapping year-chunk
    # files), different fetched_at -- exactly what _dedupe_latest_fetch
    # exists for.
    pd.DataFrame(
        {
            "Tarih": ["2024-1"],
            "TP_KFE_TR": ["150.5"],
            "_fetched_at": ["2026-09-20T00:00:00+00:00"],
        }
    ).to_parquet(
        evds.DATAGROUPS_DATA_ROOT / "bie_kfe" / "2024_rerun.parquet", index=False
    )

    gold.build_gold(["evds:TP.KFE.TR"])

    stats_path = gold.GOLD_ROOT / "dedup_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["dropped_duplicates"] == 1


def test_get_accepts_year_month_bounds_and_keeps_whole_end_month(monkeypatch, tmp_path):
    """B'nin ölçtüğü çökme: gold.get(ids, "2021-01", "2025-12") ->
    ConversionException: invalid date field format (DuckDB "YYYY-MM"'i DATE
    olarak parse edemiyor -- docs/contracts.md'nin imzası ve FakeLake bunu
    kullanıyor). Ayrıca end'i naif olarak ayın 1'ine normalize etmek o ayın
    geri kalanını sessizce düşürür -- end ayın SON gününe, start ayın ilk
    gününe normalize edilmeli."""
    monkeypatch.chdir(tmp_path)
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SERIE_CODE": "TP.KTF12",
                "DATAGROUP_CODE": "bie_ktf",
                "SERIE_NAME": "Konut Kredisi",
                "SERIE_NAME_ENG": "Housing Loan",
                "FREQUENCY_STR": "HAFTALIK(CUMA)",
                "BIRIMI": "%",
                "START_DATE": "01-01-2024",
                "END_DATE": "01-07-2026",
                "TOPIC_TITLE_TR": None,
                "TAG": None,
                "METADATA_LINK": None,
                "NOTE": None,
            }
        ]
    ).to_parquet(evds.CATALOG_PATH, index=False)

    year_dir = evds.DATAGROUPS_DATA_ROOT / "bie_ktf"
    year_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Tarih": [
                "29-11-2024",  # -> Monday 2024-11-25, before December
                "06-12-2024",  # -> Monday 2024-12-02
                "13-12-2024",  # -> Monday 2024-12-09
                "20-12-2024",  # -> Monday 2024-12-16
                "27-12-2024",  # -> Monday 2024-12-23
            ],
            "TP_KTF12": ["1", "2", "3", "4", "5"],
        }
    ).to_parquet(year_dir / "2024.parquet", index=False)

    gold.build_gold(["evds:TP.KTF12"])

    result = gold.get(["evds:TP.KTF12"], start="2024-12", end="2024-12")

    assert len(result) == 4
    assert result["period"].min().date() == date(2024, 12, 2)
    assert result["period"].max().date() == date(2024, 12, 23)


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


def test_wide_warns_on_mismatched_frequencies_and_reports_fill_counts(
    monkeypatch, tmp_path, caplog
):
    """B'nin ölçtüğü: wide(['evds:TP.KTF12','evds:TP.TUKFIY2025.GENEL'],
    '2021-01','2025-12') -> 311 satır, hiçbirinde iki seri birlikte dolu
    değil (biri haftalık, biri aylık) -- sessizce kullanılamaz bir tablo
    dönüyor."""
    monkeypatch.chdir(tmp_path)
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SERIE_CODE": "TP.KTF12",
                "DATAGROUP_CODE": "bie_ktf",
                "SERIE_NAME": "Konut Kredisi",
                "SERIE_NAME_ENG": "Housing Loan",
                "FREQUENCY_STR": "HAFTALIK(CUMA)",
                "BIRIMI": "%",
                "START_DATE": "01-01-2024",
                "END_DATE": "01-07-2026",
                "TOPIC_TITLE_TR": None,
                "TAG": None,
                "METADATA_LINK": None,
                "NOTE": None,
            },
            {
                "SERIE_CODE": "TP.TUKFIY2025.GENEL",
                "DATAGROUP_CODE": "bie_tukfiy",
                "SERIE_NAME": "Genel Endeks",
                "SERIE_NAME_ENG": "General Index",
                "FREQUENCY_STR": "AYLIK",
                "BIRIMI": "Endeks",
                "START_DATE": "01-01-2024",
                "END_DATE": "01-07-2026",
                "TOPIC_TITLE_TR": None,
                "TAG": None,
                "METADATA_LINK": None,
                "NOTE": None,
            },
        ]
    ).to_parquet(evds.CATALOG_PATH, index=False)

    ktf_dir = evds.DATAGROUPS_DATA_ROOT / "bie_ktf"
    ktf_dir.mkdir(parents=True)
    pd.DataFrame(
        {"Tarih": ["05-01-2024", "12-01-2024"], "TP_KTF12": ["1", "2"]}
    ).to_parquet(ktf_dir / "2024.parquet", index=False)

    tuk_dir = evds.DATAGROUPS_DATA_ROOT / "bie_tukfiy"
    tuk_dir.mkdir(parents=True)
    pd.DataFrame(
        {"Tarih": ["2024-1", "2024-2"], "TP_TUKFIY2025_GENEL": ["100", "101"]}
    ).to_parquet(tuk_dir / "2024.parquet", index=False)

    gold.build_gold(["evds:TP.KTF12", "evds:TP.TUKFIY2025.GENEL"])

    with caplog.at_level(logging.WARNING, logger="lakehouse_agent.lake.gold"):
        result = gold.wide(["evds:TP.KTF12", "evds:TP.TUKFIY2025.GENEL"])

    assert any("farklı frekanslarda" in r.message for r in caplog.records)
    assert not result.empty


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


def test_run_sql_limit_is_not_bypassed_by_a_literal_containing_the_word_limit(
    monkeypatch, tmp_path
):
    """B'nin ölçtüğü: "WHERE unit <> 'limitless'" -- sorguda "limit" alt
    dizgesi geçtiği için eski kod LIMIT eklemiyordu, sınırsız satır döndü."""
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    result = gold.run_sql("SELECT *, 'limitless' AS note FROM observations", limit=1)

    assert len(result) == 1


def test_run_sql_allows_cte_with_clause(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)
    gold.build_gold(["evds:TP.KFE.TR"])

    result = gold.run_sql(
        "WITH recent AS (SELECT * FROM observations) SELECT * FROM recent", limit=1
    )

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


def test_reading_parquet_directly_yields_different_period_dtype_than_get(
    monkeypatch, tmp_path
):
    """Documents the exact hazard gold.py's module docstring now warns
    about: reading observations.parquet directly gives a different `period`
    dtype than gold.get() -- merging the two on "period" would silently
    match zero rows. Do not "fix" this by changing the parquet dtype (B's
    agent layer normalizes it); this test exists so the docstring's claim
    doesn't silently go stale."""
    monkeypatch.chdir(tmp_path)
    _write_bronze_catalog_and_year(tmp_path)

    obs_path, _ = gold.build_gold(["evds:TP.KFE.TR"])

    from_parquet = pd.read_parquet(obs_path)
    from_get = gold.get(["evds:TP.KFE.TR"])

    assert not pd.api.types.is_datetime64_any_dtype(from_parquet["period"])
    assert pd.api.types.is_datetime64_any_dtype(from_get["period"])
