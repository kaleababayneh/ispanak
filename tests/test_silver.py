"""Tests for lakehouse_agent.lake.silver. No network access -- small,
hand-built bronze DataFrames with manually computed expected long output."""

import json
import math
from datetime import date

import pandas as pd

from lakehouse_agent.ingest import bddk_monthly, evds
from lakehouse_agent.lake import silver


def test_melt_bronze_datagroup_monthly_basic():
    bronze = pd.DataFrame(
        {
            "Tarih": ["2024-1", "2024-2"],
            "TP_KFE_TR": ["150.0", "151.5"],
            "_fetched_at": ["2026-09-15T00:00:00+00:00", "2026-09-15T00:00:00+00:00"],
        }
    )

    long = silver.melt_bronze_datagroup(bronze, ["TP.KFE.TR"], "AYLIK", None)

    assert list(long["period"]) == [date(2024, 1, 1), date(2024, 2, 1)]
    assert list(long["value"]) == [150.0, 151.5]
    assert list(long["series_id"]) == ["evds:TP.KFE.TR", "evds:TP.KFE.TR"]
    assert list(long["freq"]) == ["AYLIK", "AYLIK"]


def test_melt_bronze_datagroup_weekly_rounds_down_to_monday():
    # Friday-anchored HAFTALIK(CUMA): 05-01-2024 is a Friday -> that week's Monday is 01-01-2024.
    bronze = pd.DataFrame({"Tarih": ["05-01-2024"], "TP_KTF12": ["42.5"]})

    long = silver.melt_bronze_datagroup(bronze, ["TP.KTF12"], "HAFTALIK(CUMA)", "%")

    assert long["period"].iloc[0] == date(2024, 1, 1)


def test_melt_bronze_datagroup_missing_tokens_become_nan_not_zero():
    bronze = pd.DataFrame(
        {"Tarih": ["2024-1", "2024-2", "2024-3"], "X": ["-", "n.a.", ""]}
    )

    long = silver.melt_bronze_datagroup(bronze, ["X"], "AYLIK", None)

    assert all(math.isnan(v) for v in long["value"])


def test_melt_bronze_datagroup_ignores_columns_outside_requested_serie_codes():
    bronze = pd.DataFrame(
        {"Tarih": ["2024-1"], "TP_KFE_TR": ["150.0"], "TP_OTHER_SERIES": ["999.0"]}
    )

    long = silver.melt_bronze_datagroup(bronze, ["TP.KFE.TR"], "AYLIK", None)

    assert set(long["series_id"]) == {"evds:TP.KFE.TR"}


def test_dedupe_latest_fetch_keeps_newest_fetched_at_row():
    long = pd.DataFrame(
        {
            "series_id": ["evds:TP.KFE.TR", "evds:TP.KFE.TR"],
            "period": [date(2024, 1, 1), date(2024, 1, 1)],
            "freq": ["AYLIK", "AYLIK"],
            "value": [150.0, 999.0],
            "unit": [None, None],
            "source": ["evds", "evds"],
            "fetched_at": ["2026-09-01T00:00:00+00:00", "2026-09-15T00:00:00+00:00"],
        }
    )

    result = silver._dedupe_latest_fetch(long)

    assert len(result) == 1
    assert result["value"].iloc[0] == 999.0


def test_build_silver_observations_end_to_end(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SERIE_CODE": "TP.KFE.TR",
                "DATAGROUP_CODE": "bie_kfe",
                "FREQUENCY_STR": "AYLIK",
                "BIRIMI": None,
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

    result = silver.build_silver_observations(["evds:TP.KFE.TR"])

    assert list(result["series_id"]) == ["evds:TP.KFE.TR", "evds:TP.KFE.TR"]
    assert list(result["period"]) == [date(2024, 1, 1), date(2024, 2, 1)]
    assert list(result["value"]) == [150.0, 151.5]


def test_build_silver_observations_no_matching_catalog_rows_returns_empty(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=["SERIE_CODE", "DATAGROUP_CODE", "FREQUENCY_STR", "BIRIMI"]
    ).to_parquet(evds.CATALOG_PATH, index=False)

    result = silver.build_silver_observations(["evds:TP.NOT.IN.CATALOG"])

    assert result.empty
    assert list(result.columns) == silver.OBSERVATION_COLUMNS


def test_to_float_bddk_converts_turkish_format_but_not_native_numbers():
    assert silver._to_float_bddk(594545) == 594545.0
    assert silver._to_float_bddk("1.234,5") == 1234.5
    assert math.isnan(silver._to_float_bddk("-"))
    assert math.isnan(silver._to_float_bddk(None))


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


def test_build_bddk_silver_observations_reads_value_by_cell_index(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    _write_bddk_fixture(tmp_path)

    result = silver.build_bddk_silver_observations(
        ["bddk:aylik:tuketici:konut:TL:sektor"]
    )

    assert list(result["series_id"]) == ["bddk:aylik:tuketici:konut:TL:sektor"]
    assert list(result["period"]) == [date(2024, 1, 1)]
    assert list(result["value"]) == [594545.0]
    assert list(result["unit"]) == ["milyon TL"]
    assert list(result["source"]) == [silver.BDDK_SOURCE]
