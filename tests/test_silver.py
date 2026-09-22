"""Tests for lakehouse_agent.lake.silver. No network access -- small,
hand-built bronze DataFrames with manually computed expected long output."""

import math
from datetime import date

import pandas as pd
import pytest

from lakehouse_agent.ingest import evds
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
    assert list(long["freq"]) == ["M", "M"]
    assert list(long["source"]) == ["EVDS", "EVDS"]


def test_melt_bronze_datagroup_weekly_rounds_down_to_monday():
    # Friday-anchored HAFTALIK(CUMA): 05-01-2024 is a Friday -> that week's Monday is 01-01-2024.
    bronze = pd.DataFrame({"Tarih": ["05-01-2024"], "TP_KTF12": ["42.5"]})

    long = silver.melt_bronze_datagroup(bronze, ["TP.KTF12"], "HAFTALIK(CUMA)", "%")

    assert long["period"].iloc[0] == date(2024, 1, 1)
    assert long["freq"].iloc[0] == "W"


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

    result, dropped = silver._dedupe_latest_fetch(long)

    assert len(result) == 1
    assert result["value"].iloc[0] == 999.0
    assert dropped == 1


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

    result, dropped = silver.build_silver_observations(["evds:TP.KFE.TR"])

    assert list(result["series_id"]) == ["evds:TP.KFE.TR", "evds:TP.KFE.TR"]
    assert list(result["period"]) == [date(2024, 1, 1), date(2024, 2, 1)]
    assert list(result["value"]) == [150.0, 151.5]
    assert dropped == 0


def test_build_silver_observations_no_matching_catalog_rows_returns_empty(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    evds.CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=["SERIE_CODE", "DATAGROUP_CODE", "FREQUENCY_STR", "BIRIMI"]
    ).to_parquet(evds.CATALOG_PATH, index=False)

    result, dropped = silver.build_silver_observations(["evds:TP.NOT.IN.CATALOG"])

    assert result.empty
    assert list(result.columns) == silver.OBSERVATION_COLUMNS
    assert dropped == 0


def test_to_float_bddk_converts_turkish_format_but_not_native_numbers():
    assert silver._to_float_bddk(594545) == 594545.0
    assert silver._to_float_bddk("1.234,5") == 1234.5
    assert math.isnan(silver._to_float_bddk("-"))
    assert math.isnan(silver._to_float_bddk(None))


def test_to_float_unrecognized_token_becomes_nan_not_a_crash():
    assert math.isnan(silver._to_float("N.A."))
    assert math.isnan(silver._to_float("—"))


def test_to_float_bddk_unrecognized_token_becomes_nan_not_a_crash():
    assert math.isnan(silver._to_float_bddk("null"))
    assert math.isnan(silver._to_float_bddk("—"))


def test_map_freq_covers_all_contract_frequencies():
    assert silver.map_freq("AYLIK") == "M"
    assert silver.map_freq("HAFTALIK(CUMA)") == "W"
    assert silver.map_freq("HAFTALIK(ÇARŞAMBA)") == "W"
    assert silver.map_freq("GÜNLÜK") == "D"
    assert silver.map_freq("İŞ GÜNÜ") == "D"
    assert silver.map_freq("ÜÇ AYLIK") == "Q"
    assert silver.map_freq("YILLIK") == "A"
    with pytest.raises(ValueError):
        silver.map_freq("BILINMEYEN")
