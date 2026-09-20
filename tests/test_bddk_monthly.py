"""Tests for ingest/bddk_monthly.py. No network: `_post_report` is mocked."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lakehouse_agent.ingest import bddk_monthly

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bddk" / "basit_rapor_sample.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_find_row_returns_matching_cell():
    payload = _load_fixture()
    cell = bddk_monthly.find_row(payload, "Tüketici Kredileri - Konut")
    assert cell == ["Sektör", 2, "Tüketici Kredileri - Konut", "", 594545, 62, 594607]


def test_find_row_raises_when_label_missing():
    payload = _load_fixture()
    with pytest.raises(bddk_monthly.BddkRowMismatchError):
        bddk_monthly.find_row(payload, "Olmayan Satır")


def test_load_row_definitions_parses_real_yaml():
    row_defs = bddk_monthly.load_row_definitions()
    assert len(row_defs) == 3
    by_id = {rd.series_id: rd for rd in row_defs}
    assert "bddk:aylik:tuketici:konut:TL:sektor" in by_id
    konut = by_id["bddk:aylik:tuketici:konut:TL:sektor"]
    assert konut.tablo_no == 4
    assert konut.satir_etiketi == "Tüketici Kredileri - Konut"
    assert konut.cell_index == 4
    assert konut.kind == "stock"
    assert konut.unit == "milyon TL"
    assert konut.cum_type == "none"


@pytest.fixture
def row_defs():
    return [
        bddk_monthly.BddkRow(
            tablo_no=4,
            satir_etiketi="Tüketici Kredileri - Konut",
            series_id="bddk:aylik:tuketici:konut:TL:sektor",
            cell_index=4,
            kind="stock",
            unit="milyon TL",
            cum_type="none",
        )
    ]


def test_fetch_month_skips_http_when_output_exists(tmp_path, monkeypatch, row_defs):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_post_report(tablo_no, year, month):
        calls.append((tablo_no, year, month))
        return _load_fixture()

    monkeypatch.setattr(bddk_monthly, "_post_report", fake_post_report)

    out_path = bddk_monthly._month_path(2024, 1)
    out_path.parent.mkdir(parents=True)
    pd.DataFrame(columns=bddk_monthly.BRONZE_MONTH_COLUMNS).to_parquet(
        out_path, index=False
    )

    result = bddk_monthly.fetch_month(2024, 1, row_defs)
    assert result.outcome == "cached"
    assert calls == []


def test_fetch_month_writes_expected_cell_json(tmp_path, monkeypatch, row_defs):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        bddk_monthly, "_post_report", lambda tablo_no, year, month: _load_fixture()
    )

    result = bddk_monthly.fetch_month(2024, 3, row_defs)
    assert result.outcome == "ok"

    df = pd.read_parquet(result.path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["yil"] == 2024
    assert row["ay"] == 3
    assert row["series_id"] == "bddk:aylik:tuketici:konut:TL:sektor"
    assert json.loads(row["cell_json"]) == [
        "Sektör",
        2,
        "Tüketici Kredileri - Konut",
        "",
        594545,
        62,
        594607,
    ]

    raw = json.loads(bddk_monthly._raw_path(2024, 3).read_text(encoding="utf-8"))
    assert "4" in raw


def test_fetch_month_raises_on_row_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        bddk_monthly, "_post_report", lambda tablo_no, year, month: _load_fixture()
    )

    bad_row_defs = [
        bddk_monthly.BddkRow(
            tablo_no=4,
            satir_etiketi="Bu Satır Yok",
            series_id="bddk:aylik:hayali",
            cell_index=4,
            kind="stock",
            unit="milyon TL",
            cum_type="none",
        )
    ]
    with pytest.raises(bddk_monthly.BddkRowMismatchError):
        bddk_monthly.fetch_month(2024, 5, bad_row_defs)

    assert not bddk_monthly._month_path(2024, 5).exists()
