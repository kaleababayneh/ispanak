"""Contract tests for lake/gold.py's get()/wide(), run against the REAL
gold data on disk (data/gold/*.parquet), not a synthetic fixture --
skipped entirely when that data doesn't exist.

B's own test_lake_contract.py (not on this branch yet) only exercises a
FakeLake stand-in. A fake can't catch a real signature/behavior mismatch
against the actual DuckDB-backed implementation -- measured: FakeLake calls
`wide(series_ids, start, end, freq="M")`; the real `wide()` had no `freq`
parameter at all, so swapping the fake for the real gold.py raised
TypeError the moment they were wired together. Today's ~23 review findings
mostly came from exactly this gap: fixture-only tests that never exercised
the real accumulated data/signature. These tests call the real
`gold.get`/`gold.wide` directly, against whatever `data/gold/` the last
`build_gold()` run actually produced.
"""

import pandas as pd
import pytest

from lakehouse_agent.lake import gold

pytestmark = pytest.mark.skipif(
    not (gold.OBSERVATIONS_PATH.exists() and gold.SERIES_CATALOG_PATH.exists()),
    reason="gerçek gold verisi yok (data/gold/*.parquet) -- önce build_gold çalıştırılmalı",
)


def _real_catalog() -> pd.DataFrame:
    return pd.read_parquet(gold.SERIES_CATALOG_PATH)


def test_get_returns_exact_contract_columns_on_real_data():
    series_id = _real_catalog()["series_id"].iloc[0]

    result = gold.get([series_id])

    assert list(result.columns) == gold.OBSERVATIONS_COLUMNS
    if not result.empty:
        assert pd.api.types.is_datetime64_any_dtype(result["period"])


def test_wide_accepts_freq_kwarg_like_fakelake_does():
    """B'nin FakeLake'i `wide(series_ids, start, end, freq="M")` çağırıyor --
    gerçek wide() bu imzayı kabul etmezse TypeError patlar. FakeLake yerine
    doğrudan gerçek gold.wide()'ı çağırarak bu boşluğu kapatıyor."""
    series_ids = list(_real_catalog()["series_id"].unique()[:2])

    result = gold.wide(series_ids, None, None, freq="M")

    assert "period" in result.columns
    if not result.empty:
        assert pd.api.types.is_datetime64_any_dtype(result["period"])
    for series_id in series_ids:
        assert series_id in result.columns


def test_wide_default_freq_aligns_mixed_frequency_series_on_real_data():
    """Sözleşme: farklı frekanslı seriler istendiğinde wide() varsayılan
    olarak hizalamalı, sessizce neredeyse boş bir tablo dönmemeli (B'nin
    ölçtüğü: 311 satır, hiçbir satırda iki seri birlikte dolu değil)."""
    catalog = _real_catalog()
    freqs = catalog.groupby("freq")["series_id"].first()
    if len(freqs) < 2:
        pytest.skip("gerçek katalogda birden fazla frekans yok")

    series_ids = list(freqs.iloc[:2])

    result = gold.wide(series_ids)

    value_cols = [c for c in result.columns if c != "period"]
    assert len(value_cols) == len(series_ids)
    if not result.empty:
        assert int(result[value_cols].notna().all(axis=1).sum()) > 0
