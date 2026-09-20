"""Tests for lake/catalog.py. No network: embedding is mocked/synthetic."""

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.lake import catalog
from lakehouse_agent.lake.catalog import (
    QUERY_INSTRUCTION_PREFIX,
    _exact_code_match,
    build_catalog_index,
    build_search_text,
    embed_with_cache,
    filter_catalog,
    search_catalog,
)

# The 6 scenario series (from gold.py SCENARIO_SERIES) that must survive filtering.
SCENARIO_CODES = [
    "TP.KTF12",
    "TP.KFE.TR",
    "TP.TUKFIY2025.GENEL",
    "TP.AKONUTSAT1.KTRTOPLAM",
    "TP.AKONUTSAT2.KTRTOPLAM",
    "TP.HPBITABLO6.3",
]


def _row(
    serie_code,
    serie_name="Bir seri",
    datagroup_code="bie_test",
    datagroup_name="Test Grubu",
    end_date="01-07-2026",
    freq="AYLIK",
    topic="KONU",
    tag="etiket",
    birimi="TL",
    datasource="kaynak",
):
    return {
        "SERIE_CODE": serie_code,
        "SERIE_NAME": serie_name,
        "DATAGROUP_CODE": datagroup_code,
        "DATAGROUP_NAME": datagroup_name,
        "END_DATE": end_date,
        "FREQUENCY_STR": freq,
        "TOPIC_TITLE_TR": topic,
        "TAG": tag,
        "BIRIMI": birimi,
        "DATASOURCE": datasource,
    }


SCENARIO_NAMES = {
    "TP.KTF12": "Konut Kredisi (TL, Akım, %)",
    "TP.KFE.TR": "Konut Fiyat Endeksi (KFE)",
    "TP.TUKFIY2025.GENEL": "Genel Endeks",
    "TP.AKONUTSAT1.KTRTOPLAM": "Türkiye_Konut_Toplam Satışlar",
    "TP.AKONUTSAT2.KTRTOPLAM": "Türkiye_Konut_İpotekli Satışlar",
    "TP.HPBITABLO6.3": "1.1.1.Konut (TL + YP)",
}


def _make_catalog_df() -> pd.DataFrame:
    rows = [
        _row(
            code,
            serie_name=SCENARIO_NAMES[code],
            datagroup_code="bie_akonutsat4" if "AKONUTSAT" in code else "bie_test",
        )
        for code in SCENARIO_CODES
    ]
    # Noise rows that must be filtered out.
    rows += [
        _row("TP.ARSIVSERI", serie_name="Eski Arşiv Serisi"),
        _row("TP.OLUSERI", end_date="01-01-2020"),  # dead: END_DATE < 2025-06
        _row("TP.YILLIKSERI", freq="YILLIK"),
        _row("TP.IMFSERI", datagroup_code="bie_imfcpind"),
        _row(
            "TP.AKONUTSAT4.KTR510",  # province breakdown, no TOPLAM
            datagroup_code="bie_akonutsat4",
            serie_name="Ankara_Konut_İkinci El Satışlar",
        ),
    ]
    return pd.DataFrame(rows)


def test_filter_catalog_keeps_scenario_series():
    df = _make_catalog_df()
    filtered = filter_catalog(df)
    kept_codes = set(filtered["SERIE_CODE"])
    for code in SCENARIO_CODES:
        assert code in kept_codes, f"{code} filtre sonrası kayboldu"
    assert "TP.ARSIVSERI" not in kept_codes
    assert "TP.OLUSERI" not in kept_codes
    assert "TP.YILLIKSERI" not in kept_codes
    assert "TP.IMFSERI" not in kept_codes
    assert "TP.AKONUTSAT4.KTR510" not in kept_codes


def test_embed_with_cache_skips_previously_embedded_text(tmp_path):
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return np.stack([np.ones(8, dtype=np.float32) * i for i in range(len(texts))])

    texts = ["metin bir", "metin iki"]
    vectors1, failed1 = embed_with_cache(texts, cache_dir=tmp_path, embed_fn=fake_embed)
    assert failed1 == []
    assert len(calls) == 1

    vectors2, failed2 = embed_with_cache(texts, cache_dir=tmp_path, embed_fn=fake_embed)
    assert failed2 == []
    assert len(calls) == 1  # second call served entirely from cache
    np.testing.assert_array_equal(vectors1, vectors2)


def _fake_embed_deterministic(texts, is_query: bool = False):
    """Synthetic embedding: hash-based, deterministic, dim 8.

    Honors `is_query` the same way the real `_embed_batch` does (prepend
    the instruction prefix before hashing), so a query and a document with
    otherwise-identical text still land on different fake vectors -- the
    same asymmetry the real model has.
    """
    if is_query:
        texts = [QUERY_INSTRUCTION_PREFIX + t for t in texts]
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(rng.random(8, dtype=np.float64).astype(np.float32))
    return np.stack(out)


@pytest.fixture
def catalog_table(tmp_path):
    df = _make_catalog_df()
    return build_catalog_index(
        df,
        tmp_path / "catalog.lance",
        cache_dir=tmp_path / "emb_cache",
        embed_fn=_fake_embed_deterministic,
        with_vectors=True,
    )


def test_search_catalog_exact_code_match_ranked_first(catalog_table):
    results = search_catalog(
        "TP.KTF12", catalog_table, top_k=5, embed_fn=_fake_embed_deterministic
    )
    assert results.iloc[0]["SERIE_CODE"] == "TP.KTF12"


def test_search_catalog_fallback_without_vector(tmp_path):
    df = filter_catalog(_make_catalog_df())
    df = df.copy()
    df["text"] = df.apply(build_search_text, axis=1)

    import lancedb
    from lancedb.index import FTS

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("catalog_no_vector", df, mode="overwrite")
    table.create_index("text", config=FTS())

    results = search_catalog("konut", table, top_k=5, use_vector=False)
    assert not results.empty
    assert "vector" not in results.columns


class _FakeMiaLLM:
    """Spies on the texts actually sent to MiaLLM.embed(), no real call."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts, *, tag=None):
        self.calls.append(list(texts))
        return [[0.0] * 8 for _ in texts]


def test_embed_batch_prepends_instruction_prefix_only_for_queries(monkeypatch):
    fake_llm = _FakeMiaLLM()
    monkeypatch.setattr(catalog, "_mia_llm", fake_llm)

    catalog._embed_batch(["konut kredisi"], is_query=False)
    assert fake_llm.calls[-1] == ["konut kredisi"]

    catalog._embed_batch(["konut kredisi"], is_query=True)
    assert fake_llm.calls[-1] == [QUERY_INSTRUCTION_PREFIX + "konut kredisi"]


def test_search_catalog_embeds_query_with_instruction_prefix(catalog_table):
    """Regression: search_catalog must ask for the QUERY-side embedding
    (is_query=True), never the document-side one -- Qwen3-Embedding is
    asymmetric (see catalog.QUERY_INSTRUCTION_PREFIX)."""
    calls = []

    def spy_embed(texts, is_query=False):
        calls.append((list(texts), is_query))
        return _fake_embed_deterministic(texts, is_query=is_query)

    search_catalog(
        "konut kredisi faiz oranı", catalog_table, top_k=5, embed_fn=spy_embed
    )

    assert calls, "embed_fn hiç çağrılmadı"
    texts, is_query = calls[0]
    assert texts == ["konut kredisi faiz oranı"]
    assert is_query is True


def test_exact_code_match_finds_row_independent_of_ranking(tmp_path):
    """Regression: the old code only looked for an exact SERIE_CODE match
    inside the already-fetched FTS/vector top-k hit lists. Since
    build_search_text() never puts SERIE_CODE into the indexed `text`
    field, a code query like "TP.KTF12" has no reason to rank inside any
    top-k window on a large table -- that's exactly why the pinning never
    fired on the real 19646-row index. `_exact_code_match` must find the
    row via a direct table-wide filter, regardless of top-k or ranking.
    """
    decoys = [_row(f"TP.DECOY{i}", serie_name=f"Alakasız Seri {i}") for i in range(30)]
    target = _row("TP.KTF12", serie_name=SCENARIO_NAMES["TP.KTF12"])
    df = pd.DataFrame([*decoys, target])
    df = df.copy()
    df["text"] = df.apply(build_search_text, axis=1)

    import lancedb

    db = lancedb.connect(str(tmp_path))
    table = db.create_table("catalog_many_decoys", df, mode="overwrite")

    result = _exact_code_match(table, "tp.ktf12")  # case-insensitive too

    assert len(result) == 1
    assert result.iloc[0]["SERIE_CODE"] == "TP.KTF12"


def test_search_catalog_exact_match_wins_even_with_tiny_top_k(tmp_path):
    decoys = [_row(f"TP.DECOY{i}", serie_name=f"Alakasız Seri {i}") for i in range(30)]
    target = _row("TP.KTF12", serie_name=SCENARIO_NAMES["TP.KTF12"])
    df = pd.DataFrame([*decoys, target])

    table = build_catalog_index(
        df,
        tmp_path / "catalog.lance",
        cache_dir=tmp_path / "emb_cache",
        embed_fn=_fake_embed_deterministic,
        with_vectors=True,
    )

    results = search_catalog(
        "TP.KTF12", table, top_k=1, embed_fn=_fake_embed_deterministic
    )

    assert len(results) == 1
    assert results.iloc[0]["SERIE_CODE"] == "TP.KTF12"
