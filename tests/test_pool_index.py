"""The pool index's search text and ranking rules (lake/pool_index.py).

No network and no embeddings: the ranking works on (series_id, score) lists,
and the one index test stands its vectors in with two dimensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.lake import gold, pool_index
from lakehouse_agent.tools.lake_adapter import LakeAdapter

# --- places -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Kahramanmaraş_Konut_Toplam Satışlar", {"kahramanmaras"}),
        ("TR10 (İstanbul)", {"istanbul"}),
        ("İZMİR - Nakdi Krediler", {"izmir"}),
        # Mauritius' ISO code is not Muş, the exchange is not the province.
        ("Morityus (MUS) Tüketici Fiyat Endeksi (2010=100)", set()),
        ("Borsa İstanbul AŞ'den Alınan Paylar", set()),
        ("Tüketici Kredileri - Konut (TP)", set()),
    ],
)
def test_a_name_says_which_province_it_is_about(name, expected):
    assert pool_index.places(name) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("İstanbul konut kredisi", frozenset({"istanbul"})),
        ("konut kredisi bakiyesi", frozenset()),
        ("illere göre konut kredisi", None),  # asks for the breakdown itself
        ("bölgesel konut fiyatları", None),
    ],
)
def test_a_query_names_places_or_asks_for_the_breakdown(query, expected):
    assert pool_index.query_places(query) == expected


# --- what the ranking reads -----------------------------------------------------


def _catalog(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **dict.fromkeys(gold.SERIES_CATALOG_COLUMNS),
                "series_id": sid,
                "name_tr": name,
            }
            for sid, name in rows
        ],
        columns=gold.SERIES_CATALOG_COLUMNS,
    )


_EVDS = pd.DataFrame(
    {
        "SERIE_CODE": ["TP.A.TR", "TP.A.K632", "TP.KFE.TR10", "TP.HKFE01"],
        "DATAGROUP_CODE": ["bie_a", "bie_a", "bie_kfe", "bie_hkfe"],
        "DATAGROUP_NAME": [
            "Konut Satış İstatistikleri",
            "Konut Satış İstatistikleri",
            "Konut Fiyat Endeksi",
            "Konut Fiyat Endeksi (KFE) (2017=100) (Arşiv)",
        ],
        "SEVIYE": [1, 2, 1, 1],
        "UST_SERIE_CODE": ["-1", "TP.A.TR", "-1", "-1"],
    }
)


def test_series_meta_reads_places_parents_and_the_archive_mark():
    catalog = _catalog(
        ("evds:TP.A.TR", "Türkiye_Konut_Toplam Satışlar"),
        ("evds:TP.A.K632", "Kahramanmaraş_Konut_Toplam Satışlar"),
        ("evds:TP.KFE.TR10", "TR10 (İstanbul)"),
        ("evds:TP.HKFE01", "Konut Fiyat Endeksi (KFE)"),
        ("bddk:finturk:krediler:izmir:nakdi_krediler:sektor", "İZMİR - Nakdi Krediler"),
    )

    meta = pool_index.series_meta(catalog, _EVDS).set_index("series_id")

    national = meta.loc["evds:TP.A.TR"]
    assert national["table"] == "Konut Satış İstatistikleri"
    assert national["places"] == "" and pd.isna(national["family"])
    assert meta.loc["evds:TP.A.K632", "family"] == "evds:TP.A.TR"  # its parent
    assert meta.loc["evds:TP.KFE.TR10", "family"] == "evds-group:bie_kfe"  # no parent
    assert bool(meta.loc["evds:TP.HKFE01", "archive"])  # marked on the datagroup
    finturk = meta.loc["bddk:finturk:krediler:izmir:nakdi_krediler:sektor"]
    assert finturk["places"] == "izmir"
    assert finturk["family"] == "finturk:krediler:nakdi_krediler"


def test_the_search_text_carries_the_datagroup_and_no_double_spaces():
    row = _catalog(("evds:TP.X", "(USD)  ABD  Doları")).iloc[0]
    row["topic"] = "TCMB DÖVİZ KURLARI"

    text = pool_index.search_text(row, table="Döviz Kurları")
    assert text.startswith("(USD) ABD Doları | Döviz Kurları | TCMB DÖVİZ KURLARI")
    # A table that only repeats the topic adds nothing.
    assert "Kurları | TCMB" not in pool_index.search_text(
        row, table="TCMB Döviz Kurları"
    )


# --- rank -----------------------------------------------------------------------


def _meta(*rows: tuple[str, str, bool, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"series_id": sid, "table": None, "places": p, "archive": a, "family": f}
            for sid, p, a, f in rows
        ]
    )


_PROVINCES = _meta(
    ("fin:balikesir", "balikesir", False, "fin:konut"),
    ("fin:istanbul", "istanbul", False, "fin:konut"),
    ("fin:bolu", "bolu", False, "fin:konut"),
    ("bddk:konut", "", False, None),
    ("evds:kfe_arsiv", "", True, None),
    ("evds:kfe", "", False, None),
)
_SCORED = [
    ("fin:balikesir", 0.62),
    ("fin:bolu", 0.61),
    ("evds:kfe_arsiv", 0.60),
    ("fin:istanbul", 0.59),
    ("bddk:konut", 0.58),
    ("evds:kfe", 0.55),
]


def test_a_query_without_a_place_sees_national_and_current_series_first():
    ranked, standing = pool_index.rank("konut kredisi", _SCORED, _PROVINCES)

    assert [sid for sid, _ in ranked] == [
        "bddk:konut",
        "evds:kfe",
        # Not asked for, each by similarity: one candidate for three provinces.
        "fin:balikesir",
        "evds:kfe_arsiv",
    ]
    assert dict(ranked)["bddk:konut"] == 0.58  # scores are not touched
    assert standing == {"fin:balikesir": 2}


def test_a_named_province_stands_for_its_family():
    ranked, standing = pool_index.rank("İstanbul konut kredisi", _SCORED, _PROVINCES)

    ids = [sid for sid, _ in ranked]
    assert ids.index("fin:istanbul") < ids.index("evds:kfe_arsiv")
    assert "fin:balikesir" not in ids and "fin:bolu" not in ids
    assert standing == {"fin:istanbul": 2}


def test_asking_for_the_archive_or_the_breakdown_puts_nothing_back():
    ranked, _ = pool_index.rank("illere göre arşiv konut", _SCORED, _PROVINCES)

    assert [sid for sid, _ in ranked][:2] == ["fin:balikesir", "evds:kfe_arsiv"]


def test_without_stored_meta_the_order_is_the_similarity_order():
    assert pool_index.rank("konut", _SCORED, None) == (_SCORED, {})


# --- the index keeps the meta; the adapter says what a candidate stands for --


_NATIONAL, _FAMILY = [1.0, 0.0], [0.9, 0.1]


def _documents(texts):
    vectors = [_FAMILY if " - " in text else _NATIONAL for text in texts]
    return np.array(vectors), []


def test_a_search_over_the_index_puts_the_national_series_first(tmp_path):
    catalog = _catalog(
        ("bddk:finturk:bireysel:balikesir:konut_kredisi:sektor", "BALIKESİR - Konut"),
        ("bddk:finturk:bireysel:bolu:konut_kredisi:sektor", "BOLU - Konut"),
        ("bddk:aylik:tuketici:konut:TL:sektor", "Tüketici Kredileri Konut"),
    )
    catalog_path = tmp_path / "series_catalog.parquet"
    catalog.to_parquet(catalog_path)
    index_dir = tmp_path / "index"
    pool_index.build(catalog, path=index_dir, embed_documents=_documents)

    opened = pool_index.PoolIndex.open(index_dir)
    assert opened is not None and opened.meta is not None
    assert set(opened.meta["family"].dropna()) == {"finturk:bireysel:konut_kredisi"}

    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_query=lambda _q: np.array([0.9, 0.1]),  # nearest to the provinces
        index_path=index_dir,
        in_memory_limit=0,
    )
    adapter._coverage = dict

    hits, notes = adapter.search_detailed("konut kredisi bakiyesi")

    assert hits[0].series_id == "bddk:aylik:tuketici:konut:TL:sektor"
    assert len(hits) == 2  # the two provinces are one candidate
    assert any("1 baska il/bolge" in note for note in notes)


def _grouped_meta() -> pd.DataFrame:
    """A pool where one datagroup's name is what the query spells out."""
    return pd.DataFrame(
        [
            {
                "series_id": "evds:kfe",
                "table": "Konut Fiyat Endeksi",
                "places": "",
                "archive": False,
                "family": None,
                "group": "bie_kfe",
                "head": True,
            },
            {
                "series_id": "evds:kfe_arsiv",
                "table": "Hedonik Konut Fiyat Endeksi",
                "places": "",
                "archive": True,
                "family": None,
                "group": "bie_hkfe",
                "head": True,
            },
            {
                "series_id": "evds:tufe_konut_sigortasi",
                "table": "Tüketici Fiyat Endeksi (2025=100)",
                "places": "",
                "archive": False,
                "family": None,
                "group": "bie_tukfiy2025",
                "head": False,
            },
        ]
    )


def test_a_table_the_query_names_is_led_by_its_own_total():
    """Naming a table used to demote its sub-items and nothing more, so the
    table's own total competed on cosine alone against every unrelated series.
    Measured on the real pool: "Konut Fiyat Endeksi (KFE)" scored 0.639 while
    four housing-related CPI rows scored 0.645-0.684, and the five candidates
    held none of them."""
    scored = [
        ("evds:tufe_konut_sigortasi", 0.684),
        ("evds:kfe_arsiv", 0.726),
        ("evds:kfe", 0.639),
    ]

    ordered, _standing = pool_index.rank("konut fiyat endeksi", scored, _grouped_meta())

    assert [sid for sid, _ in ordered][0] == "evds:kfe"
    # The archived head of an archived table is not promoted past it.
    assert [sid for sid, _ in ordered][-1] == "evds:kfe_arsiv"


def test_a_query_that_names_no_table_keeps_the_similarity_order():
    """The promotion must not disturb every other query."""
    scored = [
        ("evds:tufe_konut_sigortasi", 0.684),
        ("evds:kfe", 0.639),
    ]

    ordered, _standing = pool_index.rank("mevduat faizi", scored, _grouped_meta())

    assert [sid for sid, _ in ordered] == [s for s, _ in scored]
