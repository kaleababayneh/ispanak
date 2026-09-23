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


def test_nearest_scores_the_hits_it_returns(tmp_path):
    """The similarity has to be a real measurement, not a constant.

    `nearest` reads a column it does not select by name unless asked to, and
    LanceDB warns it will stop supplying that column on its own. Every caller
    ranks on this number, so the failure would be either a `KeyError` at
    search time or -- worse -- a flat score that silently turns the ranking
    into insertion order. Asserting the gap catches both.
    """
    catalog = _catalog(
        ("bddk:aylik:tuketici:konut:TL:sektor", "Tüketici Kredileri Konut"),
        ("bddk:finturk:bireysel:bolu:konut_kredisi:sektor", "BOLU - Konut"),
    )
    index_dir = tmp_path / "index"
    pool_index.build(catalog, path=index_dir, embed_documents=_documents)

    opened = pool_index.PoolIndex.open(index_dir)
    assert opened is not None

    # _NATIONAL exactly: the national series sits on the query, the province
    # family a little off it.
    scored = opened.nearest(np.array(_NATIONAL, dtype=np.float32), k=2)

    assert [series_id for series_id, _ in scored] == [
        "bddk:aylik:tuketici:konut:TL:sektor",
        "bddk:finturk:bireysel:bolu:konut_kredisi:sektor",
    ]
    assert scored[0][1] == pytest.approx(1.0, abs=1e-6)
    assert scored[0][1] > scored[1][1]


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


# --- rank: another country's copy, and an older base year -----------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Almanya (DEU) Tüketici Fiyat Endeksi (2010=100)", ("almanya", "DEU")),
        ("Güney Kore (KOR) GSYH, Nominal", ("guney kore", "KOR")),
        # Ours is not abroad, though it sits in the same table.
        ("Türkiye (TUR) Tüketici Fiyat Endeksi (2010=100)", None),
        # A digit or a dot in the name is still not a country.
        ("F.1.2.Özel Çekme Hakları (SDR)", None),
        # Deliberately a match: the pattern is generous and the datagroup
        # threshold is the guard. On its own this row would be read as a
        # country; `_countries_of_tables` drops it because BDDK rows carry
        # no datagroup. See the two tests below.
        ("ADANA - Diğer Mevduat (TRY)", ("adana - diger mevduat", "TRY")),
    ],
)
def test_a_country_is_read_from_the_name_only_where_there_is_one(name, expected):
    assert pool_index.country_match(name) == expected


def test_a_table_with_one_code_is_not_a_country_breakdown():
    """The pattern alone would mark the house price index as a country:
    "Konut Fiyat Endeksi (KFE)" is three words and three capitals too. What
    separates them is that a real country table carries dozens of codes."""
    rows = [
        {
            "series_id": "evds:kfe",
            "name": "Konut Fiyat Endeksi (KFE)",
            "group": "bie_kfe",
        },
        *(
            {
                "series_id": f"evds:imf_{code}",
                "name": f"{label} ({code}) Tuketici Fiyat Endeksi",
                "group": "bie_imfcpind",
            }
            for label, code in [
                ("Arnavutluk", "ALB"),
                ("Ermenistan", "ARM"),
                ("Avusturya", "AUT"),
                ("Azerbaycan", "AZE"),
                ("Belcika", "BEL"),
                ("Brezilya", "BRA"),
                ("Isvicre", "CHE"),
                ("Sili", "CHL"),
                ("Kolombiya", "COL"),
                ("Cekya", "CZE"),
            ]
        ),
    ]
    frame = pd.DataFrame(
        [
            {
                "series_id": r["series_id"],
                "group": r["group"],
                "country": pool_index.country_match(r["name"]),
            }
            for r in rows
        ]
    )

    marked = pool_index._countries_of_tables(frame)

    assert marked[0] == ""  # the house price index is not a country
    assert marked[1] == "arnavutluk"
    assert sum(1 for m in marked if m) == 10


def test_the_older_edition_of_a_rebased_table_is_the_one_marked():
    """Both TÜFE editions are live and neither is marked "(Arşiv)", so the
    base year in the name is all there is to go on. The names do not agree
    on the rest, which is why every parenthetical is dropped before they are
    compared."""
    meta = pd.DataFrame(
        [
            {"group": "bie_tukfiy2025", "table": "Tüketici Fiyat Endeksi (2025=100)"},
            {
                "group": "bie_tukfiy2003",
                "table": "Tüketici Fiyat Endeksi (Genel) (2003=100)",
            },
            {"group": "bie_kfe", "table": "Konut Fiyat Endeksi"},
            {
                "group": "bie_gecinme",
                "table": "Geçinme Endeksi (Ücretliler) (1968=100)",
            },
        ]
    )

    superseded = pool_index._superseded(meta)

    assert superseded == {"bie_tukfiy2003": "2003"}


_CPI_META = pd.DataFrame(
    [
        {
            "series_id": "evds:imf_deu",
            "table": "IMF - Tüketici Fiyat Endeksi (2010=100)",
            "places": "",
            "country": "almanya",
            "archive": False,
            "family": None,
            "group": "bie_imfcpind",
            "head": True,
        },
        {
            "series_id": "evds:tufe_2003",
            "table": "Tüketici Fiyat Endeksi (Genel) (2003=100)",
            "places": "",
            "country": "",
            "archive": False,
            "family": None,
            "group": "bie_tukfiy2003",
            "head": True,
        },
        {
            "series_id": "evds:tufe_2025",
            "table": "Tüketici Fiyat Endeksi (2025=100)",
            "places": "",
            "country": "",
            "archive": False,
            "family": None,
            "group": "bie_tukfiy2025",
            "head": True,
        },
    ]
)
#: The order cosine alone produced, measured 23 Sep on the real pool.
_CPI_SCORED = [
    ("evds:imf_deu", 0.6680),
    ("evds:tufe_2003", 0.6665),
    ("evds:tufe_2025", 0.6484),
]


def test_the_contract_index_wins_a_query_that_names_no_country_or_base():
    """The eval's A6 in one assertion: on cosine alone "tüketici fiyat
    endeksi" answered with Germany's CPI, and the series the contract fixes
    came third."""
    ranked, _ = pool_index.rank("tüketici fiyat endeksi", _CPI_SCORED, _CPI_META)

    assert [sid for sid, _ in ranked][0] == "evds:tufe_2025"
    assert dict(ranked)["evds:tufe_2025"] == 0.6484  # scores are not touched


def test_naming_the_country_asks_for_that_country():
    ranked, _ = pool_index.rank(
        "Almanya tüketici fiyat endeksi", _CPI_SCORED, _CPI_META
    )

    assert [sid for sid, _ in ranked][0] == "evds:imf_deu"


def test_naming_the_base_year_asks_for_that_edition():
    ranked, _ = pool_index.rank("2003=100 tüfe endeksi", _CPI_SCORED, _CPI_META)

    assert [sid for sid, _ in ranked][0] == "evds:tufe_2003"


def test_an_index_built_before_these_rules_still_ranks():
    """`data/pool_index` is derived and gitignored, so a stale copy without
    the `country` column has to rank worse, not raise."""
    stale = _CPI_META.drop(columns=["country"])

    ranked, _ = pool_index.rank("tüketici fiyat endeksi", _CPI_SCORED, stale)

    # The country rule cannot fire without its column, so Germany keeps the
    # cosine lead it had; what must not happen is a crash, and the base-year
    # rule still separates the two Turkish editions.
    ids = [sid for sid, _ in ranked]
    assert len(ids) == 3
    assert ids.index("evds:tufe_2025") < ids.index("evds:tufe_2003")


# --- what the 23 Sep review found ------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Two letters, and the policy-rate table's other survivor. Missing it
        # meant "politika faiz oranı" could answer with the euro area rate:
        # the family collapses to one candidate, so whichever of Türkiye and
        # the euro area is not demoted wins on cosine alone.
        (
            "Euro Bölgesi (XM) Merkez Bankası Politika Faiz Oranı",
            ("euro bolgesi", "XM"),
        ),
        ("Gürcistan(Geo) Tüketici Fiyat Endeksi", ("gurcistan", "GEO")),
        ("Bosna-Hersek (BIH) Cari İşlemler Dengesi Net", ("bosna-hersek", "BIH")),
        (
            "Güney Kıbrıs Rum Yönetimi (CYP) Tüketici Fiyat Endeksi",
            ("guney kibris rum yonetimi", "CYP"),
        ),
        ("Türkiye (TUR) Tüketici Fiyat Endeksi", None),
    ],
)
def test_the_pattern_reaches_the_countries_it_used_to_miss(name, expected):
    assert pool_index.country_match(name) == expected


def test_a_series_outside_any_datagroup_is_never_a_country():
    """Group-less rows used to share the bucket "None". With `evds=None` --
    the documented degraded build, and what a fresh clone gets, since bronze
    is gitignored -- that bucket cleared the threshold and marked "Konut
    Fiyat Endeksi (KFE)" as a country."""
    codes = [
        "ALB",
        "ARM",
        "AUT",
        "AZE",
        "BEL",
        "BRA",
        "CHE",
        "CHL",
        "COL",
        "CZE",
        "DNK",
    ]
    names = [f"Ulke{code.lower()} ({code}) Tuketici Fiyat Endeksi" for code in codes]
    names.append("Konut Fiyat Endeksi (KFE)")
    frame = pd.DataFrame(
        {
            "group": [None] * len(names),
            "country": [pool_index.country_match(name) for name in names],
        }
    )
    # Every row matches the pattern and there are more than MIN_COUNTRY_CODES
    # distinct codes, so only the missing datagroup can keep them unmarked.
    assert sum(1 for m in frame["country"] if m) == len(names)

    assert set(pool_index._countries_of_tables(frame)) == {""}


@pytest.mark.parametrize(
    ("query", "country", "named"),
    [
        # The eval's own A3 turned the rule off for every Chinese series.
        ("toplam konut satışları içindeki payı nedir", "cin", False),
        ("ABD doları cinsinden GSYH", "cin", False),
        ("karşılık ayrılan krediler", "sili", False),
        ("Çin'in enflasyonu", "cin", True),
        ("Almanya tüketici fiyat endeksi", "almanya", True),
    ],
)
def test_a_country_is_named_as_a_word_not_as_a_fragment(query, country, named):
    assert pool_index._country_named(country, pool_index._fold(query)) is named


def test_running_the_country_pass_twice_does_not_raise():
    frame = pd.DataFrame({"group": ["g"], "country": [("almanya", "DEU")]})

    once = pool_index._countries_of_tables(frame)

    assert pool_index._countries_of_tables(frame.assign(country=once)) == [""]


def test_an_index_missing_a_ranking_column_says_so():
    """The manifest digest hashes ids and search texts, so a rule that adds a
    meta column changes neither. Without this note a stale index silently
    ranks the way it used to."""
    meta = pd.DataFrame(
        {
            "series_id": ["evds:X"],
            "table": [None],
            "places": [""],
            "country": [""],
            "archive": [False],
            "family": [None],
        }
    )

    assert pool_index.meta_notes(meta) == []
    notes = pool_index.meta_notes(meta.drop(columns=["country"]))
    assert len(notes) == 1
    assert "country" in notes[0] and "yeniden kur" in notes[0]
