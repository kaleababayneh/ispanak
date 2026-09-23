"""The pool's search index: every gold series' search text and its vector,
in LanceDB.

The organisers' guidance on vector stores (Slack, docs/decisions.md): not
for the data itself, for "which data is in which table, what does each
column mean". This is that map, for the whole pool -- tens of thousands of
EVDS series, every BDDK bulletin row, every FinTürk province.

At that size the vectors cannot live in the search call: 45k texts x 4096
dimensions is ~1.5 GB of float64, and embedding them is minutes of MIA calls
(measured 22 Sep: 32.7k texts in 164 s, 8 workers). So the index is built
once, after gold
(`python -m lakehouse_agent.lake.pool_index`), through the on-disk embedding
cache (`lake.catalog.embed_with_cache`: a text already embedded is never
embedded again), and a search only embeds its query.

The index records a digest of the catalog it was built from. A search over a
changed catalog still uses it -- most of it is right -- and says how many pool
series it does not cover; those are found by the keyword match instead.

Scores are cosine similarities (LanceDB's cosine distance is 1 - cos), so the
0.40 floor and the approval gate's ratio keep their meaning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A LanceDB database directory. Derived and large: gitignored (data/*).
POOL_INDEX_PATH = Path("data/pool_index")
TABLE_NAME = "pool"
MANIFEST_NAME = "manifest.json"
#: The ids the index covers, apart from the vectors: reading them from the
#: table would read the vector column too (hundreds of MB).
IDS_NAME = "ids.parquet"

_FREQ_LABEL_TR = {
    "D": "günlük",
    "W": "haftalık",
    "M": "aylık",
    "Q": "üç aylık",
    "A": "yıllık",
}
_KIND_LABEL_TR = {
    "stock": "stok (bakiye)",
    "flow": "akım",
    "rate": "oran",
    "index": "endeks",
}
_SOURCE_LABEL_TR = {
    "EVDS": "TCMB EVDS",
    "BDDK-Aylik": "BDDK Aylık Bülten",
    "BDDK-Haftalik": "BDDK Haftalık Bülten",
    "BDDK-FinTurk": "BDDK FinTürk, illere göre",
}
#: BDDK's column abbreviations, spelled out once in the text a search reads:
#: "(TP)" alone gives an embedding nothing to match "Türk lirası" against.
_ABBREVIATIONS = {
    "TP": "Türk parası",
    "YP": "yabancı para",
    "DTH": "döviz tevdiat hesabı",
}
_ABBREVIATION = re.compile(r"\b(TP|YP|DTH)\b")


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        return value
    return None if missing else value


def search_text(row: pd.Series, *, table: str | None = None) -> str:
    """What a pool series is matched on.

    Plan 2.5's fields -- name, topic, tags, unit, frequency, source -- plus
    what kind of quantity it is. The last one carries weight for BDDK rows,
    which have no topic or tags: "stok (bakiye)" is what lets "konut kredisi
    bakiyesi" find "Tüketici Kredileri - Konut". Only labels for codes the
    catalog already holds are added, and BDDK's column abbreviations spelled
    out; no description is invented.

    `table` is the source table the series sits in, when the catalog does not
    say it: an EVDS datagroup's name. "Genel Endeks" says nothing alone; in
    "Tüketici Fiyat Endeksi (2025=100)" it is the CPI (measured 22 Sep: without
    it, "tüketici fiyat endeksi" ranked the headline index below twenty of its
    own sub-items and other countries' CPIs).
    """
    source = _clean(row.get("source"))
    name = _clean(row.get("name_tr"))
    name = " ".join(str(name).split()) if name is not None else None
    spelled = sorted(
        {_ABBREVIATIONS[m] for m in _ABBREVIATION.findall(str(name or ""))}
    )
    topic = _clean(row.get("topic"))
    table = " ".join(table.split()) if isinstance(table, str) else None
    if table and _fold(table) in (_fold(name or ""), _fold(topic or "")):
        table = None
    parts = [
        name,
        ", ".join(spelled) or None,
        table,
        topic,
        _clean(row.get("tags")),
        _clean(row.get("unit")),
        _FREQ_LABEL_TR.get(_clean(row.get("freq"))),
        _KIND_LABEL_TR.get(_clean(row.get("kind"))),
        _SOURCE_LABEL_TR.get(source, source),
    ]
    return " | ".join(str(part).strip() for part in parts if part not in (None, ""))


def catalog_texts(
    catalog: pd.DataFrame, tables: dict[str, str] | None = None
) -> list[str]:
    tables = tables or {}
    return [
        search_text(row, table=tables.get(row["series_id"]))
        for _, row in catalog.iterrows()
    ]


# --- ranking: what a reader of five candidates needs ---------------------
#
# Cosine similarity alone, over tens of thousands of series, fills the five
# candidates with near copies of one another (measured 22 Sep on 32.7k
# series, 10 of 21 labelled queries hit in the top five): 81 provinces'
# "Takipteki Konut Kredisi" above the national balance, the archived 2017=100
# house price index above the current one. The rules below reorder and group
# candidates by what the sources themselves say about a series -- its place,
# EVDS's own "(Arşiv)" mark, its parent -- never by the query's topic. The
# score stays the cosine similarity: the floor and the approval gate read it.

#: The 81 provinces, folded (`_fold`). A series naming one is that
#: province's share of a figure: EVDS's province tables
#: ("Kahramanmaraş_Konut_Toplam Satışlar"), KFE's regions ("TR10
#: (İstanbul)"). FinTürk carries the province in its id instead.
PROVINCES = frozenset(
    {
        "adana", "adiyaman", "afyonkarahisar", "agri", "aksaray", "amasya",
        "ankara", "antalya", "ardahan", "artvin", "aydin", "balikesir",
        "bartin", "batman", "bayburt", "bilecik", "bingol", "bitlis", "bolu",
        "burdur", "bursa", "canakkale", "cankiri", "corum", "denizli",
        "diyarbakir", "duzce", "edirne", "elazig", "erzincan", "erzurum",
        "eskisehir", "gaziantep", "giresun", "gumushane", "hakkari", "hatay",
        "igdir", "isparta", "istanbul", "izmir", "kahramanmaras", "karabuk",
        "karaman", "kars", "kastamonu", "kayseri", "kilis", "kirikkale",
        "kirklareli", "kirsehir", "kocaeli", "konya", "kutahya", "malatya",
        "manisa", "mardin", "mersin", "mugla", "mus", "nevsehir", "nigde",
        "ordu", "osmaniye", "rize", "sakarya", "samsun", "sanliurfa", "siirt",
        "sinop", "sirnak", "sivas", "tekirdag", "tokat", "trabzon", "tunceli",
        "usak", "van", "yalova", "yozgat", "zonguldak",
    }
)  # fmt: skip
_PLACE = re.compile(r"(?<![a-z0-9])(" + "|".join(sorted(PROVINCES)) + r")(?![a-z0-9])")
#: Look like a province and are not: a country's ISO code ("Morityus (MUS)"
#: is Mauritius, not Muş -- eight IMF series) and the exchange.
_ISO_CODE = re.compile(r"\([A-Z]{3}\)")
_EXCHANGE = re.compile(r"borsa\s+[iİ]stanbul", re.IGNORECASE)
#: Words by which a query asks for the breakdown itself ("illere göre").
_BREAKDOWN_WORDS = frozenset(
    {
        "il", "iller", "illere", "illerde", "illerin", "ilce", "ilceler",
        "bolge", "bolgeler", "bolgelere", "bolgesel", "sehir", "sehirler",
    }
)  # fmt: skip
_TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def _fold(text: str) -> str:
    """Lower-case ASCII, spaces squeezed: "İSTANBUL" and "istanbul" meet."""
    folded = str(text).replace("İ", "i").replace("I", "ı").lower()
    return " ".join(folded.translate(_TR_FOLD).split())


def places(text: str) -> frozenset[str]:
    """The provinces a series name or a query names, folded."""
    cleaned = _EXCHANGE.sub(" ", _ISO_CODE.sub(" ", str(text)))
    return frozenset(_PLACE.findall(_fold(cleaned)))


def query_places(query: str) -> frozenset[str] | None:
    """The provinces a query asks about; None when it asks for the breakdown
    itself ("illere göre konut kredisi"), so that no place is put back."""
    words = set(re.findall(r"[a-z]+", _fold(query)))
    if words & _BREAKDOWN_WORDS:
        return None
    return places(query)


def series_meta(
    catalog: pd.DataFrame, evds: pd.DataFrame | None = None
) -> pd.DataFrame:
    """What the ranking reads about each pool series, one row each:

    * `table`: the EVDS datagroup's name (search text context, see
      `search_text`); None elsewhere, where the catalog's topic is the table.
    * `places`: the provinces it is about, "|"-joined; empty for a national
      series.
    * `archive`: EVDS marks the series or its datagroup "(Arşiv)" -- kept for
      its history, replaced by a current series.
    * `family`: set for a place's share of a figure, the same for every
      place: EVDS's parent series (or the datagroup, for KFE's regions that
      have no parent), FinTürk's table and column. One candidate stands for
      the family.
    * `group`, `head`: the EVDS datagroup and whether the series is the one
      EVDS lists first in it. That first series is the table's own total
      ("Genel Endeks", "Toplam Sanayi", "Türkiye"), which a query naming the
      table asks for -- see `rank`.

    `evds` is the bronze catalog's SERIE_CODE, DATAGROUP_CODE,
    DATAGROUP_NAME, SEVIYE and UST_SERIE_CODE; without it EVDS series get no
    table and no parent.
    """
    by_code = (
        evds.drop_duplicates("SERIE_CODE").set_index("SERIE_CODE")
        if evds is not None and not evds.empty
        else None
    )
    first_of_group: set[str] = set()
    if by_code is not None and "SCREEN_ORDER" in by_code.columns:
        order = by_code.reset_index().sort_values("SCREEN_ORDER", kind="stable")
        first_of_group = set(order.drop_duplicates("DATAGROUP_CODE")["SERIE_CODE"])
    rows = []
    for series_id, name in zip(catalog["series_id"], catalog["name_tr"], strict=True):
        name = str(_clean(name) or "")
        table = None
        family = None
        group = None
        head = False
        if series_id.startswith("bddk:finturk:"):
            parts = series_id.split(":")
            found = frozenset({parts[3]})
            family = f"finturk:{parts[2]}:{parts[4]}"
        else:
            found = places(name)
        if series_id.startswith("evds:") and by_code is not None:
            code = series_id.split(":", 1)[1]
            if code in by_code.index:
                source = by_code.loc[code]
                table = _clean(source.get("DATAGROUP_NAME"))
                group = _clean(source.get("DATAGROUP_CODE"))
                head = code in first_of_group
                parent = _clean(source.get("UST_SERIE_CODE"))
                if found:
                    family = (
                        f"evds:{parent}"
                        if parent not in (None, "-1", -1)
                        else f"evds-group:{source.get('DATAGROUP_CODE')}"
                    )
        archive = "arsiv" in _fold(name) or "arsiv" in _fold(table or "")
        rows.append(
            {
                "series_id": series_id,
                "table": table,
                "places": "|".join(sorted(found)),
                "archive": archive,
                "family": family,
                "group": group,
                "head": head,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["series_id", "table", "places", "archive", "family", "group", "head"],
    )


def evds_hierarchy() -> pd.DataFrame | None:
    """The bronze EVDS catalog's table and hierarchy columns, if it is here."""
    from lakehouse_agent.ingest.evds import CATALOG_PATH

    if not CATALOG_PATH.exists():
        return None
    columns = ["SERIE_CODE", "DATAGROUP_CODE", "DATAGROUP_NAME", "SEVIYE"]
    columns += ["UST_SERIE_CODE", "SCREEN_ORDER"]
    frame = pd.read_parquet(CATALOG_PATH, columns=columns)
    frame["UST_SERIE_CODE"] = frame["UST_SERIE_CODE"].astype("string")
    return frame


def rank(
    query: str, scored: list[tuple[str, float]], meta: pd.DataFrame | None
) -> tuple[list[tuple[str, float]], dict[str, int]]:
    """Candidates (best cosine first) in the order a reader should see them,
    and, per candidate standing for a family, how many other places it hides.

    Tiers, each keeping the similarity order. First the total of a table the
    query spelled out ("konut fiyat endeksi" asks for the KFE itself). Then
    everything else; and after those, a series the query did not ask for --
    an archived series unless the query says "arşiv", a province's share
    unless the query names that province (or asks for the breakdown), and a
    table's sub-item when the query named the table itself ("tüketici fiyat
    endeksi" asks for the CPI, not for "011. Gıda"; EVDS lists the table's
    own total first, see `series_meta`). Then one candidate per family: the
    first, which is the province the query names if it names one.
    """
    if meta is None or meta.empty:
        return scored, {}
    wanted = query_places(query)
    wants_archive = "arsiv" in _fold(query)
    asked = tables_asked(query, meta)
    info = meta.set_index("series_id")
    has_groups = "group" in info.columns

    def _row(series_id: str) -> pd.Series | None:
        return info.loc[series_id] if series_id in info.index else None

    def _sub_item(row: pd.Series) -> bool:
        """In a table the query named, and not that table's own total."""
        return has_groups and row["group"] in asked and not bool(row.get("head", False))

    def _asked_total(row: pd.Series) -> bool:
        """The own total of a table the query spelled out."""
        return has_groups and row["group"] in asked and bool(row.get("head", False))

    def tier(series_id: str) -> tuple[int, int]:
        """(promoted first, then how late). Two keys, not one sum.

        Naming a table used to demote its sub-items and do nothing else, so
        the table's own total stayed in the same tier as every unrelated
        series and had to win on cosine alone. Measured on "konut fiyat
        endeksi": `Konut Fiyat Endeksi (KFE)` scored 0.639 while four
        housing-related *CPI* rows scored 0.645 to 0.684, so the five
        candidates held none of them. The docstring already said such a query
        asks for the table's own total; now it does.

        Kept separate from `late` so the promotion cannot cancel a demotion:
        the archived head of an archived table stays behind, which is why
        `TP.HKFE01` (Arşiv, 0.726) does not take the place of the current
        index.
        """
        row = _row(series_id)
        if row is None:
            return (1, 0)
        late = 0
        if bool(row["archive"]) and not wants_archive:
            late += 1
        named = frozenset(filter(None, str(row["places"] or "").split("|")))
        if wanted is not None and named and not named & wanted:
            late += 1
        if _sub_item(row):
            late += 1
        return (0 if late == 0 and _asked_total(row) else 1, late)

    ordered = sorted(scored, key=lambda pair: tier(pair[0]))
    families = meta["family"].copy()
    if has_groups and asked:
        sub = meta["group"].isin(asked) & ~meta["head"].astype(bool)
        families = families.where(~sub | families.notna(), "grup:" + meta["group"])
    sizes = families.value_counts()
    family_of = dict(zip(meta["series_id"], families, strict=True))
    out: list[tuple[str, float]] = []
    standing: dict[str, int] = {}
    seen: set[str] = set()
    for series_id, score in ordered:
        family = family_of.get(series_id)
        if isinstance(family, str):
            if family in seen:
                continue
            seen.add(family)
            if sizes.get(family, 1) > 1:
                standing[series_id] = int(sizes[family]) - 1
        out.append((series_id, score))
    return out, standing


def tables_asked(query: str, meta: pd.DataFrame) -> set[str]:
    """The EVDS datagroups whose name the query spells out ("tüketici fiyat
    endeksi" -> "Tüketici Fiyat Endeksi (2025=100)"): every word of the query
    is in the table's name. Such a query asks for the table's own total, not
    for one of its rows."""
    if "group" not in meta.columns:
        return set()
    words = set(re.findall(r"[a-z]+", _fold(query)))
    if len(words) < 2:
        return set()
    names = meta[["group", "table"]].dropna().drop_duplicates("group")
    return {
        group
        for group, table in zip(names["group"], names["table"], strict=True)
        if words <= set(re.findall(r"[a-z]+", _fold(table)))
    }


def digest(series_ids: list[str], texts: list[str]) -> str:
    """What an index was built from: every id with its text."""
    hasher = hashlib.sha256()
    for series_id, text in sorted(zip(series_ids, texts, strict=True)):
        hasher.update(f"{series_id}\t{text}\n".encode())
    return hasher.hexdigest()


EmbedDocuments = Callable[[list[str]], tuple[np.ndarray, list[int]]]


def _embed_documents_cached(texts: list[str]) -> tuple[np.ndarray, list[int]]:
    from lakehouse_agent.lake import catalog

    return catalog.embed_with_cache(texts)


def build(
    catalog: pd.DataFrame,
    *,
    path: Path = POOL_INDEX_PATH,
    embed_documents: EmbedDocuments = _embed_documents_cached,
    evds: pd.DataFrame | None = None,
) -> dict:
    """Embed every catalog row's search text and write the index. Rows whose
    embedding failed are left out and counted in the manifest; a search
    finds them by keyword. `evds` is `evds_hierarchy()`: the datagroup names
    for the texts and the parents for the ranking (`series_meta`), stored
    beside the ids. Returns the manifest."""
    import lancedb
    from lancedb.index import FTS

    catalog = catalog.reset_index(drop=True)
    meta = series_meta(catalog, evds)
    texts = catalog_texts(
        catalog, dict(zip(meta["series_id"], meta["table"], strict=True))
    )
    vectors, failed = embed_documents(texts)
    vectors = np.asarray(vectors, dtype=np.float32)
    keep = [i for i in range(len(texts)) if i not in set(failed)]
    frame = pd.DataFrame(
        {
            "series_id": catalog["series_id"].iloc[keep].to_numpy(),
            "text": [texts[i] for i in keep],
            "vector": [vectors[i] for i in keep],
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(path))
    table = db.create_table(TABLE_NAME, frame, mode="overwrite")
    table.create_index("text", config=FTS())
    from lakehouse_agent.config import get_agent_settings

    try:
        model = get_agent_settings().mia_embed_model
    except Exception:  # noqa: BLE001 - no key here is not a reason not to build
        model = ""
    manifest = {
        "digest": digest(list(catalog["series_id"]), texts),
        "rows": len(frame),
        "failed": len(failed),
        "dim": int(vectors.shape[1]) if len(vectors) else 0,
        # Which model wrote these vectors. Without it, an index built by one
        # embedding model and queried by another of the same width returns
        # cosine numbers that mean nothing at all, and nothing anywhere says
        # so; a different width at least raises. `LakeAdapter` compares it.
        "embed_model": model,
        "built_at": datetime.now(UTC).isoformat(),
    }
    meta[meta["series_id"].isin(set(frame["series_id"]))].to_parquet(
        path / IDS_NAME, index=False
    )
    (path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    logger.info("havuz indeksi: %d seri, %d embed edilemedi", len(frame), len(failed))
    return manifest


@dataclass
class PoolIndex:
    """An open index. `ids` is what it covers, `meta` what `rank` reads
    (`series_meta`; None for an index built before it was stored)."""

    table: Any
    manifest: dict
    ids: frozenset[str]
    meta: pd.DataFrame | None = None

    @classmethod
    def open(cls, path: Path = POOL_INDEX_PATH) -> PoolIndex | None:
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            return None
        import lancedb

        try:
            table = lancedb.connect(str(path)).open_table(TABLE_NAME)
            stored = pd.read_parquet(path / IDS_NAME)
            ids = frozenset(stored["series_id"])
            # Inside the guard with the rest: the manifest is written last and
            # not atomically, so a build killed at that moment leaves a file
            # that parses as nothing. Read outside, it raised out of every
            # later search instead of degrading to the keyword match this
            # handler exists for.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a broken index is a missing one
            logger.warning("havuz indeksi acilamadi (%s): %s", type(exc).__name__, exc)
            return None
        meta = stored if "family" in stored.columns else None
        return cls(table=table, manifest=manifest, ids=ids, meta=meta)

    def nearest(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """(series_id, cosine similarity), best first."""
        hits = (
            self.table.search(np.asarray(query_vector, dtype=np.float32))
            .distance_type("cosine")
            .limit(k)
            .select(["series_id"])
            .to_pandas()
        )
        return [
            (series_id, 1.0 - float(distance))
            for series_id, distance in zip(
                hits["series_id"], hits["_distance"], strict=True
            )
        ]

    def keyword(self, query: str, k: int) -> list[str]:
        """Candidate ids for a keyword match, from the full-text index."""
        hits = (
            self.table.search(query, query_type="fts")
            .limit(k)
            .select(["series_id"])
            .to_pandas()
        )
        return list(hits["series_id"])


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from lakehouse_agent.lake import gold

    manifest = build(pd.read_parquet(gold.SERIES_CATALOG_PATH), evds=evds_hierarchy())
    logger.info("havuz indeksi yazildi: %s %s", POOL_INDEX_PATH, manifest)
