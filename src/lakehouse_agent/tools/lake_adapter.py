"""The one lake surface the agent is written against.

`docs/contracts.md` describes a single object with seven functions. The data
lane implements those across four modules -- `lake.gold` (get/wide),
`analysis.transforms` (to_freq), `lake.quality` (profile_table) and
`lake.catalog` (search) -- and does not implement three of them at all. This
class is the join: it is what `FakeLake` is a stand-in *for*, so swapping one
for the other changes the object the agent is handed and nothing else.

Two things it adds rather than forwards, both because the contract asks for
them and the data lane does not provide them:

* **Unknown series ids raise `SeriesNotFound`.** `gold.get` answers an
  unknown id with an empty frame, which reaches the model as "no data for
  2021-2025" when the real problem is a typo in a series id. The exception
  carries the ids, so the loop turns it into a message the model can act on.
* **`search` answers over the pool's own catalog.** The organisers' guidance
  (Slack, recorded in docs/decisions.md): a vector index is for "which data is
  in which table, what does each column mean" -- a map of *our* lake. The
  earlier index mapped EVDS's catalog instead (19.6k series, almost none of
  them in the lake) and, because BDDK has no such catalog, could never return
  a BDDK series at all, although the scenario's credit balance is one.

`prepare_table` is implemented here too: nothing in the data lane offers it,
and the demo-time path (a table arriving from a URL) needs it.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lakehouse_agent.analysis import transforms
from lakehouse_agent.contracts import CatalogHit, QualityReport, SeriesNotFound
from lakehouse_agent.lake import gold, pool_index, quality
from lakehouse_agent.lake.pool_index import POOL_INDEX_PATH, PoolIndex, search_text
from lakehouse_agent.tools.lakehouse import column_name

logger = logging.getLogger(__name__)

#: Candidates below this cosine similarity are not offered at all: a series
#: that has nothing to do with the question is worse than no candidate, since
#: the model will happily build an analysis on it.
#:
#: Measured 22 Sep 2026, Qwen3-Embedding-8B, the 9-series pool. The weakest
#: *relevant* top hit was 0.470 ("enflasyon" -> TÜFE). Of seven unrelated
#: queries, four top out below 0.40 (dolar kuru 0.329, altın 0.262, cari açık
#: 0.230, işsizlik 0.386) and three do not (BIST 100 0.495, mevduat faizi
#: 0.462, otomobil kredisi 0.457). No threshold separates those three from
#: relevant hits, so this one only removes the clearly unrelated; telling
#: "BIST 100" from CPI is left to the model reading the candidates' names.
MIN_SIMILARITY = 0.40

#: The keyword fallback's floor: at least one of the query's words matched.
MIN_LEXICAL_SCORE = 1e-9

#: After a failed embedding call, searches skip it for this long and use the
#: keyword match straight away. A hung endpoint costs a whole client timeout
#: (45 s) per call, so re-trying it on every search spent most of a turn;
#: one turn's budget later it is worth trying again.
EMBED_RETRY_AFTER_S = 180.0

#: Up to this many pool series, a search without a built index embeds the
#: catalog itself (tests, a small demo pool). Beyond it, that would mean
#: embedding tens of thousands of texts inside one search call.
IN_MEMORY_LIMIT = 2000

#: How many nearest series a search over the pool index reads before
#: `pool_index.rank` orders and groups them: 81 provinces of one figure must
#: not use up the candidates a national series would have come from, and a
#: table's own total can sit below hundreds of its rows (measured 22 Sep:
#: "tüketici fiyat endeksi" puts TÜFE's "Genel Endeks" 232nd by cosine,
#: behind its sub-indices). Reading 500 costs ~0.1 s of the search.
SEARCH_DEPTH = 500

#: The note a keyword-fallback search carries (see `search_detailed`).
KEYWORD_FALLBACK_NOTE = (
    "Anlamsal arama yapilamadi; seriler kelime eslesmesiyle bulundu, "
    "siralama daha kaba olabilir."
)

#: Short, frequent words that say nothing about which series is meant, in
#: their folded spelling (see `_tokens`).
_STOPWORDS = frozenset({"ve", "ile", "icin", "bir", "bu", "mi", "ne", "nasil"})

#: Turkish letters folded to ASCII, so "tufe" meets "TÜFE" and "satis" meets
#: "Satışlar": a query typed without them is the ordinary case.
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")

EmbedDocuments = Callable[[list[str]], np.ndarray]
EmbedQuery = Callable[[str], np.ndarray]


def _embed_documents_cached(texts: list[str]) -> np.ndarray:
    """Pool catalog texts as vectors, through the on-disk embedding cache, so
    after the first search of a new catalog no document is embedded again."""
    from lakehouse_agent.lake import catalog

    vectors, failed = catalog.embed_with_cache(texts)
    if failed:
        raise RuntimeError(f"{len(failed)} katalog metni embed edilemedi")
    return vectors


def _embed_query(query: str) -> np.ndarray:
    from lakehouse_agent.lake import catalog

    return catalog.embed_query(query)


class LakeAdapter:
    """The real data lane, presented as the contract's lake surface."""

    def __init__(
        self,
        *,
        catalog_path: Path | None = None,
        embed_documents: EmbedDocuments | None = None,
        embed_query: EmbedQuery | None = None,
        clock: Callable[[], float] = time.monotonic,
        index_path: Path | None = None,
        in_memory_limit: int = IN_MEMORY_LIMIT,
    ) -> None:
        self._catalog_path = Path(catalog_path or gold.SERIES_CATALOG_PATH)
        self._catalog: pd.DataFrame | None = None
        self._texts: list[str] | None = None
        self._index_path = Path(index_path or POOL_INDEX_PATH)
        self._index: PoolIndex | None = None
        self._index_checked = False
        self._index_notes: list[str] = []
        self._in_memory_limit = in_memory_limit
        self._embed_documents = embed_documents or _embed_documents_cached
        self._embed_query = embed_query or _embed_query
        self._clock = clock
        self._doc_vectors: np.ndarray | None = None
        self._embed_down_until: float | None = None
        self._coverage_cache: dict[str, tuple[str, str]] | None = None

    # -- catalog -----------------------------------------------------------

    def _series_catalog(self) -> pd.DataFrame:
        """`series_catalog.parquet`, read once. Cached because every call that
        validates a series id would otherwise re-read the file."""
        if self._catalog is None:
            if not self._catalog_path.exists():
                raise FileNotFoundError(
                    f"series_catalog bulunamadi: {self._catalog_path}. "
                    "Once `python -m lakehouse_agent.lake.gold` calistirin."
                )
            self._catalog = pd.read_parquet(self._catalog_path)
        return self._catalog

    def _require_usable(self, series_ids: list[str], *, caller: str) -> None:
        """Reject an id list that cannot produce a well-formed wide table.

        `FakeLake.wide` refuses both of these, so the adapter has to as well:
        code written against the fake gets a `ValueError` naming the problem,
        and without this it got an `IOException` from DuckDB (empty list) or a
        table with `<id>_x` / `<id>_y` columns (duplicates) instead.
        """
        if not series_ids:
            raise ValueError(f"{caller}() icin en az bir series_id gerekli.")
        duplicates = sorted({s for s in series_ids if series_ids.count(s) > 1})
        if duplicates:
            raise ValueError(
                f"series_ids listesinde tekrar var: {', '.join(duplicates)}"
            )

    def _require_known(self, series_ids: list[str]) -> None:
        """Raise before touching the data, naming every unknown id at once.

        One call with three typos should report three, not send the caller
        round the loop once per id.
        """
        known = set(self._series_catalog()["series_id"])
        missing = [sid for sid in series_ids if sid not in known]
        if missing:
            raise SeriesNotFound(missing, "Once search_catalog ile arama yapin.")

    # -- reads -------------------------------------------------------------

    def get(self, series_ids: list[str], start: str, end: str) -> pd.DataFrame:
        ids = list(series_ids)
        self._require_known(ids)
        return gold.get(ids, start, end)

    def wide(
        self, series_ids: list[str], start: str, end: str, freq: str | None = "M"
    ) -> pd.DataFrame:
        ids = list(series_ids)
        self._require_usable(ids, caller="wide")
        self._require_known(ids)
        result = gold.wide(ids, start, end, freq)

        # An empty window comes back from `gold.wide` as the LONG frame -- the
        # seven observation columns -- because there was nothing to pivot. The
        # contract says wide is always `period` plus one column per series, and
        # a caller reading `result[series_id]` should get an empty column, not
        # a KeyError precisely when there is nothing to see.
        if result.empty or "period" not in result.columns:
            return pd.DataFrame({"period": pd.Series(dtype="datetime64[ns]")}).assign(
                **{sid: pd.Series(dtype="float64") for sid in ids}
            )
        for sid in ids:
            if sid not in result.columns:
                result[sid] = pd.Series(dtype="float64")
        return result[["period", *ids]]

    # -- pure transforms ---------------------------------------------------

    def to_freq(
        self,
        df: pd.DataFrame,
        kind: str | None,
        freq: str,
        *,
        agg_method: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Aggregate to a lower frequency, refusing to go the other way.

        CLAUDE.md #4 and `docs/contracts.md` both say high-to-low only, and
        `FakeLake.to_freq` enforces it; `analysis.transforms.to_freq` does not,
        so the guard lives here until it moves into the data lane. Resampling
        monthly rows to weeks invents no numbers -- the empty weeks drop out --
        but it relabels monthly observations as weekly ones, and every reader
        downstream believes the label.
        """
        source = _infer_freq(df["period"]) if "period" in df.columns else None
        if source is not None and _FREQ_RANK.get(source, 0) > _FREQ_RANK.get(freq, 0):
            raise ValueError(
                f"Yuksek frekansa cevirme desteklenmiyor: {source} -> {freq}. "
                "Dusuk frekanstan yuksege gitmek veri icat etmektir."
            )
        return transforms.to_freq(df, kind, freq, agg_method=agg_method)

    # -- search ------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> list[CatalogHit]:
        """Pool series matching a natural-language query, best first."""
        hits, _notes = self.search_detailed(query, k)
        return hits

    def search_detailed(
        self, query: str, k: int = 20
    ) -> tuple[list[CatalogHit], list[str]]:
        """`search`, plus notes the caller should pass on to the user.

        Searches exactly what the pool holds: every row of `series_catalog`,
        whatever its source, and nothing else -- so a hit is always a series
        the very next `get()` can serve, and an empty result is the honest
        "we do not have that".

        A query naming a series outright -- by id, bare code or the column
        name the tools give it (`evds:TP.KTF12`, `TP.KTF12'nin`, `TP_KTF12`)
        -- is answered by those series alone (`match="code"`). Otherwise the
        score is the cosine similarity between the query and each series'
        search text (`lake.pool_index.search_text`), an absolute value
        comparable across queries: read from the pool index when one is
        built (`python -m lakehouse_agent.lake.pool_index`), computed here
        for a small pool (IN_MEMORY_LIMIT). If the embedding call fails, or a
        large pool has no index, a keyword match stands in and a note says
        so -- and that an empty result then proves nothing. An embedding
        failure is remembered for `EMBED_RETRY_AFTER_S`, so the searches
        after it do not each wait out the same dead endpoint.
        """
        catalog_df = self._series_catalog().reset_index(drop=True)
        named = named_series(query, catalog_df["series_id"])
        if named:
            return [self._hit(catalog_df.iloc[i], 1.0, match="code") for i in named][
                :k
            ], []

        small = len(catalog_df) <= self._in_memory_limit
        index = None if small else self._pool_index()
        notes = [] if small else list(self._index_notes)
        standing: dict[int, int] = {}
        try:
            if small:
                scores = self._semantic_scores(query, self._catalog_texts())
                order = np.argsort(-scores, kind="stable")
                scored = [(int(i), float(scores[i])) for i in order]
            elif index is not None:
                scored, standing = self._index_scores(query, index, catalog_df, k)
            else:
                notes.append(
                    "Havuz arama indeksi kurulmamis; seriler kelime eslesmesiyle "
                    "bulundu, siralama daha kaba olabilir."
                )
                return self._keyword_hits(query, catalog_df, index, k), notes
        except Exception as exc:  # noqa: BLE001 - any embedding failure falls back
            logger.warning(
                "search: anlamsal arama yapilamadi (%s: %s), kelime eslesmesi",
                type(exc).__name__,
                exc,
            )
            notes.append(KEYWORD_FALLBACK_NOTE)
            return self._keyword_hits(query, catalog_df, index, k), notes

        chosen = [
            (position, score) for position, score in scored if score >= MIN_SIMILARITY
        ]
        hits = [
            self._hit(catalog_df.iloc[position], score, match="semantic")
            for position, score in chosen[:k]
        ]
        for (position, _score), hit in zip(chosen[:k], hits, strict=True):
            if standing.get(position):
                notes.append(
                    f"{hit.name_tr}: ayni kalemin {standing[position]} baska il/bolge "
                    "serisi de var; belirli bir yer icin aramaya yer adini ekle."
                )
        return hits, notes

    def _catalog_texts(self) -> list[str]:
        """Every catalog row's search text, built once per adapter."""
        if self._texts is None:
            self._texts = [
                search_text(row) for _, row in self._series_catalog().iterrows()
            ]
        return self._texts

    def _pool_index(self) -> PoolIndex | None:
        """The built index, opened once; with a note when it was built from
        another catalog than the one being searched."""
        if not self._index_checked:
            self._index_checked = True
            self._index = PoolIndex.open(self._index_path)
            if self._index is not None:
                ids = set(self._series_catalog()["series_id"])
                uncovered = len(ids - self._index.ids)
                if uncovered:
                    self._index_notes.append(
                        f"Arama indeksi guncel degil: {uncovered} havuz serisi "
                        "indekste yok, bunlar anlamsal aramada cikmaz."
                    )
        return self._index

    def _index_scores(
        self, query: str, index: PoolIndex, catalog_df: pd.DataFrame, k: int
    ) -> tuple[list[tuple[int, float]], dict[int, int]]:
        """(catalog position, cosine similarity) from the pool index, in the
        order `pool_index.rank` gives the nearest SEARCH_DEPTH above the
        floor; ids the index knows but the catalog no longer has are dropped.
        Also, per position standing for a family of places, how many it
        hides."""
        self._check_embedding_cooldown()
        try:
            vector = np.asarray(self._embed_query(query), dtype=np.float64).reshape(-1)
        except Exception:
            self._embed_down_until = self._clock() + EMBED_RETRY_AFTER_S
            raise
        self._embed_down_until = None
        positions = {sid: i for i, sid in enumerate(catalog_df["series_id"])}
        nearest = [
            (series_id, score)
            for series_id, score in index.nearest(vector, max(SEARCH_DEPTH, 4 * k))
            if series_id in positions and score >= MIN_SIMILARITY
        ]
        ranked, standing = pool_index.rank(query, nearest, index.meta)
        return (
            [(positions[series_id], score) for series_id, score in ranked],
            {positions[series_id]: n for series_id, n in standing.items()},
        )

    def _keyword_hits(
        self,
        query: str,
        catalog_df: pd.DataFrame,
        index: PoolIndex | None,
        k: int,
    ) -> list[CatalogHit]:
        """The coarse fallback: share of the query's words each text holds.

        On a large pool the index's full-text search picks the candidates
        first, so the scoring does not walk tens of thousands of rows -- and
        its own (BM25) order breaks the ties, which the share alone cannot:
        over 45k series hundreds of texts hold every word of a short query
        and would otherwise be ordered by where they sit in the catalog.
        `pool_index.rank` then applies the same rules as a semantic search.
        """
        texts = self._catalog_texts()
        ids = catalog_df["series_id"]
        positions = list(range(len(catalog_df)))
        by_index: dict[int, int] = {}
        if index is not None:
            try:
                ranked = index.keyword(query, SEARCH_DEPTH)
                order = {sid: rank for rank, sid in enumerate(ranked)}
                found = [i for i, sid in enumerate(ids) if sid in order]
                if found:
                    positions = found
                    by_index = {i: order[ids.iloc[i]] for i in found}
            except Exception as exc:  # noqa: BLE001 - fall back to scoring everything
                logger.warning("search: tam metin indeksi kullanilamadi: %s", exc)
        # The id joins the keyword text only: "toplam tuketici kredileri"
        # can then find the BDDK row named "Tüketici Kredileri (2+3+4)".
        scored = [
            (i, lexical_score(query, f"{texts[i]} | {ids.iloc[i]}")) for i in positions
        ]
        scored = [(i, score) for i, score in scored if score >= MIN_LEXICAL_SCORE]
        scored.sort(key=lambda pair: (-pair[1], by_index.get(pair[0], 0)))
        if index is not None:
            ordered, _standing = pool_index.rank(
                query, [(ids.iloc[i], score) for i, score in scored], index.meta
            )
            places = {sid: i for i, sid in enumerate(ids)}
            scored = [(places[sid], score) for sid, score in ordered]
        return [
            self._hit(catalog_df.iloc[i], score, match="keyword")
            for i, score in scored[:k]
        ]

    def _check_embedding_cooldown(self) -> None:
        if (
            self._embed_down_until is not None
            and self._clock() < self._embed_down_until
        ):
            raise RuntimeError("embedding kisa sure once basarisiz oldu")

    def _semantic_scores(self, query: str, texts: list[str]) -> np.ndarray:
        """Cosine similarity of the query to every pool series' text.

        The document vectors are computed once per adapter: the catalog is
        read once too, so the two cannot drift apart. Raises while a recent
        failure is still cooling down, without calling the endpoint.
        """
        self._check_embedding_cooldown()
        try:
            if self._doc_vectors is None:
                self._doc_vectors = _unit_rows(self._embed_documents(texts))
            query_vector = _unit_rows(np.reshape(self._embed_query(query), (1, -1)))[0]
        except Exception:
            self._embed_down_until = self._clock() + EMBED_RETRY_AFTER_S
            raise
        self._embed_down_until = None
        return self._doc_vectors @ query_vector

    def _hit(
        self, row: pd.Series, score: float, *, match: str | None = None
    ) -> CatalogHit:
        """One pool catalog row as the contract's `CatalogHit`.

        Everything comes from `series_catalog`, so `kind`, `agg_method` and
        `cum_type` are the values the lake itself computes with, not guesses.
        `start_date`/`end_date` are the pool's coverage, not the catalog's:
        see `_coverage`.
        """
        first, last = self._coverage().get(row["series_id"], (None, None))
        return CatalogHit(
            series_id=row["series_id"],
            name_tr=_clean(row.get("name_tr")) or row["series_id"],
            name_en=_clean(row.get("name_en")),
            source=_clean(row.get("source")),
            freq=_clean(row.get("freq")),
            unit=_clean(row.get("unit")) or None,
            kind=_clean(row.get("kind")),
            agg_method=_clean(row.get("agg_method")),
            cum_type=_clean(row.get("cum_type")),
            start_date=first,
            end_date=last,
            score=round(score, 4),
            match=match,
        )

    def _coverage(self) -> dict[str, tuple[str, str]]:
        """First and last period holding a value, per pool series, 'YYYY-MM'.

        The catalog's own `start_date`/`end_date` are the source's publication
        range -- TP.KTF12 reads 2002 to 2026-09 there, while the pool holds
        2021 to 2026-06, and TP.HPBITABLO6.3 has no values before 2024-06. A
        candidate shown with the source's range invites promising data the
        lake does not have. Empty (not an error) if gold cannot be read, and
        then not cached: one failed read (a rebuild holding the file, say)
        would otherwise blank the coverage for the adapter's whole life.

        Read through `gold.coverage`, over the whole pool: the model-facing
        `run_sql` would cap it at a thousand series.
        """
        if self._coverage_cache is None:
            try:
                frame = gold.coverage()
            except Exception as exc:  # noqa: BLE001 - coverage is decoration
                logger.warning("search: kapsam okunamadi (%s)", exc)
                return {}
            self._coverage_cache = {
                row.series_id: (
                    pd.Timestamp(row.first).strftime("%Y-%m"),
                    pd.Timestamp(row.last).strftime("%Y-%m"),
                )
                for row in frame.itertuples(index=False)
            }
        return self._coverage_cache

    def catalog_lookup(self, series_ids: list[str]) -> pd.DataFrame:
        """The catalog rows for these ids, in the order asked for."""
        ids = list(series_ids)
        self._require_known(ids)
        catalog_df = self._series_catalog().set_index("series_id")
        return catalog_df.loc[ids].reset_index()

    # -- demo-time tables --------------------------------------------------

    def prepare_table(
        self, df: pd.DataFrame, hints: dict[str, Any] | None = None
    ) -> tuple[pd.DataFrame, QualityReport]:
        """Minimal clean-up for a table that arrived from outside the pool.

        Rows whose period cannot be parsed are dropped and the count goes on
        the quality card: `profile_table` runs on the cleaned frame, so without
        that line the card describes a table the caller never handed over, and
        the human approving a quality gate never learns anything went missing.
        """
        out = df.copy()
        dropped = 0
        if "period" in out.columns:
            out["period"] = pd.to_datetime(out["period"], errors="coerce")
            dropped = int(out["period"].isna().sum())
            out = out.dropna(subset=["period"]).sort_values("period")
        out = out.reset_index(drop=True)

        report = self.profile_table(out)
        if dropped:
            report = report.model_copy(
                update={
                    "issues": [
                        f"{dropped} satir tarihi okunamadigi icin dusuruldu.",
                        *report.issues,
                    ]
                }
            )
        return out, report

    def profile_table(self, df: pd.DataFrame) -> QualityReport:
        """The quality card, with `period` handed over the way the data lane
        expects it.

        `quality.profile_table` builds its expected calendar out of
        `datetime.date` objects, because gold's parquet DATE column reads back
        that way -- its own module docstring says so. Everything on the agent
        side carries `datetime64`, and `pd.Timestamp("2021-01-01")` does not
        equal `date(2021, 1, 1)`, so the observed set never intersected the
        expected one: a complete 12-month table came back as
        `missing_periods=12` with an empty `issues` list, i.e. a quality gate
        reporting total absence and flagging nothing. Converting here is the
        adapter's whole job -- bridging one lane's dtype to the other's.
        """
        return quality.profile_table(_periods_as_dates(df))


#: How coarse each frequency is; `to_freq` only ever moves up this scale.
_FREQ_RANK = {"D": 0, "W": 1, "M": 2, "Q": 3, "A": 4}


def _infer_freq(periods: pd.Series) -> str | None:
    """Guess a series' own frequency from the gaps between its periods.

    None when there is too little to tell, so the caller allows the conversion
    rather than refusing on a guess.
    """
    if len(periods) < 2:
        return None
    gap = pd.to_datetime(periods).sort_values().diff().dt.days.median()
    if pd.isna(gap):
        return None
    for name, (low, high) in {
        "D": (0, 2),
        "W": (6, 8),
        "M": (27, 32),
        "Q": (88, 93),
        "A": (360, 370),
    }.items():
        if low <= gap <= high:
            return name
    return None


def _periods_as_dates(df: pd.DataFrame) -> pd.DataFrame:
    """`period` as `datetime.date`, which is what `lake/quality.py` compares
    against. Returns the frame untouched when there is no period column."""
    if "period" not in df.columns:
        return df
    out = df.copy()
    out["period"] = pd.to_datetime(out["period"], errors="coerce").dt.date
    return out


def _tokens(text: str) -> list[str]:
    """Words, lower-cased with Turkish dotted/dotless I handled and then
    folded to ASCII, split on underscores too ("Türkiye_Konut_Toplam" is three
    words), short and meaningless words dropped."""
    folded = text.replace("I", "ı").replace("İ", "i").lower().translate(_FOLD)
    return [
        word
        for word in re.findall(r"[^\W_]+", folded)
        if len(word) >= 3 and word not in _STOPWORDS
    ]


def lexical_score(query: str, text: str) -> float:
    """Share of the query's words that appear in `text`.

    Matched on the first five letters, so Turkish suffixes do not decide it:
    "kredisi" and "kredileri" both reach "kredi". The fallback for when the
    embedding call is unavailable -- coarse on purpose, and reported as such.
    """
    words = _tokens(query)
    if not words:
        return 0.0
    candidates = _tokens(text)

    def found(word: str) -> bool:
        return any(
            other.startswith(word[:5]) or word.startswith(other[:5])
            for other in candidates
        )

    return sum(found(word) for word in words) / len(words)


#: A run of the characters a series id, code or column name is made of.
_CODE_RUN = re.compile(r"[0-9a-z_.:+-]+")


def named_series(query: str, series_ids: Any) -> list[int]:
    """Positions of series the query names outright: by id, bare code or the
    column name the tools give it (`column_name`).

    Whole tokens only, so "TP.KTF12 serisini getir" counts and "TP.KTF1" does
    not. A token ends at anything a code cannot hold, which takes care of
    quotes, brackets and a Turkish suffix after an apostrophe ("TP.KTF12'nin",
    "(TP.KTF12)"); a trailing full stop or colon is punctuation, not code.
    """
    tokens = {
        token.strip(".:-+")
        for token in _CODE_RUN.findall(query.casefold())
        if token.strip(".:-+")
    }
    matches = []
    for position, series_id in enumerate(series_ids):
        code = series_id.partition(":")[2]
        forms = {series_id, code, column_name(series_id)} - {""}
        if any(form.casefold() in tokens for form in forms):
            matches.append(position)
    return matches


def _unit_rows(vectors: Any) -> np.ndarray:
    """Rows scaled to length one, so a dot product is a cosine similarity.

    Refuses a missing or all-zero vector: a failed embedding comes back as
    NaN, and a NaN score would quietly sort to one end of the list.
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("gecersiz embedding vektoru (NaN veya sifir)")
    return matrix / norms


def _clean(value: Any) -> Any:
    """Parquet reads absent text as a missing value; the contract models want
    None. `pd.isna` covers every spelling pandas uses -- `float('nan')`,
    `pd.NA`, `pd.NaT` -- where an `isinstance(value, float)` check caught only
    the first and let the others through as the literal text "<NA>"/"NaT".
    """
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        # `pd.isna` on a list/array answers elementwise; such a value is
        # present by definition.
        return value
    return None if missing else value
