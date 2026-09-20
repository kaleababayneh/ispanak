"""Search index over the EVDS bronze series catalog (plan 2.5).

Scope: `data/bronze/evds/catalog.parquet` has ~54k rows. Embedding all of
them (measured against real MIA calls, single worker) takes ~242 minutes --
unacceptable. `filter_catalog` narrows scope first, `embed_with_cache`
parallelizes (8 workers) and caches by text hash to disk so a re-run never
re-embeds a text it has already seen. `search_catalog` combines vector +
full-text search with reciprocal rank fusion, and falls back to full-text
only when no vector column is present.

`_embed_batch` below routes the actual MIA call through
`lakehouse_agent.llm.MiaLLM` -- the single door to MIA, so retry and usage
logging aren't bypassed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from lakehouse_agent.config import get_agent_settings
from lakehouse_agent.llm import MiaLLM

logger = logging.getLogger(__name__)

# Built lazily, not at import time: calling get_agent_settings() at module
# scope would fail test_no_module_reads_settings_at_import_time and force
# MIA_API_KEY on every environment that merely imports this module (tests
# pass their own `embed_fn` mock and never reach `_embed_batch`).
_mia_llm: MiaLLM | None = None


def _get_mia_llm() -> MiaLLM:
    global _mia_llm
    if _mia_llm is None:
        try:
            _mia_llm = MiaLLM(get_agent_settings())
        except ValidationError as exc:
            raise RuntimeError(
                "MIA_API_KEY ortam değişkeni ayarlı değil; embed edilemiyor."
            ) from exc
    return _mia_llm


# Qwen3-Embedding is ASYMMETRIC: a search query needs this instruction
# prefix, a document does not. Measured 17 Sep 2026 on the real 19646-row
# index, manual cosine similarity, query "konut kredisi faiz oranı":
#   prefixsiz -> "Market ve Alışveriş Merkezleri" 0.710, "Yemek" 0.685  [ÇÖP]
#   prefixli  -> "TP.KBK.TRY.18 Konut Kredisi" 0.649, "TP.KTF12" 0.641  [DOĞRU]
# Same pattern for "enflasyon": prefixsiz top hit "P3.Mevduat TL"; prefixli
# surfaces the actual enflasyon series. Without the prefix the query vector
# lands in a different region of the embedding space than the (unprefixed)
# document vectors, so similarity scores are close to noise.
#
# The 19646 indexed document vectors were built WITHOUT this prefix and
# stay that way -- re-embedding them would throw away the 17855-vector disk
# cache and cost real MIA calls for nothing. Only query-side embedding
# (`is_query=True`) gets the prefix; document embedding never does.
QUERY_INSTRUCTION_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)

CATALOG_EMB_CACHE_DIR = Path("data/catalog_emb_cache")
EMBEDDING_DIM = 4096  # Qwen3-Embedding-8B

MIN_END_DATE = pd.Timestamp("2025-06-01")
ALLOWED_FREQUENCIES = {
    "AYLIK",
    "HAFTALIK(CUMA)",
    "HAFTALIK(ÇARŞAMBA)",
    "ÜÇ AYLIK",
    "GÜNLÜK",
    "İŞ GÜNÜ",
}
EXCLUDED_DATAGROUP_CODES = {"bie_imfcpind", "bie_imfgdpusdn"}

BATCH_SIZE = 64
MAX_WORKERS = 8
PROGRESS_EVERY_N_BATCHES = 20
RRF_K = 60


def filter_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow the bronze EVDS catalog to series worth indexing for search.

    Drops: archived series, series dead before 2025-06, frequencies outside
    the scenario's scope, IMF cross-country datagroups (pollute search with
    near-duplicate country series), and province-level breakdowns of
    "konutsat" (housing sales) datagroups -- keep only the TOPLAM rows.
    """
    n_start = len(df)

    name = df["SERIE_NAME"].fillna("")
    dg_name = df["DATAGROUP_NAME"].fillna("")
    not_archived = ~(
        name.str.contains("Arşiv", case=False)
        | dg_name.str.contains("Arşiv", case=False)
    )

    end_date = pd.to_datetime(df["END_DATE"], format="%d-%m-%Y", errors="coerce")
    not_dead = end_date >= MIN_END_DATE

    allowed_freq = df["FREQUENCY_STR"].isin(ALLOWED_FREQUENCIES)

    not_imf = ~df["DATAGROUP_CODE"].isin(EXCLUDED_DATAGROUP_CODES)

    is_konutsat = df["DATAGROUP_CODE"].str.contains("konutsat", case=False, na=False)
    is_toplam = df["SERIE_CODE"].str.contains("TOPLAM", na=False)
    konutsat_ok = ~is_konutsat | is_toplam

    keep = not_archived & not_dead & allowed_freq & not_imf & konutsat_ok
    result = df[keep].reset_index(drop=True)

    logger.info(
        "filter_catalog: %d -> %d satır (%d filtrelendi)",
        n_start,
        len(result),
        n_start - len(result),
    )
    return result


def build_search_text(row: pd.Series) -> str:
    """Concatenate the fields worth matching a natural-language query against."""
    parts = [
        row.get("SERIE_NAME"),
        row.get("TOPIC_TITLE_TR"),
        row.get("TAG"),
        row.get("BIRIMI"),
        row.get("FREQUENCY_STR"),
        row.get("DATASOURCE"),
    ]
    return " ".join(
        str(p) for p in parts if p not in (None, "", "nan") and not pd.isna(p)
    )


def _embed_batch(texts: list[str], is_query: bool = False) -> np.ndarray:
    """MIA embedding call, routed through lakehouse_agent.llm.MiaLLM.
    Qwen3-Embedding-8B, dim 4096; measured 17.3s/batch of 64 on a single
    worker (hence the 8-worker parallelism in `embed_with_cache`).

    `is_query=True` prepends QUERY_INSTRUCTION_PREFIX to every text (see the
    measurement above the constant's definition) -- use it ONLY for the
    search query itself. Document/catalog-row embedding (the default,
    `is_query=False`) must never get this prefix: the indexed vectors were
    built without it, and this model's asymmetric scheme specifically wants
    an unprefixed corpus against a prefixed query, not both prefixed.
    `MiaLLM.embed()` has no notion of this prefix, so it is applied here,
    before the plain text is handed to it.
    """
    if is_query:
        texts = [QUERY_INSTRUCTION_PREFIX + text for text in texts]
    vectors = _get_mia_llm().embed(texts, tag="catalog.embed")
    return np.array(vectors, dtype=np.float32)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(text: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_text_hash(text)}.npy"


def embed_with_cache(
    texts: list[str],
    *,
    cache_dir: Path = CATALOG_EMB_CACHE_DIR,
    embed_fn=_embed_batch,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
) -> tuple[np.ndarray, list[int]]:
    """Embed `texts`, reusing cached vectors by text hash.

    Returns (vectors, failed_indices). A failed batch is skipped and logged;
    its rows come back as all-NaN in `vectors` and their positions are
    listed in `failed_indices`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    vectors: list[np.ndarray | None] = [None] * len(texts)
    to_embed: list[int] = []
    for i, text in enumerate(texts):
        path = _cache_path(text, cache_dir)
        if path.exists():
            vectors[i] = np.load(path)
        else:
            to_embed.append(i)

    logger.info(
        "embed_with_cache: %d/%d cache'ten okundu, %d embed edilecek",
        len(texts) - len(to_embed),
        len(texts),
        len(to_embed),
    )

    batches = [
        to_embed[i : i + batch_size] for i in range(0, len(to_embed), batch_size)
    ]
    failed_indices: list[int] = []
    start = time.monotonic()
    completed = 0

    def _run_batch(batch_indices: list[int]) -> tuple[list[int], np.ndarray | None]:
        batch_texts = [texts[i] for i in batch_indices]
        try:
            return batch_indices, embed_fn(batch_texts)
        except Exception:
            logger.exception("embed batch başarısız (%d metin)", len(batch_indices))
            return batch_indices, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch_indices, result in pool.map(_run_batch, batches):
            completed += 1
            if result is None:
                failed_indices.extend(batch_indices)
            else:
                for idx, vec in zip(batch_indices, result, strict=True):
                    vec = np.asarray(vec, dtype=np.float32)
                    vectors[idx] = vec
                    np.save(_cache_path(texts[idx], cache_dir), vec)

            if completed % PROGRESS_EVERY_N_BATCHES == 0 or completed == len(batches):
                elapsed = time.monotonic() - start
                remaining = (
                    (elapsed / completed) * (len(batches) - completed)
                    if completed
                    else 0.0
                )
                logger.info(
                    "embed_with_cache: batch %d/%d, geçen %.1fs, tahmini kalan %.1fs",
                    completed,
                    len(batches),
                    elapsed,
                    remaining,
                )

    if failed_indices:
        logger.warning("embed_with_cache: başarısız index'ler: %s", failed_indices)

    dim = next((v.shape[0] for v in vectors if v is not None), EMBEDDING_DIM)
    filled = np.stack(
        [
            v if v is not None else np.full(dim, np.nan, dtype=np.float32)
            for v in vectors
        ]
    )
    return filled, failed_indices


def build_catalog_index(
    df: pd.DataFrame,
    db_uri: str | Path,
    *,
    table_name: str = "catalog",
    cache_dir: Path = CATALOG_EMB_CACHE_DIR,
    embed_fn=_embed_batch,
    with_vectors: bool = True,
) -> lancedb.table.Table:  # noqa: F821 -- lazy import below
    """Filter, build search text, optionally embed, and write a LanceDB
    table with a full-text index on `text`.

    `db_uri` is a LanceDB **database directory** (created if missing), not a
    table path -- LanceDB itself names the table's on-disk dataset
    `<db_uri>/<table_name>.lance`. Passing something like ".../catalog.lance"
    as `db_uri` with the default `table_name` used to silently create
    ".../catalog.lance/catalog.lance" (`db_uri`'s parent as the DB, `.name`
    as the table) -- a caller reconnecting to the exact `db_uri` they passed
    in then saw an empty database. `db_uri` is now used as-is, so reconnecting
    to it is symmetric:

        >>> catalog.build_catalog_index(filtered_df, "data/catalog.lance")
        >>> lancedb.connect("data/catalog.lance").open_table("catalog").count_rows()
    """
    import lancedb
    from lancedb.index import FTS

    filtered = filter_catalog(df).copy()
    filtered["text"] = filtered.apply(build_search_text, axis=1)

    if with_vectors:
        vectors, failed = embed_with_cache(
            list(filtered["text"]), cache_dir=cache_dir, embed_fn=embed_fn
        )
        filtered["vector"] = list(vectors)
        if failed:
            logger.warning("build_catalog_index: %d satır vektörsüz kaldı", len(failed))

    db = lancedb.connect(str(db_uri))
    table = db.create_table(table_name, filtered, mode="overwrite")
    table.create_index("text", config=FTS())
    return table


def _reciprocal_rank_fusion(
    rankings: Iterable[list[str]], *, k: int = RRF_K
) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, series_code in enumerate(ranking):
            scores[series_code] = scores.get(series_code, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _exact_code_match(table: lancedb.table.Table, query: str) -> pd.DataFrame:  # noqa: F821
    """Direct, table-wide lookup for `query` as a literal SERIE_CODE.

    Deliberately independent of the FTS/vector top-k hit lists: a dotted
    code like "TP.KTF12" is not natural-language text, so nothing guarantees
    it ranks inside either search's top-k window on a 19646-row index (that
    was the bug -- checking only the already-fetched hits meant this path
    almost never fired). A `.where()` filter scans the whole table instead.
    """
    escaped = query.strip().replace("'", "''")
    return (
        table.search()
        .where(f"UPPER(SERIE_CODE) = UPPER('{escaped}')")
        .limit(1)
        .to_pandas()
    )


def search_catalog(
    query: str,
    table: lancedb.table.Table,  # noqa: F821
    *,
    top_k: int = 20,
    use_vector: bool = True,
    embed_fn=_embed_batch,
) -> pd.DataFrame:
    """Hybrid search: vector + full-text, combined by reciprocal rank fusion.

    Falls back to full-text-only when `use_vector=False` or the table has
    no `vector` column. A query starting with "TP." gets its exact
    SERIE_CODE match pinned to the top of the results, found by scanning the
    whole table (`_exact_code_match`), not just the top-k hit lists.

    The query vector is embedded with `is_query=True` (Qwen3-Embedding's
    asymmetric instruction prefix, see `QUERY_INSTRUCTION_PREFIX`) -- the
    indexed document vectors have no prefix, so this must not either.
    """
    has_vector_column = "vector" in table.schema.names

    exact_hit = None
    if query.strip().upper().startswith("TP."):
        hit = _exact_code_match(table, query)
        if not hit.empty:
            exact_hit = hit

    fts_hits = table.search(query, query_type="fts").limit(top_k * 2).to_pandas()
    fts_ranking = list(fts_hits["SERIE_CODE"])

    if use_vector and has_vector_column:
        query_vector = embed_fn([query], is_query=True)[0]
        vector_hits = table.search(query_vector).limit(top_k * 2).to_pandas()
        vector_ranking = list(vector_hits["SERIE_CODE"])
        fused_codes = _reciprocal_rank_fusion([vector_ranking, fts_ranking])
        all_hits = pd.concat([vector_hits, fts_hits]).drop_duplicates("SERIE_CODE")
    else:
        fused_codes = fts_ranking
        all_hits = fts_hits.drop_duplicates("SERIE_CODE")

    if exact_hit is not None:
        exact_code = exact_hit.iloc[0]["SERIE_CODE"]
        all_hits = pd.concat([exact_hit, all_hits]).drop_duplicates("SERIE_CODE")
        fused_codes = [exact_code] + [c for c in fused_codes if c != exact_code]

    ordered = fused_codes[:top_k]
    all_hits = all_hits.set_index("SERIE_CODE")
    return all_hits.loc[[c for c in ordered if c in all_hits.index]].reset_index()
