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
* **`search` returns `list[CatalogHit]`.** `catalog.search_catalog` hands back
  the raw bronze EVDS catalog columns plus a LanceDB table handle in its
  arguments; the agent should not know either exists.

`prepare_table` is implemented here too: nothing in the data lane offers it,
and the demo-time path (a table arriving from a URL) needs it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from lakehouse_agent.analysis import transforms
from lakehouse_agent.contracts import CatalogHit, QualityReport, SeriesNotFound
from lakehouse_agent.lake import gold, quality, silver

logger = logging.getLogger(__name__)

#: Where `build_index.py` writes the LanceDB catalog index.
CATALOG_INDEX_PATH = Path("data/catalog.lance")

#: `catalog.search_catalog` fuses its two rankings with reciprocal rank fusion
#: at k=60 but does not hand the fused value back -- the score columns that do
#: come back are per-backend and were measured as NaN. The order is the real
#: output, so the score is rebuilt from the rank with the same formula. What
#: matters downstream is that consecutive hits sit close together: the
#: ambiguity gate compares the top two as a *ratio*, never as an absolute gap.
RRF_K = 60

_BRONZE_TO_HIT = {
    "name_tr": "SERIE_NAME",
    "name_en": "SERIE_NAME_ENG",
    "source": "DATASOURCE",
    "unit": "BIRIMI",
    "start_date": "START_DATE",
    "end_date": "END_DATE",
}


class LakeIndexMissing(RuntimeError):
    """The catalog search index has not been built yet."""


class LakeAdapter:
    """The real data lane, presented as the contract's lake surface."""

    def __init__(
        self,
        *,
        index_path: Path = CATALOG_INDEX_PATH,
        catalog_path: Path | None = None,
    ) -> None:
        self._index_path = Path(index_path)
        self._catalog_path = Path(catalog_path or gold.SERIES_CATALOG_PATH)
        self._catalog: pd.DataFrame | None = None
        self._table: Any = None

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
        self, df: pd.DataFrame, kind: str, freq: str
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
        return transforms.to_freq(df, kind, freq)

    # -- search ------------------------------------------------------------

    def _open_index(self) -> Any:
        if self._table is None:
            if not self._index_path.exists():
                raise LakeIndexMissing(
                    f"Katalog arama indeksi yok: {self._index_path}. "
                    "Once `uv run python build_index.py` calistirin."
                )
            import lancedb

            self._table = lancedb.connect(self._index_path).open_table("catalog")
        return self._table

    def search(self, query: str, k: int = 20) -> list[CatalogHit]:
        """Catalog hits, narrowed to series the pool can actually serve.

        The index and the pool are not the same universe. `build_index.py`
        indexes the filtered bronze EVDS catalog (~19.6k series); gold holds
        the handful `build_gold` was run for. Returning an unfiltered hit
        hands the model a series id that the very next `get()` rejects with
        "not in the catalog -- search first", which is the advice it just
        followed: the loop does not close. Measured on a real index: the top
        hit for "konut kredisi faiz orani" was `TP.KBK.TRY.18`, which is not
        in gold.

        So hits outside the pool are dropped and counted. An empty result is
        the honest answer to "we have nothing for that" -- pulling a series
        that is not in the pool is the `ingest` tool's job, not search's.
        """
        from lakehouse_agent.lake import catalog

        hits = catalog.search_catalog(query, self._open_index(), top_k=k)
        known = set(self._series_catalog()["series_id"])

        kept: list[CatalogHit] = []
        for _, row in hits.iterrows():
            hit = self._to_hit(row, len(kept))
            if hit.series_id in known:
                kept.append(hit)

        dropped = len(hits) - len(kept)
        if dropped:
            logger.info(
                "search(%r): %d isabetin %d'i havuzda yok, elendi",
                query,
                len(hits),
                dropped,
            )
        return kept

    def _to_hit(self, row: pd.Series, rank: int) -> CatalogHit:
        """One bronze catalog row as the contract's `CatalogHit`.

        `kind`, `agg_method` and `cum_type` stay None on purpose: they are not
        EVDS catalog fields, they are decided per series in `gold.py`, and a
        guessed value here would be presented to the model as a fact.
        """
        values: dict[str, Any] = {
            field: _clean(row.get(source)) for field, source in _BRONZE_TO_HIT.items()
        }
        raw_freq = _clean(row.get("FREQUENCY_STR"))
        try:
            freq = silver.map_freq(raw_freq) if raw_freq else None
        except ValueError:
            logger.warning("search: bilinmeyen frekans %r, bos birakildi", raw_freq)
            freq = None
        return CatalogHit(
            series_id=f"evds:{row['SERIE_CODE']}",
            name_tr=values.pop("name_tr") or str(row["SERIE_CODE"]),
            freq=freq,
            score=1.0 / (RRF_K + rank + 1),
            **values,
        )

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
