"""Tests for the adapter that presents the data lane as one lake surface.

The adapter is thin on purpose, so what is worth testing is exactly the part
that is *not* forwarding: the unknown-series guard, the catalog-hit mapping,
and the errors raised when the files those depend on have not been built.

No network: the catalog is a small parquet written into `tmp_path`, and
search gets two-dimensional stand-in embeddings instead of MIA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lakehouse_agent.contracts import CatalogHit, QualityReport, SeriesNotFound
from lakehouse_agent.lake import gold
from lakehouse_agent.tools.lake_adapter import (
    EMBED_RETRY_AFTER_S,
    LakeAdapter,
    lexical_score,
    named_series,
    search_text,
)

KNOWN = "evds:TP.KTF12"
OTHER = "evds:TP.KFE.TR"


@pytest.fixture
def catalog_path(tmp_path):
    rows = []
    for series_id, name, kind in (
        (KNOWN, "Konut Kredisi Faizi", "rate"),
        (OTHER, "Konut Fiyat Endeksi", "index"),
    ):
        row = dict.fromkeys(gold.SERIES_CATALOG_COLUMNS)
        row.update(
            series_id=series_id, name_tr=name, kind=kind, freq="M", source="EVDS"
        )
        rows.append(row)
    path = tmp_path / "series_catalog.parquet"
    pd.DataFrame(rows, columns=gold.SERIES_CATALOG_COLUMNS).to_parquet(path)
    return path


@pytest.fixture
def adapter(catalog_path, tmp_path):
    return LakeAdapter(catalog_path=catalog_path)


@pytest.fixture
def monthly_frame():
    return pd.DataFrame(
        {
            "period": pd.date_range("2021-01-01", periods=12, freq="MS"),
            "value": [float(i) for i in range(1, 13)],
        }
    )


# --- the unknown-series guard ----------------------------------------------


def test_get_rejects_a_series_that_is_not_in_the_catalog(adapter):
    """`gold.get` answers an unknown id with an empty frame, which reaches the
    model as "no data for this period" when the real problem is a typo. The
    exception is what lets the loop say "search the catalog first" instead."""
    with pytest.raises(SeriesNotFound):
        adapter.get(["evds:BOYLE.BIR.SERI.YOK"], "2021-01", "2021-06")


def test_the_error_names_every_unknown_id_at_once(adapter):
    """One call with three typos should report three. Reporting the first
    sends the model round the loop once per id."""
    with pytest.raises(SeriesNotFound) as caught:
        adapter.get([KNOWN, "evds:YOK.BIR", "evds:YOK.IKI"], "2021-01", "2021-06")

    message = str(caught.value)
    assert "evds:YOK.BIR" in message
    assert "evds:YOK.IKI" in message
    assert KNOWN not in message


def test_wide_applies_the_same_guard(adapter):
    with pytest.raises(SeriesNotFound):
        adapter.wide(["evds:YOK"], "2021-01", "2021-06")


@pytest.mark.parametrize(
    ("ids", "match"),
    [([], "en az bir series_id"), ([KNOWN, KNOWN], "tekrar var")],
    ids=["bos liste", "tekrarli id"],
)
def test_wide_refuses_an_id_list_it_cannot_pivot(adapter, ids, match):
    """`FakeLake.wide` refuses both, so the adapter has to as well -- otherwise
    the same call raises a clear ValueError against the fake and an IOException
    from DuckDB (empty) or produces `<id>_x`/`<id>_y` columns (duplicates)
    against the real lane."""
    with pytest.raises(ValueError, match=match):
        adapter.wide(ids, "2021-01", "2021-06")


def test_the_pivot_guard_runs_before_the_catalog_is_read(adapter, tmp_path):
    """An empty list is malformed whether or not a catalog exists, so it must
    not depend on one being built first."""
    no_catalog = LakeAdapter(catalog_path=tmp_path / "yok.parquet")
    with pytest.raises(ValueError, match="en az bir series_id"):
        no_catalog.wide([], "2021-01", "2021-06")


def test_catalog_lookup_applies_the_same_guard(adapter):
    with pytest.raises(SeriesNotFound):
        adapter.catalog_lookup(["evds:YOK"])


def test_the_guard_runs_before_the_data_is_touched(adapter, monkeypatch):
    """Validation first: a query built from a typo should never reach DuckDB,
    so an unbuilt gold layer cannot turn a clear error into a confusing one."""
    called = False

    def _boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("gold.get cagrilmamaliydi")

    monkeypatch.setattr(gold, "get", _boom)
    with pytest.raises(SeriesNotFound):
        adapter.get(["evds:YOK"], "2021-01", "2021-06")
    assert not called


def test_get_forwards_the_window_untouched(adapter, monkeypatch):
    """The adapter validates and forwards; it must not quietly reinterpret the
    window, or the caller's "2021-01" would mean one thing to the agent and
    another to the lake."""
    seen = {}

    def spy(series_ids, start, end):
        seen.update(series_ids=list(series_ids), start=start, end=end)
        return pd.DataFrame()

    monkeypatch.setattr(gold, "get", spy)
    adapter.get([KNOWN], "2021-01", "2021-06")

    assert seen == {"series_ids": [KNOWN], "start": "2021-01", "end": "2021-06"}


def test_wide_forwards_the_frequency_it_was_given(adapter, monkeypatch):
    """`freq` is what makes a weekly and a monthly series line up at all, so a
    dropped argument turns a joint read into a mostly-empty table."""
    seen = {}

    def spy(series_ids, start, end, freq):
        seen.update(freq=freq)
        return pd.DataFrame()

    monkeypatch.setattr(gold, "wide", spy)
    adapter.wide([KNOWN], "2021-01", "2021-06", "W")

    assert seen["freq"] == "W"


# --- catalog_lookup --------------------------------------------------------


def test_catalog_lookup_returns_rows_in_the_order_asked_for(adapter):
    result = adapter.catalog_lookup([OTHER, KNOWN])
    assert result["series_id"].tolist() == [OTHER, KNOWN]


def test_catalog_lookup_returns_the_full_catalog_schema(adapter):
    result = adapter.catalog_lookup([KNOWN])
    assert list(result.columns) == gold.SERIES_CATALOG_COLUMNS


# --- catalog reading -------------------------------------------------------


def test_a_missing_catalog_says_which_file_and_what_to_run(tmp_path):
    adapter = LakeAdapter(catalog_path=tmp_path / "yok.parquet")
    with pytest.raises(FileNotFoundError, match="series_catalog"):
        adapter.catalog_lookup([KNOWN])


def test_the_catalog_is_read_once_not_per_call(adapter, catalog_path, monkeypatch):
    """Every id check would otherwise re-read the file, and the loop checks on
    every tool call."""
    reads = 0
    real_read = pd.read_parquet

    def counting_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counting_read)
    adapter.catalog_lookup([KNOWN])
    adapter.catalog_lookup([OTHER])
    assert reads == 1


# --- search: over the pool's own catalog ----------------------------------

# Two-dimensional stand-ins for embeddings: "faiz" points one way, "fiyat" the
# other. Enough to check ranking, scores and filtering without a network.
_FAIZ, _FIYAT = [1.0, 0.0], [0.0, 1.0]


def _fake_documents(texts):
    return np.array([_FAIZ if "Faiz" in text else _FIYAT for text in texts])


def _fake_query(query):
    if "faiz" in query:
        return np.array([1.0, 0.6])
    if "fiyat" in query:
        return np.array([0.6, 1.0])
    return np.array([-1.0, -1.0])  # unrelated: points away from both


def _unavailable(*_args):
    raise RuntimeError("MIA_API_KEY ortam degiskeni ayarli degil")


@pytest.fixture
def searchable(catalog_path):
    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=_fake_documents,
        embed_query=_fake_query,
    )
    adapter._coverage = lambda: {KNOWN: ("2021-01", "2026-06")}
    return adapter


def test_search_ranks_the_pool_by_similarity_to_the_query(searchable):
    hits = searchable.search("konut kredisi faiz orani")

    assert [hit.series_id for hit in hits] == [KNOWN, OTHER]
    # Cosine of (1, 0.6) with (1, 0) and with (0, 1): 0.857 and 0.514.
    assert hits[0].score == pytest.approx(1 / np.hypot(1, 0.6), abs=1e-4)
    assert hits[1].score == pytest.approx(0.6 / np.hypot(1, 0.6), abs=1e-4)


def test_a_candidate_below_the_similarity_floor_is_not_offered(searchable):
    """(1, 0.2) is 0.196 from the price index: not a candidate, not a guess."""
    searchable._embed_query = lambda _query: np.array([1.0, 0.2])

    assert [hit.series_id for hit in searchable.search("faiz")] == [KNOWN]


def test_a_hit_carries_the_lakes_own_kind_frequency_and_source(searchable):
    """Everything comes from series_catalog -- the values the lake computes
    with -- so kind is known, not left empty as the EVDS index had to."""
    hit = searchable.search("faiz")[0]

    assert isinstance(hit, CatalogHit)
    assert (hit.kind, hit.freq, hit.source) == ("rate", "M", "EVDS")


def test_a_hit_carries_the_pools_coverage_not_the_sources(searchable):
    """The catalog's dates are the source's publication range; TP.KTF12 reads
    2002 to 2026-09 there. What a candidate must show is what the lake holds."""
    hit = searchable.search("faiz")[0]

    assert (hit.start_date, hit.end_date) == ("2021-01", "2026-06")


@pytest.mark.parametrize("query", ["TP.KTF12 serisini getir", "evds:TP.KTF12"])
def test_a_query_naming_a_series_returns_that_series_alone(catalog_path, query):
    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=_unavailable,
        embed_query=_unavailable,
    )
    adapter._coverage = dict

    hits = adapter.search(query)

    assert [(hit.series_id, hit.score) for hit in hits] == [(KNOWN, 1.0)]


def test_a_query_about_something_the_pool_lacks_finds_nothing(searchable):
    """Empty is the honest answer. A far-fetched candidate is worse: the model
    will happily build an analysis on it."""
    assert searchable.search("dolar kuru") == []


def test_without_embeddings_search_falls_back_to_keywords_and_says_so(catalog_path):
    """No key or MIA down must not take search with it: the pool is a few
    rows of text, and a coarser ranking beats none."""
    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=_fake_documents,
        embed_query=_unavailable,
    )
    adapter._coverage = dict

    hits, notes = adapter.search_detailed("konut kredisi faizi")

    assert hits[0].series_id == KNOWN
    assert notes and "kelime" in notes[0]


def test_a_broken_document_vector_also_falls_back(catalog_path):
    """A failed embedding comes back as NaN; a NaN score would sort quietly."""
    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=lambda texts: np.full((len(texts), 2), np.nan),
        embed_query=_fake_query,
    )
    adapter._coverage = dict

    _hits, notes = adapter.search_detailed("faiz")

    assert notes


def test_the_pool_is_embedded_once_not_per_search(catalog_path):
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return _fake_documents(texts)

    adapter = LakeAdapter(
        catalog_path=catalog_path, embed_documents=counting, embed_query=_fake_query
    )
    adapter._coverage = dict

    adapter.search("faiz")
    adapter.search("fiyat")

    assert calls == [2]


def test_search_text_says_what_kind_of_quantity_a_series_is():
    """A BDDK row has no topic or tags; its kind is what lets "konut kredisi
    bakiyesi" reach it. Absent fields are skipped, never written as "nan"."""
    text = search_text(
        pd.Series(
            {
                "name_tr": "Tüketici Kredileri - Konut",
                "topic": float("nan"),
                "tags": None,
                "unit": "milyon TL",
                "freq": "M",
                "kind": "stock",
                "source": "BDDK-Aylik",
            }
        )
    )

    assert text == (
        "Tüketici Kredileri - Konut | milyon TL | aylık | stok (bakiye) "
        "| BDDK Aylık Bülten"
    )


def test_keyword_matching_survives_turkish_suffixes():
    assert lexical_score("konut kredisi", "Tüketici Kredileri - Konut") == 1.0
    assert lexical_score("dolar kuru", "Tüketici Kredileri - Konut") == 0.0


@pytest.mark.parametrize(
    "missing",
    [float("nan"), pd.NA, pd.NaT],
    ids=["float nan", "pd.NA", "pd.NaT"],
)
def test_every_pandas_spelling_of_missing_becomes_none(searchable, missing):
    """pandas has three. An `isinstance(value, float)` check caught only the
    first and let the others through, so the hit carried the literal text
    "<NA>" or "NaT" into whatever rendered it."""
    row = pd.Series({"series_id": KNOWN, "name_tr": "X", "name_en": missing})

    assert searchable._hit(row, 0.5).name_en is None


def test_an_empty_unit_is_no_unit(searchable):
    """Gold stores an index series' unit as the empty string."""
    row = pd.Series({"series_id": OTHER, "name_tr": "KFE", "unit": ""})

    assert searchable._hit(row, 0.5).unit is None


# --- transforms and quality forwarding -------------------------------------


def test_a_complete_table_reports_no_missing_periods(adapter):
    """`quality.profile_table` builds its expected calendar from
    `datetime.date`; everything agent-side carries `datetime64`, and the two
    do not compare equal. Before the adapter converted, a complete 12-month
    table came back as `missing_periods=12` with an empty `issues` list -- a
    quality gate reporting total absence and flagging nothing."""
    complete = pd.DataFrame(
        {
            "period": pd.date_range("2021-01-01", periods=12, freq="MS"),
            "value": [float(i) for i in range(12)],
        }
    )

    assert adapter.profile_table(complete).missing_periods == 0


def test_a_real_gap_is_still_counted(adapter):
    """The conversion must not make the card blind in the other direction."""
    gapped = pd.DataFrame(
        {
            "period": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-05-01"]),
            "value": [1.0, 2.0, 3.0],
        }
    )

    assert adapter.profile_table(gapped).missing_periods == 2


def test_to_freq_refuses_to_go_up_in_frequency(adapter, monthly_frame):
    """CLAUDE.md #4 and the contract both say high-to-low only, and
    `FakeLake.to_freq` enforces it -- `analysis.transforms.to_freq` does not,
    so the guard lives in the adapter until it moves into the data lane."""
    with pytest.raises(ValueError, match="Yuksek frekansa"):
        adapter.to_freq(monthly_frame, "index", "W")


def test_to_freq_still_aggregates_downwards(adapter):
    weekly = pd.DataFrame(
        {
            "period": pd.date_range("2021-01-04", periods=20, freq="W-MON"),
            "value": [10.0] * 20,
        }
    )

    frame, _ = adapter.to_freq(weekly, "rate", "M")

    assert len(frame) == 5


def test_to_freq_hands_back_the_frame_and_the_formula(adapter, monthly_frame):
    frame, formula = adapter.to_freq(monthly_frame, "rate", "M")
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(formula, str) and formula


def test_profile_table_returns_the_contract_type(adapter, monthly_frame):
    assert isinstance(adapter.profile_table(monthly_frame), QualityReport)


# --- prepare_table ---------------------------------------------------------


def test_prepare_table_sorts_and_drops_unparseable_dates(adapter):
    messy = pd.DataFrame(
        {"period": ["2021-03-01", "bozuk", "2021-01-01"], "value": [3.0, 2.0, 1.0]}
    )

    cleaned, _ = adapter.prepare_table(messy)

    assert cleaned["value"].tolist() == [1.0, 3.0]


def test_prepare_table_reports_how_many_rows_it_dropped(adapter):
    """`profile_table` runs on the cleaned frame, so the drop is invisible to
    it. Without this line the card describes a table the caller never handed
    over, and the quality gate a human approves is missing the one fact that
    mattered."""
    messy = pd.DataFrame(
        {"period": ["2021-01-01", "bozuk", "yine bozuk"], "value": [1.0, 2.0, 3.0]}
    )

    _, report = adapter.prepare_table(messy)

    assert any("2 satir" in issue for issue in report.issues), report.issues


def test_prepare_table_stays_quiet_when_nothing_was_dropped(adapter):
    clean = pd.DataFrame({"period": ["2021-01-01", "2021-02-01"], "value": [1.0, 2.0]})

    _, report = adapter.prepare_table(clean)

    assert not any("dusuruldu" in issue for issue in report.issues)


@pytest.mark.parametrize(
    "query",
    [
        "TP.KTF12'nin son degeri",
        "TP.KTF12’nin son degeri",
        "(TP.KTF12) nedir?",
        '"TP.KTF12" serisi',
        "TP.KTF12.",
        "tp_ktf12 sutunu",
    ],
)
def test_a_code_with_punctuation_or_in_column_form_is_still_named(query):
    """Split on spaces only, "TP.KTF12'nin" and the tools' own column name
    missed the exact path, and the keyword fallback cannot find a code."""
    assert named_series(query, [OTHER, KNOWN]) == [1]


def test_a_bddk_column_name_names_its_series():
    bddk = "bddk:aylik:tuketici:konut:TL:sektor"
    assert named_series("aylik_tuketici_konut_TL_sektor", [bddk]) == [0]


def test_part_of_a_code_names_nothing():
    assert named_series("TP.KTF1 ve konut", [KNOWN]) == []


def test_a_named_hit_says_how_it_was_found(catalog_path):
    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=_unavailable,
        embed_query=_unavailable,
    )
    adapter._coverage = dict

    hits = adapter.search("evds:TP.KTF12 ve TP.KFE.TR")

    assert {(hit.series_id, hit.match) for hit in hits} == {
        (KNOWN, "code"),
        (OTHER, "code"),
    }


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("tufe", "Tüketici Fiyat Endeksi | TÜFE"),
        ("konut satis", "Türkiye_Konut_Toplam Satışlar"),
        ("toplam", "Türkiye_Konut_Toplam Satışlar"),
        ("KONUT FIYAT ENDEKSI", "Konut Fiyat Endeksi"),
    ],
)
def test_keyword_matching_folds_turkish_letters_and_splits_underscores(query, text):
    """An ASCII query is the ordinary case, and "Türkiye_Konut_Toplam" was
    one word, so "toplam" could never tell the two sales series apart."""
    assert lexical_score(query, text) == 1.0


def test_a_failed_embedding_is_not_retried_on_every_search(catalog_path):
    """A hung endpoint costs a full client timeout per call; re-paying it on
    every search spent most of a turn before each fallback."""
    now = [0.0]
    calls = []

    def failing(query):
        calls.append(query)
        raise TimeoutError("embedding zaman asimi")

    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=_fake_documents,
        embed_query=failing,
        clock=lambda: now[0],
    )
    adapter._coverage = dict

    adapter.search("faiz")
    adapter.search("fiyat")
    assert calls == ["faiz"]

    now[0] += EMBED_RETRY_AFTER_S
    adapter.search("faiz")
    assert calls == ["faiz", "faiz"]


def test_a_failed_coverage_read_is_tried_again(adapter, monkeypatch):
    """Cached as empty, one failed read blanked 'kapsam' for the adapter's
    whole life -- and with it planner rule 12's coverage check."""
    answers = [
        RuntimeError("kilitli"),
        pd.DataFrame(
            {
                "series_id": [KNOWN],
                "first": [pd.Timestamp("2021-01-01")],
                "last": [pd.Timestamp("2026-06-01")],
            }
        ),
    ]

    def coverage():
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(gold, "coverage", coverage)

    assert adapter._coverage() == {}
    assert adapter._coverage() == {KNOWN: ("2021-01", "2026-06")}


# --- the pool index (large pools) -------------------------------------------


def _documents_with_failures(texts):
    return _fake_documents(texts), []


def test_a_large_pool_searches_its_built_index(catalog_path, tmp_path):
    """Past IN_MEMORY_LIMIT the catalog is not embedded inside the search:
    the index built after gold answers, with the same cosine scores."""
    from lakehouse_agent.lake import pool_index

    index_dir = tmp_path / "index"
    pool_index.build(
        pd.read_parquet(catalog_path),
        path=index_dir,
        embed_documents=_documents_with_failures,
    )

    def no_catalog_embedding(_texts):
        raise AssertionError("buyuk havuzda katalog aramada embed edilmemeli")

    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_documents=no_catalog_embedding,
        embed_query=_fake_query,
        index_path=index_dir,
        in_memory_limit=0,
    )
    adapter._coverage = dict

    hits, notes = adapter.search_detailed("faiz orani")

    assert [hit.series_id for hit in hits] == [KNOWN, OTHER]
    assert hits[0].score == pytest.approx(1 / np.hypot(1, 0.6), abs=1e-4)
    assert hits[0].match == "semantic"
    assert notes == []


def test_a_large_pool_without_an_index_says_it_searched_by_keyword(
    catalog_path, tmp_path
):
    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_query=_fake_query,
        index_path=tmp_path / "yok",
        in_memory_limit=0,
    )
    adapter._coverage = dict

    hits, notes = adapter.search_detailed("konut kredisi faiz")

    assert hits and hits[0].match == "keyword"
    assert notes and "indeksi kurulmamis" in notes[0]


def test_an_index_built_from_another_catalog_says_what_it_misses(
    catalog_path, tmp_path
):
    from lakehouse_agent.lake import pool_index

    index_dir = tmp_path / "index"
    old = pd.read_parquet(catalog_path).iloc[:1]  # KNOWN only
    pool_index.build(old, path=index_dir, embed_documents=_documents_with_failures)

    adapter = LakeAdapter(
        catalog_path=catalog_path,
        embed_query=_fake_query,
        index_path=index_dir,
        in_memory_limit=0,
    )
    adapter._coverage = dict

    _hits, notes = adapter.search_detailed("faiz")

    assert any("1 havuz serisi indekste yok" in note for note in notes)
