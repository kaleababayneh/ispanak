"""Tests for the adapter that presents the data lane as one lake surface.

The adapter is thin on purpose, so what is worth testing is exactly the part
that is *not* forwarding: the unknown-series guard, the catalog-hit mapping,
and the errors raised when the files those depend on have not been built.

No network and no LanceDB index: the catalog is a small parquet written into
`tmp_path`, and the one test that reaches for the index asserts the error you
get when it is missing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lakehouse_agent.contracts import CatalogHit, QualityReport, SeriesNotFound
from lakehouse_agent.lake import gold
from lakehouse_agent.tools.lake_adapter import LakeAdapter, LakeIndexMissing

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
    return LakeAdapter(index_path=tmp_path / "yok.lance", catalog_path=catalog_path)


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


# --- search ----------------------------------------------------------------


def test_search_without_an_index_says_which_command_builds_it(adapter):
    with pytest.raises(LakeIndexMissing, match="build_index.py"):
        adapter.search("konut kredisi faizi")


def test_a_bronze_row_becomes_a_contract_catalog_hit(adapter):
    row = pd.Series(
        {
            "SERIE_CODE": "TP.KTF12",
            "SERIE_NAME": "Konut Kredisi (TL, Akım)",
            "SERIE_NAME_ENG": "Housing Loan (TRY, Flow)",
            "FREQUENCY_STR": "HAFTALIK(CUMA)",
            "BIRIMI": "%",
            "DATASOURCE": "TCMB",
            "START_DATE": "01-01-2002",
            "END_DATE": "31-12-2026",
        }
    )

    hit = adapter._to_hit(row, rank=0)

    assert isinstance(hit, CatalogHit)
    assert hit.series_id == "evds:TP.KTF12", "series_id kaynak önekli olmalı"
    assert hit.freq == "W", "frekans sözleşme koduna çevrilmeli"
    assert hit.unit == "%"


def test_a_hit_leaves_kind_empty_rather_than_guessing_it(adapter):
    """`kind` is not an EVDS catalog field -- it is decided per series in
    gold.py. A guess here would be handed to the model as a fact, and `kind`
    is what decides whether a series is summed or averaged."""
    row = pd.Series({"SERIE_CODE": "TP.X", "SERIE_NAME": "X", "FREQUENCY_STR": "AYLIK"})

    hit = adapter._to_hit(row, rank=0)

    assert hit.kind is None
    assert hit.agg_method is None
    assert hit.cum_type is None


def test_an_unknown_frequency_leaves_freq_empty_instead_of_raising(adapter):
    """A catalog row the frequency map does not cover must not take down a
    whole search; the field is optional and the rest of the row is useful."""
    row = pd.Series(
        {"SERIE_CODE": "TP.X", "SERIE_NAME": "X", "FREQUENCY_STR": "SAATLIK"}
    )

    assert adapter._to_hit(row, rank=0).freq is None


def test_scores_fall_with_rank_and_stay_close_together(adapter):
    """The ambiguity gate compares the top two as a ratio, never as a gap, so
    what matters is that neighbouring hits sit close -- the same shape the
    real reciprocal-rank fusion produces."""
    row = pd.Series({"SERIE_CODE": "TP.X", "SERIE_NAME": "X", "FREQUENCY_STR": "AYLIK"})

    first = adapter._to_hit(row, rank=0).score
    second = adapter._to_hit(row, rank=1).score

    assert first > second > 0
    assert second / first > 0.9


@pytest.mark.parametrize(
    "missing",
    [float("nan"), pd.NA, pd.NaT],
    ids=["float nan", "pd.NA", "pd.NaT"],
)
def test_every_pandas_spelling_of_missing_becomes_none(adapter, missing):
    """pandas has three. An `isinstance(value, float)` check caught only the
    first and let the others through, so the hit carried the literal text
    "<NA>" or "NaT" into whatever rendered it."""
    row = pd.Series(
        {
            "SERIE_CODE": "TP.X",
            "SERIE_NAME": "X",
            "SERIE_NAME_ENG": missing,
            "FREQUENCY_STR": "AYLIK",
        }
    )

    assert adapter._to_hit(row, rank=0).name_en is None


def test_missing_text_fields_become_none_not_nan(adapter):
    """Parquet reads an absent string as NaN, and `CatalogHit` wants None --
    a NaN would render as the literal "nan" wherever the hit is shown."""
    row = pd.Series(
        {
            "SERIE_CODE": "TP.X",
            "SERIE_NAME": "X",
            "SERIE_NAME_ENG": float("nan"),
            "FREQUENCY_STR": "AYLIK",
        }
    )

    assert adapter._to_hit(row, rank=0).name_en is None


# --- search is narrowed to what the pool can serve --------------------------


def _bronze_row(code: str) -> dict:
    return {
        "SERIE_CODE": code,
        "SERIE_NAME": code,
        "FREQUENCY_STR": "AYLIK",
    }


def test_search_drops_hits_the_pool_cannot_serve(adapter, monkeypatch):
    """The index holds the filtered bronze catalog (~19.6k series); gold holds
    the handful `build_gold` was run for. An unfiltered hit hands the model a
    series id that the very next `get()` rejects with "search first" -- the
    advice it just followed. Measured on a real index: the top hit for
    "konut kredisi faiz orani" was TP.KBK.TRY.18, which is not in gold."""
    from lakehouse_agent.lake import catalog

    monkeypatch.setattr(adapter, "_open_index", lambda: object())
    monkeypatch.setattr(
        catalog,
        "search_catalog",
        lambda *a, **k: pd.DataFrame(
            [_bronze_row("TP.KBK.TRY.18"), _bronze_row("TP.KTF12")]
        ),
    )

    hits = adapter.search("konut kredisi faiz orani")

    assert [h.series_id for h in hits] == [KNOWN]


def test_a_search_with_nothing_in_the_pool_is_empty_not_wrong(adapter, monkeypatch):
    """Empty is the honest answer to "we have nothing for that". Pulling a
    series the pool does not hold is the `ingest` tool's job, not search's."""
    from lakehouse_agent.lake import catalog

    monkeypatch.setattr(adapter, "_open_index", lambda: object())
    monkeypatch.setattr(
        catalog, "search_catalog", lambda *a, **k: pd.DataFrame([_bronze_row("TP.YOK")])
    )

    assert adapter.search("havuzda olmayan bir sey") == []


def test_surviving_hits_are_ranked_from_zero(adapter, monkeypatch):
    """Scores come from position, so dropping a hit must not leave a gap --
    otherwise the top hit's score depends on how many unusable series happened
    to rank above it."""
    from lakehouse_agent.lake import catalog

    monkeypatch.setattr(adapter, "_open_index", lambda: object())
    monkeypatch.setattr(
        catalog,
        "search_catalog",
        lambda *a, **k: pd.DataFrame(
            [
                _bronze_row("TP.YOK.BIR"),
                _bronze_row("TP.YOK.IKI"),
                _bronze_row("TP.KTF12"),
            ]
        ),
    )

    hits = adapter.search("konut")

    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0 / 61)


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
