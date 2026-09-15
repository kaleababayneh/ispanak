"""Tests for lakehouse_agent.contracts.

These lock the cross-lane data contract. A failure here means the written
interface in docs/contracts.md and the code have drifted apart.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lakehouse_agent.contracts import (
    Artifact,
    CatalogHit,
    ColumnInfo,
    QualityReport,
    SeriesNotFound,
    Source,
    ToolResult,
    Usage,
)

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


# --- table_ref: the hallucination shield ---------------------------------


def test_table_ref_accepts_generated_refs():
    for ref in ("t01_get_series", "t02_deflate", "t99_read_url", "t04_a"):
        assert ToolResult(ok=True, summary="x", table_ref=ref).table_ref == ref


@pytest.mark.parametrize(
    "invented",
    [
        "konut_tablosu",  # a plausible name the model might make up
        "active",  # the magic value we deliberately do not support
        "T01_x",  # wrong case
        "t1_x",  # one digit instead of two
        "t01_Get",  # uppercase in the suffix
        "t01_",  # empty suffix
        "t01",  # no suffix at all
        " t01_x",  # stray whitespace
    ],
)
def test_table_ref_rejects_invented_names(invented):
    with pytest.raises(ValidationError):
        ToolResult(ok=True, summary="x", table_ref=invented)


def test_table_ref_may_be_omitted():
    """A tool that returns no table (e.g. web_search) leaves it None."""
    assert ToolResult(ok=True, summary="x").table_ref is None


# --- Usage: summable ------------------------------------------------------


def test_usage_add_sums_all_four_counters():
    a = Usage(prompt_tokens=10, completion_tokens=5, calls=1, duration_ms=100)
    b = Usage(prompt_tokens=3, completion_tokens=2, calls=1, duration_ms=50)
    total = a + b
    assert total.prompt_tokens == 13
    assert total.completion_tokens == 7
    assert total.calls == 2
    assert total.duration_ms == 150
    assert total.total_tokens == 20


def test_usage_add_is_associative_and_leaves_operands_untouched():
    a, b, c = (Usage(prompt_tokens=n, calls=1) for n in (1, 2, 3))
    assert ((a + b) + c).prompt_tokens == (a + (b + c)).prompt_tokens == 6
    assert a.prompt_tokens == 1  # not mutated in place


def test_usage_add_rejects_other_types():
    assert Usage().__add__(3) is NotImplemented


# --- Source: identity and dedupe -----------------------------------------


def test_source_requires_series_id_or_url():
    with pytest.raises(ValidationError):
        Source(fetched_at=NOW)


def test_source_key_prefers_series_id_then_url():
    assert Source(series_id="evds:TP.KTF12", fetched_at=NOW).key() == "evds:TP.KTF12"
    assert Source(url="https://x/y.pdf", fetched_at=NOW).key() == "https://x/y.pdf"


def test_sources_with_the_same_key_dedupe_to_one():
    """The session keeps one entry per underlying source; this is the key it uses."""
    a = Source(series_id="evds:TP.KTF12", fetched_at=NOW, note="ilk cekim")
    b = Source(series_id="evds:TP.KTF12", fetched_at=NOW, note="ikinci cekim")
    assert len({a.key(), b.key()}) == 1


# --- ColumnInfo: provenance ----------------------------------------------


def test_column_info_source_series_is_a_list_for_derived_columns():
    """A deflated column has two source series; a single field would drop one."""
    col = ColumnInfo(
        name="reel_konut_kredisi",
        op="deflate",
        formula="reel = nominal x TUFE_baz / TUFE_t",
        inputs=["konut_kredisi"],
        source_series=[
            "bddk:aylik:tuketici:konut:TL:sektor",
            "evds:TP.TUKFIY2025.GENEL",
        ],
    )
    assert len(col.source_series) == 2
    assert col.inputs == ["konut_kredisi"]


def test_column_info_defaults_are_empty_lists_not_none():
    col = ColumnInfo(name="x")
    assert col.source_series == []
    assert col.inputs == []


# --- QualityReport / CatalogHit / Artifact --------------------------------


def test_quality_report_defaults_are_clean():
    q = QualityReport(rows=66, date_range=("2021-01", "2026-06"), freq="M")
    assert q.missing_periods == 0
    assert q.duplicates == 0
    assert q.issues == []
    assert q.unit_guess is None  # unknown unit is itself a quality flag


def test_catalog_hit_score_is_unconstrained_float():
    """Hybrid search returns RRF scores (~1/(60+rank)), not a 0..1 similarity."""
    hit = CatalogHit(
        series_id="evds:TP.KTF12", name_tr="Konut (TL, akim)", score=0.0163
    )
    assert 0 < hit.score < 1
    assert CatalogHit(series_id="x", name_tr="y", score=42.0).score == 42.0


def test_artifact_type_is_restricted():
    assert Artifact(type="chart", ref="c01").ref == "c01"
    with pytest.raises(ValidationError):
        Artifact(type="spreadsheet", ref="c01")


# --- SeriesNotFound -------------------------------------------------------


def test_series_not_found_message_lists_the_missing_ids():
    err = SeriesNotFound(["evds:YOK.1", "evds:YOK.2"], "Once search_catalog kullanin.")
    assert "evds:YOK.1" in str(err)
    assert "evds:YOK.2" in str(err)
    assert "search_catalog" in str(err)
    assert err.missing == ["evds:YOK.1", "evds:YOK.2"]


def test_series_not_found_is_a_lookup_error():
    """Callers may catch LookupError without importing our class."""
    with pytest.raises(LookupError):
        raise SeriesNotFound(["x"])


# --- JSON serialisability (the event payload rule) ------------------------


def test_tool_result_round_trips_through_json():
    """Event payloads must be JSON-serialisable: no DataFrames, no raw datetimes."""
    result = ToolResult(
        ok=True,
        summary="60 satir, 2021-01 - 2025-12.",
        table_ref="t01_get_series",
        preview=[{"period": "2021-01-01", "konut_kredisi": 1234.5}],
        columns=[ColumnInfo(name="konut_kredisi", unit="milyon TL", kind="stock")],
        sources=[Source(series_id="evds:TP.KTF12", fetched_at=NOW)],
        warnings=["Haftalik seri aylik ortalamaya cevrildi."],
        artifacts=[Artifact(type="chart", ref="c01")],
        quality=QualityReport(rows=60, freq="M"),
    )
    payload = result.model_dump(mode="json")
    restored = ToolResult.model_validate(json.loads(json.dumps(payload)))
    assert restored.table_ref == "t01_get_series"
    assert restored.sources[0].fetched_at == NOW
    assert restored.quality is not None and restored.quality.rows == 60


def test_chart_artifacts_must_use_a_generated_ref():
    """Chart refs are looked up by key in the session store, so the same shield
    that protects table refs applies: an invented one is rejected."""
    assert Artifact(type="chart", ref="c01").ref == "c01"
    with pytest.raises(ValidationError):
        Artifact(type="chart", ref="konut_grafigi")


def test_file_and_report_artifacts_may_use_free_refs():
    """Those refs are filenames chosen by the exporter, not store keys."""
    assert Artifact(type="file", ref="konut_kredisi.xlsx").ref == "konut_kredisi.xlsx"
    assert Artifact(type="report", ref="rapor_2026-09-15.html").type == "report"
