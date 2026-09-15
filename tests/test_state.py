"""Session bookkeeping: table references, source de-duplication, step numbers.

Small surface, but three of the loop's promises rest on it. References are
generated and never taken from the model, a source cited twice appears once,
and step numbers are unique across a whole session so the traceability panel
can key on them.
"""

import pandas as pd
import pytest

from lakehouse_agent.agent.state import (
    SessionState,
    initial_state,
    new_chart_ref,
    new_table_ref,
    slug,
)
from lakehouse_agent.contracts import TABLE_REF_PATTERN, Source

FETCHED_AT = pd.Timestamp("2026-09-15T06:00:00Z").to_pydatetime()


def _source(**kwargs) -> Source:
    """`fetched_at` is required on purpose; tests should not have to care when."""
    return Source(fetched_at=FETCHED_AT, **kwargs)


def test_generated_refs_match_the_pattern_the_model_is_shown():
    """Generation and validation have to agree, or every tool call fails.

    The pattern goes into the tool schema as a `pattern` constraint, so a
    generator that produced anything outside it would reject our own
    references while still letting nothing else through.
    """
    tables: dict[str, pd.DataFrame] = {}
    for suffix in ("get_series", "deflate", "quality_table", "cok uzun bir arac adi"):
        ref = new_table_ref(tables, suffix)
        assert pd.Series([ref]).str.fullmatch(TABLE_REF_PATTERN).all(), ref
        tables[ref] = pd.DataFrame()

    assert list(tables) == [
        "t01_get_series",
        "t02_deflate",
        "t03_quality_table",
        "t04_cok_uzun_bir_arac_adi",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("get_series", "get_series"),
        ("Konut Kredisi", "konut_kredisi"),
        ("2021 verisi", "t_2021_verisi"),
        ("---", "table"),
        ("a" * 40, "a" * 24),
    ],
)
def test_slug_always_produces_something_the_pattern_accepts(text, expected):
    """Including the awkward cases: digits first, punctuation only, too long."""
    assert slug(text) == expected


def test_a_removed_table_does_not_free_its_number_for_reuse():
    """The quality gate removes a table when a person rejects it.

    Numbering by "how many tables exist" would then hand the next table a
    number still in use and overwrite it without a word. Numbering from the
    highest one seen cannot do that.
    """
    tables = {"t01_a": pd.DataFrame(), "t02_b": pd.DataFrame()}
    del tables["t01_a"]
    assert new_table_ref(tables, "c") == "t03_c"


def test_chart_refs_count_up():
    assert new_chart_ref([]) == "c01"
    assert new_chart_ref([1, 2]) == "c03"


def test_step_numbers_are_unique_across_turns():
    """The interface keys step cards on this number; a repeat overwrites a card."""
    state = initial_state("s1")
    first = [state.next_step_no() for _ in range(3)]
    state.turn_no += 1
    state.steps_this_turn = 0
    second = [state.next_step_no() for _ in range(3)]
    assert first + second == [1, 2, 3, 4, 5, 6]


def test_the_same_series_fetched_twice_is_cited_once():
    """Two steps can legitimately touch one series; the citation list is one."""
    state = SessionState(session_id="s1")
    state.add_sources([_source(series_id="evds:TP.KTF12")])
    state.add_sources(
        [
            _source(series_id="evds:TP.KTF12"),
            _source(series_id="evds:TP.TUKFIY2025.GENEL"),
        ]
    )
    assert [source.key() for source in state.sources] == [
        "evds:TP.KTF12",
        "evds:TP.TUKFIY2025.GENEL",
    ]


def test_a_url_source_and_a_series_source_are_different_entries():
    """A scraped page and a catalog series are two provenance claims, not one."""
    state = SessionState(session_id="s1")
    state.add_sources(
        [
            _source(series_id="evds:TP.KTF12"),
            _source(url="https://www.bddk.org.tr/BultenAylik"),
        ]
    )
    assert len(state.sources) == 2
