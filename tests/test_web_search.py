"""Tests for open-web search: the module and the tool.

The provider is stood in for, so nothing here reaches DuckDuckGo. What is
exercised is the part that matters when it does: what the model is handed,
what it is told about trusting it, and what happens when the search fails.
"""

from __future__ import annotations

import json

import pytest

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.tools.web_search import (
    UNTRUSTED_NOTE,
    WebSearchArgs,
    make_web_search,
)
from lakehouse_agent.web import search as search_module
from lakehouse_agent.web.search import (
    SearchResult,
    WebSearchError,
    ddg_provider,
    search,
)

RESULTS = [
    SearchResult(
        title="TCMB PPK kararı, Haziran 2023",
        url="https://www.tcmb.gov.tr/duyuru/2023-24",
        snippet="Politika faizi yüzde 8,5'ten yüzde 15'e yükseltildi.",
    ),
    SearchResult(
        title="BDDK düzenlemesi",
        url="https://www.bddk.org.tr/Mevzuat/1164",
        snippet="Konut kredisi vade sınırı değişti.",
    ),
]


def _provider(results=RESULTS, calls=None):
    def provider(query, k):
        if calls is not None:
            calls.append((query, k))
        return list(results)[:k]

    return provider


@pytest.fixture
def context(tmp_path):
    return ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM([]),
        lake=None,
        settings=AgentSettings(MIA_API_KEY="test-only", DATA_DIR=str(tmp_path)),
    )


# --- the module ---------------------------------------------------------------


def test_a_query_is_asked_once_and_then_read_from_the_cache(tmp_path):
    """A demo that asks the same thing twice should not spend a second call,
    and a rate limit later in the run must not erase what was found."""
    calls: list[tuple[str, int]] = []
    provider = _provider(calls=calls)

    first, fetched_at, cached = search(
        "TCMB Haziran 2023", 2, provider=provider, root=tmp_path
    )
    second, again_at, cached_again = search(
        "  tcmb   haziran 2023 ", 2, provider=provider, root=tmp_path
    )

    assert not cached and cached_again
    assert calls == [("TCMB Haziran 2023", 2)]  # the second never reached it
    assert [r.url for r in second] == [r.url for r in first]
    assert again_at == fetched_at

    stored = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert stored["query"] == "TCMB Haziran 2023" and stored["fetched_at"]


def test_a_provider_failure_becomes_one_sentence(tmp_path):
    def broken(_query, _k):
        raise WebSearchError("Arama yapilamadi (RatelimitException): 202")

    with pytest.raises(WebSearchError, match="Arama yapilamadi"):
        search("x y z", 3, provider=broken, root=tmp_path)


def test_an_empty_query_is_refused_before_any_call(tmp_path):
    with pytest.raises(WebSearchError, match="Bos sorgu"):
        search("   ", 3, provider=_provider(), root=tmp_path)


def test_the_real_provider_wraps_whatever_the_library_raises(monkeypatch):
    """Rate limits, timeouts and DNS all arrive as different exceptions from
    `ddgs`; the tool only has to know that the search did not happen."""

    class Boom:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, *_args, **_kwargs):
            raise RuntimeError("Ratelimit")

    monkeypatch.setitem(
        __import__("sys").modules, "ddgs", type("m", (), {"DDGS": Boom})
    )

    with pytest.raises(WebSearchError, match="RuntimeError"):
        ddg_provider("tcmb", 3)


def test_results_keep_the_address_and_lose_nothing_else(tmp_path, monkeypatch):
    """A row without an address is not a result -- it cannot be cited."""

    class Rows:
        def __init__(self, *_args, **_kwargs):
            pass

        def text(self, *_args, **_kwargs):
            return [
                {"title": "  Bir  baslik ", "href": "https://x/a", "body": " metin "},
                {"title": "adressiz", "href": "", "body": "x"},
            ]

    monkeypatch.setitem(
        __import__("sys").modules, "ddgs", type("m", (), {"DDGS": Rows})
    )

    results = ddg_provider("tcmb", 5)

    assert len(results) == 1
    assert results[0].title == "Bir baslik" and results[0].snippet == "metin"


# --- the tool -----------------------------------------------------------------


def _run(context, monkeypatch, provider=None, **kwargs):
    monkeypatch.setattr(
        search_module, "ddg_provider", provider or _provider(), raising=True
    )
    return make_web_search().fn(
        WebSearchArgs(query="TCMB Haziran 2023", **kwargs), context
    )


def test_the_model_is_told_the_results_are_data_not_instructions(context, monkeypatch):
    """The text comes from strangers and lands in the model's context; a page
    saying "ignore your previous instructions" must meet a label that says
    what this is."""
    result = _run(context, monkeypatch)

    assert result.ok
    assert UNTRUSTED_NOTE in result.summary
    assert "sayilar buradan gelemez" in result.summary


def test_every_result_is_listed_with_its_address_and_cited(context, monkeypatch):
    result = _run(context, monkeypatch)

    for row in RESULTS:
        assert row.title in result.summary
        assert row.url in result.summary
    assert [source.url for source in result.sources] == [r.url for r in RESULTS]
    assert all(source.series_id is None for source in result.sources)


def test_no_table_and_no_numbers_come_out_of_a_search(context, monkeypatch):
    """A page found by a search is not a measurement: the tool writes no
    table, so nothing downstream can compute with it."""
    result = _run(context, monkeypatch)

    assert result.table_ref is None
    assert result.preview == []
    assert context.written == {}


def test_a_search_that_finds_nothing_does_not_claim_absence(context, monkeypatch):
    result = _run(context, monkeypatch, provider=_provider(results=[]))

    assert result.ok
    assert "kanitlamaz" in result.summary


def test_a_failed_search_says_so_and_says_what_it_does_not_mean(context, monkeypatch):
    def broken(_query, _k):
        raise WebSearchError("Arama yapilamadi (Ratelimit)")

    result = _run(context, monkeypatch, provider=broken)

    assert result.ok is False
    assert "Arama yapilamadi" in result.summary
    assert "olmadigini gostermez" in result.summary


def test_a_cached_answer_is_flagged_as_one(context, monkeypatch):
    _run(context, monkeypatch)
    again = _run(context, monkeypatch)

    assert any("onbellek" in warning for warning in again.warnings)


def test_searching_is_behind_the_human_gate(context):
    spec = make_web_search()

    assert spec.approval == "fetch"
    preview = spec.approval_preview(WebSearchArgs(query="TCMB faiz"), context)
    assert preview["sorgu"] == "TCMB faiz"


def test_the_tool_is_registered_with_the_others():
    from lakehouse_agent.tools.toolset import make_registry

    assert "web_search" in make_registry().names()
