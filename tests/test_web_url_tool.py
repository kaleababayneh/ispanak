"""Tests for the `read_url` tool: what the session gets out of a document.

The fetch is stood in for, so no test reaches the network; what is exercised
is everything after it -- the tables, which one becomes active, the quality
card, the source, and what the model is told when a document holds no table.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd
import pytest

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.tools import web_url
from lakehouse_agent.tools.lake_adapter import LakeAdapter
from lakehouse_agent.tools.web_url import ReadUrlArgs, excerpt, make_read_url
from lakehouse_agent.web.fetch import Fetched, WebFetchError

PAGE = """
<html><head><title>Piyasa Verileri</title></head><body><main>
<p>Kiymetli madenler piyasasinin 2026 yilina ait islem verileri asagidadir.</p>
<table>
  <tr><th>Ay</th><th>Hacim</th></tr>
  <tr><td>2026-01</td><td>1.234</td></tr>
  <tr><td>2026-02</td><td>2.345</td></tr>
</table>
<table>
  <tr><th>Ay</th><th>Islem</th><th>Miktar</th></tr>
  <tr><td>2026-01</td><td>13.837</td><td>1.017</td></tr>
  <tr><td>2026-02</td><td>12.649</td><td>839</td></tr>
</table>
</main></body></html>
"""


@pytest.fixture
def spec():
    return make_read_url()


@pytest.fixture
def context(tmp_path):
    return ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM([]),
        lake=LakeAdapter(catalog_path=tmp_path / "yok.parquet"),
        settings=AgentSettings(MIA_API_KEY="test-only", DATA_DIR=str(tmp_path)),
    )


def _document(
    body: bytes, *, content_type: str, url: str = "https://x/veri"
) -> Fetched:
    return Fetched(
        url=url,
        final_url=url,
        status=200,
        content_type=content_type,
        body=body,
        fetched_at="2026-09-22T12:00:00+00:00",
        path=None,
    )


def _serve(monkeypatch, document, *, raises: Exception | None = None):
    def fake_fetch(url, **_kwargs):
        if raises is not None:
            raise raises
        assert url == document.url
        return document

    monkeypatch.setattr(web_url, "fetch", fake_fetch)


# --- tables ------------------------------------------------------------------


def test_every_data_table_becomes_a_session_table(spec, context, monkeypatch):
    """Not just the first one. Asking the model to call again for the second
    table costs a step, and a turn has twelve."""
    _serve(monkeypatch, _document(PAGE.encode(), content_type="text/html"))

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert result.ok
    assert len(context.written) == 2
    assert all(ref.endswith("_read_url") for ref in context.written)
    assert all(ref in result.summary for ref in context.written)


def test_the_largest_table_is_the_one_a_follow_up_works_on(spec, context, monkeypatch):
    _serve(monkeypatch, _document(PAGE.encode(), content_type="text/html"))

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert context.new_active == result.table_ref
    assert context.written[result.table_ref].shape[1] == 3  # the wider table
    assert result.quality is not None and result.quality.table_ref == result.table_ref


def test_a_table_arrives_cleaned_and_carries_a_quality_card(spec, context, monkeypatch):
    """The same `prepare_table` the lake uses, so a URL's table and a pool
    series are the same kind of thing to the analysis tools: the date column
    is `period` whatever the document called it, and Turkish-printed numbers
    are numbers."""
    csv = "donem;deger\n2026-01;1.234,5\n2026-02;2.345,6\n"
    _serve(monkeypatch, _document(csv.encode(), content_type="text/csv"))

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    frame = context.written[result.table_ref]
    assert result.quality.rows == len(frame) == 2
    assert result.preview  # the model sees rows, not the frame
    assert list(frame.columns) == ["period", "deger"]
    assert frame["deger"].tolist() == [1234.5, 2345.6]
    assert any("donem" in issue for issue in result.quality.issues)


def test_a_short_table_is_previewed_whole(spec, context, monkeypatch):
    """An answer may only quote numbers a tool put in front of the model
    (agent/verifier.py). A twelve-month bulletin has to arrive whole, or the
    months in the middle cannot be talked about."""
    rows = "\n".join(f"2026-{month:02d};{month}00" for month in range(1, 13))
    _serve(
        monkeypatch,
        _document(f"donem;deger\n{rows}\n".encode(), content_type="text/csv"),
    )

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert len(result.preview) == 12


def test_the_document_is_cited_with_the_time_it_was_read(spec, context, monkeypatch):
    _serve(monkeypatch, _document(PAGE.encode(), content_type="text/html"))

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    source = result.sources[0]
    assert source.url == "https://x/veri"
    assert source.series_id is None  # not a pool series
    assert source.name == "Piyasa Verileri"
    assert source.fetched_at == datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def test_nothing_is_written_to_the_pool(spec, context, monkeypatch):
    """Gold is what it was; a demo-time document lives in the session."""
    _serve(monkeypatch, _document(PAGE.encode(), content_type="text/html"))

    spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert not (context.settings.gold_dir / "observations.parquet").exists()


# --- documents with no table --------------------------------------------------


def test_a_page_of_links_tells_the_model_what_to_open_next(spec, context, monkeypatch):
    """The brief's first target: a page holding no table and linking 21 PDFs."""
    page = (
        b"<html><body><a href='/dosyalar/kmp_au.pdf'>Altin Islemleri</a>"
        b"<a href='/kurumsal'>Hakkimizda</a></body></html>"
    )
    _serve(
        monkeypatch,
        _document(
            page, content_type="text/html", url="https://borsa.example.com/veriler"
        ),
    )

    result = spec.fn(ReadUrlArgs(url="https://borsa.example.com/veriler"), context)

    assert result.ok and result.table_ref is None
    assert "https://borsa.example.com/dosyalar/kmp_au.pdf" in result.summary
    assert "Altin Islemleri" in result.summary
    assert context.written == {}


def test_an_address_that_cannot_be_fetched_is_reported_not_raised(
    spec, context, monkeypatch
):
    _serve(
        monkeypatch,
        _document(b"", content_type=""),
        raises=WebFetchError("ic aga cozuluyor"),
    )

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert result.ok is False
    assert "ic aga" in result.summary
    assert context.written == {}


def test_a_document_that_cannot_be_parsed_suggests_another_format(
    spec, context, monkeypatch
):
    _serve(monkeypatch, _document(b"%PDF-1.4 bozuk", content_type="application/pdf"))

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert result.ok is False
    assert "PDF" in result.summary and "okunamadi" in result.summary


def test_reading_a_cached_document_says_so(spec, context, monkeypatch):
    document = _document(PAGE.encode(), content_type="text/html")
    _serve(monkeypatch, Fetched(**{**document.__dict__, "from_cache": True}))

    result = spec.fn(ReadUrlArgs(url="https://x/veri"), context)

    assert any("onbellek" in warning for warning in result.warnings)


# --- the excerpt --------------------------------------------------------------


def test_the_excerpt_follows_the_goal_and_keeps_the_rows_around_it():
    """A number means nothing without the row it sits in, so the lines on
    either side come too."""
    text = "\n".join(
        [
            "KIYMETLI MADENLER PIYASASI",
            "Gumus islemleri 2026",
            "Ocak 5.000",
            "Altin islemleri 2026",
            "Ocak 93.824.682.381",
            "Subat 90.534.914.196",
        ]
    )

    got = excerpt(text, "altin hacmi")

    assert "Altin islemleri 2026" in got
    assert "Ocak 93.824.682.381" in got  # the line after the hit
    assert "Ocak 5.000" in got  # and the line before it
    assert "KIYMETLI MADENLER PIYASASI" not in got  # nowhere near the goal


def test_without_a_goal_the_excerpt_is_the_opening():
    text = "Birinci satir\nIkinci satir\nUcuncu satir"

    assert excerpt(text, None, limit=25).startswith("Birinci satir")


def test_a_goal_nothing_matches_falls_back_to_the_opening():
    assert excerpt("Altin\nGumus", "bakir").startswith("Altin")


# --- the tool's own contract --------------------------------------------------


def test_fetching_an_address_is_behind_the_human_gate(spec, context):
    """It leaves our process, costs time and can fail mid-demo."""
    assert spec.approval == "fetch"
    preview = spec.approval_preview(
        ReadUrlArgs(url="https://x/veri", goal="altin"), context
    )
    assert preview["adres"] == "https://x/veri" and preview["amac"] == "altin"


def test_the_tool_is_registered_with_the_others():
    from lakehouse_agent.tools.toolset import make_registry

    assert "read_url" in make_registry().names()


def test_an_excel_document_becomes_one_table_per_sheet(spec, context, monkeypatch):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"ay": ["2026-01"], "deger": [1]}).to_excel(
            writer, sheet_name="aylik", index=False
        )
        pd.DataFrame({"yil": [2026], "deger": [2]}).to_excel(
            writer, sheet_name="yillik", index=False
        )
    _serve(
        monkeypatch,
        _document(
            buffer.getvalue(),
            content_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            url="https://x/veri.xlsx",
        ),
    )

    result = spec.fn(ReadUrlArgs(url="https://x/veri.xlsx"), context)

    assert len(context.written) == 2
    assert "aylik" in result.summary and "yillik" in result.summary


# --- an address the model retyped is read as the question wrote it --------


def test_a_retyped_address_is_read_as_the_question_wrote_it(
    spec, monkeypatch, tmp_path
):
    """Measured 23 Sep: the address held two plus signs already, the model
    continued the pattern (`main+menu/duyurular` -> `main+menu+duyurular`)
    and the fetch failed on a page that exists. The person's own text is the
    ground truth. `_serve` asserts on the address it is handed, so this test
    fails if the tool passes the model's copy through."""
    given = (
        "https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/duyurular/"
        "basin/2023/duy2023-25"
    )
    retyped = given.replace("main+menu/duyurular", "main+menu+duyurular")
    context = ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM([]),
        lake=LakeAdapter(catalog_path=tmp_path / "yok.parquet"),
        settings=AgentSettings(MIA_API_KEY="test-only", DATA_DIR=str(tmp_path)),
        user_message=f"{given} adresini oku ve karari ozetle.",
    )
    _serve(monkeypatch, _document(b"<p>Karar</p>", content_type="text/html", url=given))

    result = spec.fn(ReadUrlArgs(url=retyped), context)

    assert result.ok
    assert any("sorudaki okundu" in warning for warning in result.warnings)
