"""Tests for turning a downloaded document into tables (web/extract.py).

Every document here is built in the test, so nothing reaches the network and
nothing depends on a page staying the way it was. The PDF is assembled by
hand (`_pdf`): a real one-page file with a text layer and no ruling lines,
which is exactly the shape that defeated pdfplumber's default table strategy
on Borsa İstanbul's gold data.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from lakehouse_agent.web import extract as extractor


def _pdf(rows: list[tuple[int, int, str]]) -> bytes:
    """A one-page PDF with `(x, y, text)` drawn in Helvetica."""
    stream = (
        "BT /F1 10 Tf\n"
        + "".join(f"1 0 0 1 {x} {y} Tm ({text}) Tj\n" for x, y, text in rows)
        + "ET"
    )
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{start}\n%%EOF\n"
    ).encode()
    return bytes(out)


_PDF_BYTES = _pdf([(20, 160, "AY")])


# --- what kind of document is this ------------------------------------------


@pytest.mark.parametrize(
    ("body", "content_type", "url", "kind"),
    [
        (_PDF_BYTES, "", "https://x/rapor", "pdf"),  # the bytes win over the name
        (b"<html><body>x</body></html>", "", "https://x/sayfa", "html"),
        (b"a,b\n1,2\n", "text/csv", "https://x/veri", "csv"),
        (b"{}", "application/json", "https://x/veri", "json"),
        (b"\x89PNG\r\n", "image/png", "https://x/grafik.png", "image"),
        (b"duz metin", "text/plain", "https://x/not.txt", "text"),
    ],
)
def test_the_document_kind_is_read_from_the_bytes_then_the_address(
    body, content_type, url, kind
):
    assert extractor.extract(body, content_type=content_type, url=url).kind == kind


# --- pdf ---------------------------------------------------------------------


def test_a_pdf_table_without_ruling_lines_is_still_read():
    """Measured on Borsa İstanbul's `kmp_au.pdf`: the default (line-based)
    strategy returned a single column, and aligning words by their x position
    returned the table. A PDF drawn with no lines at all is that case."""
    body = _pdf(
        [
            (20, 160, "AY"),
            (120, 160, "HACIM"),
            (20, 140, "Ocak"),
            (120, 140, "1.234"),
            (20, 120, "Subat"),
            (120, 120, "2.345"),
        ]
    )

    got = extractor.extract(body, content_type="application/pdf", url="https://x/a.pdf")

    assert got.kind == "pdf"
    assert got.title == "AY HACIM"
    assert "Subat 2.345" in got.text
    assert len(got.tables) == 1
    frame = got.tables[0].frame
    assert frame.shape[1] == 2
    assert "Ocak" in frame.to_string() and "2.345" in frame.to_string()


def test_a_pdf_with_no_text_layer_goes_to_ocr():
    """A scanned bulletin has nothing to extract; the pages are rendered and
    read by MIA's OCR model, three images per request."""
    calls: list[int] = []

    def ocr(images: list[bytes]) -> str:
        calls.append(len(images))
        assert images and images[0][:4] == b"\x89PNG"
        return "TARANMIS SAYFA"

    got = extractor.extract(
        _pdf([]), content_type="application/pdf", url="https://x/tarama.pdf", ocr=ocr
    )

    assert calls == [1]
    assert got.text == "TARANMIS SAYFA"
    assert any("OCR" in note for note in got.notes)


def test_a_pdf_with_no_text_and_no_ocr_says_what_is_missing():
    got = extractor.extract(
        _pdf([]), content_type="application/pdf", url="https://x/t.pdf"
    )

    assert got.text == ""
    assert any("metin katmani yok" in note for note in got.notes)


# --- html --------------------------------------------------------------------


_PAGE = """
<html><head><title>Endeks - Borsa</title></head><body>
<table><tr><td>Menu</td></tr></table>
<main>
<p>Bu sayfa BIST TUM-100 endeksini anlatir ve 2026 yilinda yayimlanmistir.</p>
<table>
  <tr><th>Endeks Kodu</th><th>Deger</th></tr>
  <tr><td>XTUMY</td><td>12.345</td></tr>
  <tr><td>XU100</td><td>9.876</td></tr>
</table>
<table><tr><th>Guncel</th><th>Onceki</th></tr></table>
<a href="/dosyalar/altin.pdf">Altin Islemleri</a>
<a href="/hakkimizda">Kurumsal</a>
</main></body></html>
"""


def test_only_data_tables_survive():
    """A page's `<table>` is as often layout as data. One-column menus and
    empty shells (the values a page fills in with JavaScript) are dropped
    before anything downstream sees them."""
    got = extractor.extract(_PAGE.encode(), content_type="text/html", url="https://x/e")

    assert got.title == "Endeks - Borsa"
    assert len(got.tables) == 1
    frame = got.tables[0].frame
    assert list(frame.columns) == ["Endeks Kodu", "Deger"]
    assert frame.shape == (2, 2)


def test_the_data_files_a_page_links_to_are_listed_with_their_names():
    """The brief's own first target is a page with no table and 21 PDF links;
    without this the model has nothing to open next."""
    got = extractor.extract(
        b"<html><body><a href='/dosyalar/altin.pdf'>Altin Islemleri</a>"
        b"<a href='/hakkimizda'>Kurumsal</a></body></html>",
        content_type="text/html",
        url="https://borsa.example.com/veriler/piyasa",
    )

    assert got.links == [
        ("https://borsa.example.com/dosyalar/altin.pdf", "Altin Islemleri")
    ]
    assert any("veri dosyasina baglaniyor" in note for note in got.notes)


def test_a_page_with_no_table_and_no_files_says_javascript_may_be_the_reason():
    got = extractor.extract(
        b"<html><body><p>Veriler yukleniyor</p></body></html>",
        content_type="text/html",
        url="https://x/e",
    )

    assert got.tables == []
    assert any("JavaScript" in note for note in got.notes)


def test_needs_browser_render_is_true_for_an_empty_spa_root():
    body = b'<html><body><div id="root"></div></body></html>'
    found = extractor.extract(body, content_type="text/html", url="https://x/e")

    assert extractor.needs_browser_render(body, found) is True


def test_needs_browser_render_is_true_for_a_noscript_javascript_notice():
    filler = "Bu sayfa uzun bir metin icerir ve JavaScript'ten bahsetmez. " * 10
    body = (
        f"<html><body><noscript>Bu siteyi kullanmak icin JavaScript "
        f"gereklidir.</noscript><p>{filler}</p></body></html>"
    ).encode()
    found = extractor.extract(body, content_type="text/html", url="https://x/e")

    assert len(found.text) >= extractor.MIN_TEXT_CHARS_FOR_STATIC  # isolate the signal
    assert extractor.needs_browser_render(body, found) is True


def test_needs_browser_render_is_false_for_a_normal_page():
    filler = "Bu sayfa uzun bir metin icerir. " * 20
    body = f"<html><body><p>{filler}</p></body></html>".encode()
    found = extractor.extract(body, content_type="text/html", url="https://x/e")

    assert len(found.text) >= extractor.MIN_TEXT_CHARS_FOR_STATIC
    assert extractor.needs_browser_render(body, found) is False


def test_needs_browser_render_is_false_when_a_table_was_already_found():
    body = (
        b"<html><body><table><tr><th>Ay</th><th>Deger</th></tr>"
        b"<tr><td>Ocak</td><td>1</td></tr><tr><td>Subat</td><td>2</td></tr>"
        b"</table></body></html>"
    )
    found = extractor.extract(body, content_type="text/html", url="https://x/e")

    assert found.tables  # sanity: the static (first) layer already worked
    assert extractor.needs_browser_render(body, found) is False


def test_needs_browser_render_is_false_for_non_html():
    body = b"a,b\n1,2\n3,4\n"
    found = extractor.extract(body, content_type="text/csv", url="https://x/e.csv")

    assert extractor.needs_browser_render(body, found) is False


def test_the_text_is_the_page_not_its_menu():
    """Measured on borsaistanbul.com: without this the excerpt handed to the
    model is thousands of characters of navigation."""
    got = extractor.extract(_PAGE.encode(), content_type="text/html", url="https://x/e")

    assert "BIST TUM-100 endeksini anlatir" in got.text
    assert "Kurumsal" not in got.text  # a two-word menu item, no digits


def test_data_a_page_carries_inside_itself_becomes_a_table():
    """Pages that build their tables in the browser often still ship the rows
    in the page. That is read -- no endpoint is guessed."""
    page = (
        '<html><body><script type="application/json">'
        '[{"ay": "2026-01", "deger": 1}, {"ay": "2026-02", "deger": 2}]'
        "</script></body></html>"
    )

    got = extractor.extract(page.encode(), content_type="text/html", url="https://x/e")

    assert len(got.tables) == 1
    assert list(got.tables[0].frame.columns) == ["ay", "deger"]
    assert any("gomulu" in note for note in got.notes)


# --- spreadsheets and text ---------------------------------------------------


def test_every_sheet_of_a_workbook_becomes_a_table():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"ay": ["2026-01", "2026-02"], "deger": [1, 2]}).to_excel(
            writer, sheet_name="aylik", index=False
        )
        pd.DataFrame({"yil": [2025, 2026], "deger": [3, 4]}).to_excel(
            writer, sheet_name="yillik", index=False
        )

    got = extractor.extract(
        buffer.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        url="https://x/veri.xlsx",
    )

    assert [table.name for table in got.tables] == ["aylik", "yillik"]
    assert got.tables[0].frame.shape == (2, 2)


def test_a_semicolon_separated_export_is_read_as_columns():
    """Turkish exports are as often ";" as ","; the separator is sniffed."""
    got = extractor.extract(
        b"ay;deger\n2026-01;1\n2026-02;2\n",
        content_type="text/csv",
        url="https://x/veri.csv",
    )

    assert len(got.tables) == 1
    assert list(got.tables[0].frame.columns) == ["ay", "deger"]


def test_an_image_is_read_by_ocr():
    got = extractor.extract(
        b"\x89PNG\r\n",
        content_type="image/png",
        url="https://x/t.png",
        ocr=lambda _i: "ALTIN 1.234",
    )

    assert got.kind == "image" and got.text == "ALTIN 1.234"
    assert got.title == "ALTIN 1.234"


def test_a_windows_turkish_document_is_still_readable():
    """cp1254 is what Turkish exports still come in; a decode error must not
    lose the document."""
    got = extractor.extract(
        "ay;deger\nŞubat;2\n".encode("cp1254"),
        content_type="text/csv",
        url="https://x/veri.csv",
    )

    assert "Şubat" in got.tables[0].frame.to_string()


# --- what the 23 Sep review found --------------------------------------------


def test_a_turkish_decimal_in_an_html_table_keeps_its_value():
    """pandas' defaults are the Anglo ones, so "30,15" came back as int64
    3015 -- a hundredfold error, and an int column `prepare_table`'s Turkish
    repair can no longer see. Every rate and FX quote on a Turkish page
    arrives written this way."""
    html = (
        "<table><tr><th>Ay</th><th>Kur</th><th>Hacim</th></tr>"
        "<tr><td>2026-01</td><td>30,15</td><td>1.234.567,89</td></tr>"
        "<tr><td>2026-02</td><td>31,40</td><td>2.345.678,90</td></tr></table>"
    )

    got = extractor.extract(html.encode(), content_type="text/html", url="https://x/t")

    frame = got.tables[0].frame
    assert frame["Kur"].tolist() == [30.15, 31.4]
    assert frame["Hacim"].tolist() == [1234567.89, 2345678.90]


def test_an_unterminated_script_does_not_take_the_document_with_it():
    """The old pattern was `<script[^>]*>.*?</script>`, and one opener with
    no closer made it scan to the end and backtrack -- measured quadratic,
    about 17 minutes at the 10 MB fetch cap, on a thread the step timeout
    abandons but cannot kill."""
    html = "<p>once</p>" + "<script>x" * 500 + "<p>sonra</p>"

    text = extractor._text_of(html)

    assert "x" * 20 not in text  # the script bodies are gone


def test_a_two_column_pdf_table_keeps_its_header():
    """The commonest printed shape is (label, value), which has exactly one
    number in two cells. A strict majority called that a header row, so no
    data row was ever found and the table came back headerless with its
    header as data -- leaving every value as text nothing could sum."""
    frame = extractor._frame_from_rows(
        [["AY", "HACIM"], ["Ocak", "1.234"], ["Subat", "2.345"]]
    )

    assert list(frame.columns) == ["AY", "HACIM"]
    assert frame["AY"].tolist() == ["Ocak", "Subat"]


def test_a_link_that_is_not_http_is_not_offered_to_the_model():
    """The link list is read out as "files you can open". A page that lists
    `file:///etc/passwd.csv` was handing the model a menu of addresses to
    try; `check_url` refuses them, but they do not belong in front of it."""
    html = (
        "<a href='file:///etc/passwd.csv'>Yillik rapor</a>"
        "<a href='https://x.example.com/veri.csv'>Gercek dosya</a>"
    )

    links = extractor._data_links(html, "https://x.example.com/")

    assert [url for url, _ in links] == ["https://x.example.com/veri.csv"]


class _Word(dict):
    """The keys pdfplumber gives each word."""

    def __init__(self, text, x0, x1, top):
        super().__init__(text=text, x0=x0, x1=x1, top=top, bottom=top + 8)


class _Row:
    def __init__(self, top):
        self.bbox = (0, top, 100, top + 10)


class _Page:
    """Enough of a pdfplumber page for `_repaired_header`.

    The columns are 0-30, 30-70, 70-100; the heading words straddle 30 and 70
    exactly as they do on Borsa İstanbul's gold table, which is what the
    cell-by-cell reading splits.
    """

    bbox = (0, 0, 100, 100)

    def __init__(self):
        self.rows = [_Row(0), _Row(20)]
        self.cells = [(0, 0, 30, 10), (30, 0, 70, 10), (70, 0, 100, 10)]

    def find_tables(self, _settings):
        return [self]

    def extract_words(self):
        return [
            _Word("AY", 2, 20, 2),
            _Word("Hacim/Volume", 26, 60, 2),  # midpoint 43 -> middle column
            _Word("TOPLAM", 72, 96, 2),
            _Word("dipnot", 2, 20, 60),  # below the first data row: ignored
        ]


def test_a_pdf_heading_is_read_word_by_word_not_cell_by_cell():
    """Without ruling lines the column boundaries come from where the numbers
    sit, and a printed heading does not line up with them: "Hacim/Volume"
    arrived as "H" in one column and "acim/Volume" in the next. A word is
    atomic, so assigning whole words to the column their midpoint falls in
    cannot split one."""
    rows = [["H", "acim/Volume", "TO"], ["Ocak", "1.234", "5"]]

    header = extractor._repaired_header(_Page(), rows)

    assert header == ["AY", "Hacim/Volume", "TOPLAM"]


def test_a_heading_that_cannot_be_rebuilt_leaves_the_table_alone():
    """A page whose columns do not match the extracted grid gets the old
    reading rather than a guess."""

    class _Mismatched(_Page):
        def find_tables(self, _settings):
            return []

    assert extractor._repaired_header(_Mismatched(), [["a", "b", "c"]]) is None


def test_a_links_entities_are_unescaped_before_it_is_offered():
    """TCMB writes `...?MOD=AJPERES&amp;CACHEID=...`. Handed over verbatim,
    the second parameter becomes `amp;CACHEID` -- a different address from
    the one the page meant."""
    from lakehouse_agent.web import extract

    html = (
        '<a href="/wcm/connect/abc/DUY2023-25.pdf?MOD=AJPERES&amp;CACHEID=X">'
        "Basin Duyurusu</a>"
    )

    links = extract._data_links(html, "https://www.tcmb.gov.tr/sayfa")

    assert links == [
        (
            "https://www.tcmb.gov.tr/wcm/connect/abc/DUY2023-25.pdf?MOD=AJPERES&CACHEID=X",
            "Basin Duyurusu",
        )
    ]
