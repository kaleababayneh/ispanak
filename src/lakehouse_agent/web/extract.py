"""Turn a downloaded document into tables and text. No network here.

What each kind needs, measured 22 Sep 2026 on the two targets the brief names:

* **PDF.** Borsa İstanbul's `kmp_au.pdf` (gold trading) carries a real text
  layer, so the text comes out whole. Its table has no ruling lines, though,
  and pdfplumber's default (line-based) strategy returned a single column.
  The text strategy -- align words into columns by their x positions --
  returned the table correctly (11 columns, one row per month). So: try the
  default, and when it comes back degenerate, try again by text.
* **HTML.** `borsaistanbul.com/endeks/xtumy` is served rendered, and
  `pandas.read_html` reads its metadata table (this is where "XTUMY" turns
  out to be *BIST TUM-100*, not BIST 100). Its "current values" table is
  empty in the HTML: that part is filled in by JavaScript. Rather than drive
  a browser, we say so -- an empty table is a fact about the page, and the
  series behind it is usually in the pool anyway.
* **Embedded JSON.** Many pages that build their tables in the browser still
  ship the data inside the page (`<script type="application/json">`,
  `__NEXT_DATA__`). That is read too: no guessing of endpoints, the data is
  right there.
* **Scanned PDFs and images** have no text to read, so they go to MIA's OCR
  model (`llm.ocr`, three images per request).

Tables that are not data -- HTML layout tables, empty shells -- are dropped
before anything downstream sees them: fewer than two columns or fewer than
two rows once blank rows and columns are gone.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Beyond this many tables from one document, the rest are named but not
#: built: a page with dozens of layout tables would otherwise fill the
#: session, and every table there costs the model prompt space on every step.
MAX_TABLES = 10
#: Pages rendered for OCR when a PDF has no text layer. Three per request
#: (the OCR client's own limit), two requests' worth.
MAX_OCR_PAGES = 6
#: The text an extraction hands on. The model reads a focused excerpt of it
#: (`tools/web_url.py`), not the whole document.
MAX_TEXT_CHARS = 20000

OcrFn = Callable[[list[bytes]], str]

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_SCRIPT_JSON = re.compile(
    r'<script[^>]*type="application/(?:ld\+)?json"[^>]*>(.*?)</script>', re.S | re.I
)
_NEXT_DATA = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S | re.I
)
#: Only the second half of the old pattern. The first half was
#: `<(script|style)[^>]*>.*?</\1>`, and an unterminated opener made `.*?` scan
#: to the end of the document and then backtrack, once per opener: measured
#: quadratic, 16 s at 1.3 MB and about 17 minutes at the 10 MB fetch cap, on a
#: thread the step timeout abandons but cannot kill. `_without_blocks` below
#: does the same job with `str.find`, in one pass.
_TAGS = re.compile(r"<[^>]+>")
_BLOCK_OPEN = re.compile(r"<(script|style)\b", re.I)
#: A number as printed here or abroad: "1.234.567,89", "1,234,567.89", "-12%".
_NUMBER = re.compile(r"[-+(]?\d[\d.,\s]*\)?%?")
#: The label is bounded rather than `(.*?)`: an unterminated <a> made that
#: scan to the end of the document once per link, the same quadratic blow-up
#: as `_TAGS` had (2 s at 0.43 MB). A link's visible text is a few words; 300
#: characters is already far more than `_data_links` keeps (80).
_HREF = re.compile(
    r"""<a[^>]+href=["']([^"']+)["'][^>]*>(.{0,300}?)</a>""", re.S | re.I
)
#: Links worth following from a page that is a list of documents: the brief's
#: own first target is one (Borsa İstanbul's market-data page holds no table,
#: it links 21 PDFs, one of which is the gold trading data).
_DATA_LINK = re.compile(r"\.(pdf|xlsx?|csv|json)(?:\?|#|$)", re.I)
MAX_LINKS = 15


@dataclass(frozen=True)
class Table:
    """One extracted table and where in the document it came from."""

    name: str
    frame: pd.DataFrame


@dataclass
class Extracted:
    """What a document turned out to hold."""

    kind: str  # pdf | html | excel | csv | json | image | text
    title: str | None = None
    text: str = ""
    tables: list[Table] = field(default_factory=list)
    skipped_tables: int = 0
    notes: list[str] = field(default_factory=list)
    #: (address, link text) for the data files a page links to.
    links: list[tuple[str, str]] = field(default_factory=list)


def extract(
    body: bytes,
    *,
    content_type: str = "",
    url: str = "",
    ocr: OcrFn | None = None,
) -> Extracted:
    """Read `body` according to what it is: content type first, then the
    address' extension, then the bytes themselves (a PDF starts with %PDF)."""
    kind = _kind(body, content_type, url)
    if kind == "pdf":
        return _from_pdf(body, ocr)
    if kind == "html":
        return _from_html(body, url)
    if kind == "excel":
        return _from_excel(body)
    if kind == "csv":
        return _from_csv(body)
    if kind == "json":
        return _from_json(body)
    if kind == "image":
        return _from_image(body, ocr)
    return Extracted(kind="text", text=_decode(body)[:MAX_TEXT_CHARS])


def _kind(body: bytes, content_type: str, url: str) -> str:
    lowered = (content_type or "").lower()
    suffix = url.lower().rsplit("?", 1)[0].rsplit("#", 1)[0]
    if body[:4] == b"%PDF" or "pdf" in lowered or suffix.endswith(".pdf"):
        return "pdf"
    if body[:2] == b"PK" and (
        "spreadsheet" in lowered or suffix.endswith((".xlsx", ".xlsm"))
    ):
        return "excel"
    if "excel" in lowered or suffix.endswith(".xls"):
        return "excel"
    if "csv" in lowered or suffix.endswith((".csv", ".tsv")):
        return "csv"
    if "json" in lowered or suffix.endswith(".json"):
        return "json"
    if lowered.startswith("image/") or suffix.endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif")
    ):
        return "image"
    if "html" in lowered or b"<html" in body[:2000].lower():
        return "html"
    return "text"


# --- pdf ---------------------------------------------------------------------


def _from_pdf(body: bytes, ocr: OcrFn | None) -> Extracted:
    import pdfplumber

    out = Extracted(kind="pdf")
    texts: list[str] = []
    tables: list[Table] = []
    with pdfplumber.open(io.BytesIO(body)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            texts.append(page.extract_text() or "")
            for index, raw in enumerate(_page_tables(page), start=1):
                frame = _frame_from_rows(raw)
                if frame is not None:
                    tables.append(Table(f"sayfa {number} tablo {index}", frame))
        pages = len(pdf.pages)

    if not any(text.strip() for text in texts):
        if ocr is None:
            out.notes.append(
                "PDF'te metin katmani yok (taranmis olabilir) ve OCR kullanilamadi."
            )
        else:
            out.text = _ocr_pdf(body, pages, ocr)
            out.notes.append(
                f"PDF'te metin katmani yok; ilk {min(pages, MAX_OCR_PAGES)} sayfa "
                "OCR ile okundu."
            )
    else:
        out.text = "\n".join(texts)[:MAX_TEXT_CHARS]

    out.title = _first_line(out.text)
    out.tables, out.skipped_tables = _keep(tables)
    return out


_TEXT_STRATEGY = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def _page_tables(page) -> list[list[list]]:  # noqa: ANN001 - pdfplumber page
    """A page's tables, by ruling lines when it has them and by the words'
    own alignment when it does not (see the module docstring)."""
    tables = [table for table in page.extract_tables() if table]
    if any(max((len(row) for row in table), default=0) >= 2 for table in tables):
        return tables
    by_text = page.extract_table(_TEXT_STRATEGY)
    if not by_text:
        return tables
    repaired = _repaired_header(page, by_text)
    if repaired is None:
        return [by_text]
    return [[repaired, *_after_header(by_text)]]


def _after_header(rows: list[list]) -> list[list]:
    """The rows from the first one that reads as data."""
    start = next(
        (
            index
            for index, row in enumerate(rows)
            if _mostly_numbers([(cell or "").strip() for cell in row])
        ),
        0,
    )
    return rows[start:]


def _repaired_header(page, rows: list[list]) -> list[str] | None:  # noqa: ANN001
    """The header read word by word instead of cell by cell, or None.

    Without ruling lines pdfplumber derives the column boundaries from where
    the *numbers* sit, and a printed heading does not line up with them: on
    Borsa İstanbul's gold table those boundaries cut straight through the
    words, so "Hacim/Volume (TL)" arrived as "H" in one column and
    "acim/Volume (TL)" in the next, "TOPLAM" as "TO" + "PLAM", "of" as "o" +
    "f". Seven of its eleven headings were damaged, and the model was handed
    column names it could not use.

    A word is atomic, so assigning whole words to the column their midpoint
    falls in cannot split one. Only the band between the table's top and its
    first data row is read, which is what keeps the document's own title --
    printed above the table, across its full width -- out of the headings.
    """
    try:
        found = page.find_tables(_TEXT_STRATEGY)
        if not found:
            return None
        table = found[0]
        first_data = next(
            (
                index
                for index, row in enumerate(rows)
                if _mostly_numbers([(cell or "").strip() for cell in row])
            ),
            None,
        )
        if not first_data or first_data >= len(table.rows):
            return None
        top, bottom = table.bbox[1], table.rows[first_data].bbox[1]
        edges = sorted(
            {cell[0] for cell in table.cells if cell}
            | {cell[2] for cell in table.cells if cell}
        )
        if len(edges) - 1 != len(rows[0]):
            return None
        columns: dict[int, list[tuple[float, float, str]]] = {}
        for word in page.extract_words():
            if not (top <= word["top"] and word["bottom"] <= bottom):
                continue
            middle = (word["x0"] + word["x1"]) / 2
            index = max(0, sum(1 for edge in edges if edge <= middle) - 1)
            columns.setdefault(index, []).append(
                (round(word["top"]), word["x0"], word["text"])
            )
        header = [
            " ".join(text for _, _, text in sorted(columns.get(index, [])))
            for index in range(len(edges) - 1)
        ]
    except Exception as exc:  # noqa: BLE001 - a heading is not worth a crash
        logger.debug("pdf basligi kelimelerden kurulamadi: %s", exc)
        return None
    return header if any(name.strip() for name in header) else None


def _ocr_pdf(body: bytes, pages: int, ocr: OcrFn) -> str:
    """Render the first pages and read them with MIA's OCR model."""
    import pypdfium2

    document = pypdfium2.PdfDocument(io.BytesIO(body))
    images: list[bytes] = []
    try:
        for index in range(min(pages, MAX_OCR_PAGES)):
            buffer = io.BytesIO()
            document[index].render(scale=2).to_pil().save(buffer, format="PNG")
            images.append(buffer.getvalue())
    finally:
        document.close()
    parts = [ocr(images[start : start + 3]) for start in range(0, len(images), 3)]
    return "\n".join(part for part in parts if part)[:MAX_TEXT_CHARS]


# --- html --------------------------------------------------------------------


def _from_html(body: bytes, url: str = "") -> Extracted:
    html = _decode(body)
    out = Extracted(kind="html")
    match = _TITLE.search(html)
    out.title = " ".join(match.group(1).split()) if match else None

    tables: list[Table] = []
    try:
        # flavor pinned: without it pandas falls back to BeautifulSoup when
        # lxml cannot parse a page, and reports the missing optional
        # dependency as the reason the page had no tables (measured on
        # borsaistanbul.com's market-data page, which is a list of links).
        # Turkish separators, not pandas' Anglo defaults. Measured: with the
        # defaults "30,15" comes back as int64 3015 -- a hundredfold error,
        # and an int column that `prepare_table`'s Turkish repair can no
        # longer see, so nothing downstream notices. Every rate, share and FX
        # quote on a Turkish page arrives this way.
        frames = pd.read_html(
            io.StringIO(html), flavor="lxml", thousands=".", decimal=","
        )
        for index, frame in enumerate(frames, start=1):
            cleaned = _clean(frame)
            if cleaned is not None:
                tables.append(Table(f"tablo {index}", cleaned))
    except ValueError:
        pass  # no <table> at all: read_html raises rather than returning []
    except Exception as exc:  # noqa: BLE001 - a broken page is not a crash
        out.notes.append(f"HTML tablolari okunamadi ({type(exc).__name__}).")

    embedded = _embedded_json(html)
    for index, frame in enumerate(embedded, start=1):
        tables.append(Table(f"sayfa icindeki json {index}", frame))
    if embedded:
        out.notes.append(
            f"Sayfanin icine gomulu {len(embedded)} veri blogu tabloya cevrildi."
        )

    out.text = _text_of(html)
    out.links = _data_links(html, url)
    out.tables, out.skipped_tables = _keep(tables)
    if not out.tables:
        out.notes.append(
            "Sayfada veri tablosu bulunamadi"
            + (
                f"; sayfa {len(out.links)} veri dosyasina baglaniyor, birini "
                "read_url ile ac."
                if out.links
                else "; tablo JavaScript ile yukleniyor olabilir."
            )
        )
    return out


def _data_links(html: str, url: str) -> list[tuple[str, str]]:
    """The PDF/Excel/CSV files a page links to, as absolute addresses.

    http(s) only. This list is read out to the model as "files you can open",
    so a page that links `file:///etc/passwd.csv` or the cloud metadata
    service would be handing it a menu of addresses to try. `check_url`
    refuses all of those at fetch time, but a list nobody should act on does
    not belong in front of the model in the first place.

    The link's own text comes from the page, so it is truncated and stripped
    of markup; it is still someone else's words, and `web_url` labels it as
    such.

    An `href` is HTML, so its entities are unescaped before it is used as an
    address. TCMB writes its PDF links as
    `...?MOD=AJPERES&amp;CACHEID=...`, and handing that to the model verbatim
    makes the second parameter `amp;CACHEID`. Harmless on that server, which
    ignores what it does not recognise; on one that reads the parameter, the
    address the model is invited to open is not the address the page meant.
    """
    from html import unescape
    from urllib.parse import urljoin, urlparse

    found: dict[str, str] = {}
    for href, label in _HREF.findall(html):
        if not _DATA_LINK.search(href):
            continue
        absolute = urljoin(url, unescape(href.strip()))
        if urlparse(absolute).scheme not in ("http", "https"):
            continue
        text = " ".join(_TAGS.sub(" ", label).split())[:80]
        found.setdefault(absolute, text)
    return list(found.items())[:MAX_LINKS]


def _embedded_json(html: str) -> list[pd.DataFrame]:
    """Data the page ships inside itself, as tables. Only lists of flat
    records become tables; anything else is left alone rather than guessed
    into a shape."""
    frames: list[pd.DataFrame] = []
    for match in (*_SCRIPT_JSON.finditer(html), *_NEXT_DATA.finditer(html)):
        try:
            payload = json.loads(match.group(1).strip())
        except (ValueError, TypeError):
            continue
        for records in _record_lists(payload):
            frame = _clean(pd.DataFrame(records))
            if frame is not None:
                frames.append(frame)
    return frames[:MAX_TABLES]


def _record_lists(payload: object, depth: int = 0) -> list[list[dict]]:
    """Every list of same-shaped objects inside a JSON document."""
    if depth > 6:
        return []
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
        if len(records) >= 2 and all(
            not isinstance(value, (dict, list))
            for record in records[:5]
            for value in record.values()
        ):
            return [records]
        return [found for item in payload for found in _record_lists(item, depth + 1)]
    if isinstance(payload, dict):
        return [
            found
            for value in payload.values()
            for found in _record_lists(value, depth + 1)
        ]
    return []


#: Below this many characters of a page's own text (after the menu is
#: stripped, `_text_of`), a real page has not loaded yet: measured against an
#: SPA shell's "you need JavaScript to run this app" placeholder, which is a
#: sentence or two. A real article or table page clears this by a wide margin.
MIN_TEXT_CHARS_FOR_STATIC = 250

_NOSCRIPT = re.compile(r"<noscript[^>]*>(.*?)</noscript>", re.S | re.I)
#: An SPA's mount point with nothing in it yet -- the two names React's and
#: Vue's own starter templates use. Matched on the `id` value, not by
#: assuming every empty `<div>` is one: a page can legitimately have an empty
#: `<div id="spacer">` that means nothing about how the page was built.
_EMPTY_SPA_ROOT = re.compile(
    r'<div[^>]*\bid=["\'](?:root|app)["\'][^>]*>\s*</div>', re.I
)


def needs_browser_render(body: bytes, extracted: Extracted) -> bool:
    """True when a headless browser stands a chance where `extracted` did not.

    Only asked about HTML that came out of `_from_html` with no table at all
    -- `_from_html` already tries a page's own `<table>`s and then the JSON it
    may carry inside itself (`_embedded_json`), and both land in the same
    `out.tables`, so "no table" already means both of those failed. A
    document with a table, or one that was never HTML, has nothing for a
    browser to add.

    Any one of three signals is enough: the page's own text (menu already
    stripped, `_text_of`) is shorter than a loaded page ever legitimately is
    (`MIN_TEXT_CHARS_FOR_STATIC`); a `<noscript>` block says outright that
    JavaScript is required; or an SPA's mount point is still empty
    (`_EMPTY_SPA_ROOT`). Pure and network-free like the rest of this module,
    so it can be tested against hand-written HTML.
    """
    if extracted.kind != "html" or extracted.tables:
        return False
    html = _decode(body)
    if _EMPTY_SPA_ROOT.search(html):
        return True
    if any(
        "javascript" in match.group(1).lower() for match in _NOSCRIPT.finditer(html)
    ):
        return True
    return len(extracted.text.strip()) < MIN_TEXT_CHARS_FOR_STATIC


#: A page's own content, when it marks it. Without this the text handed to
#: the model is the site's navigation menu -- measured on
#: borsaistanbul.com/endeks/xtumy, where the menu runs for thousands of
#: characters before the first fact about the index.
_MAIN = re.compile(r"<(main|article)\b[^>]*>(.*?)</\1>", re.S | re.I)


#: Tags that end a line of prose. Without breaking on these the whole page
#: collapses into one line, and an excerpt can only be its first characters.
_BLOCK = re.compile(
    r"</?(p|div|li|tr|td|th|h[1-6]|br|section|article|main|table|ul|ol)\b[^>]*>",
    re.I,
)
#: A menu item is a couple of words with no number in it. A page's navigation
#: is hundreds of those, and it would otherwise be all the model reads.
_MIN_LINE = 40


def _without_blocks(html: str) -> str:
    """`html` with every <script>/<style> body removed, in one forward pass.

    `str.find` never backtracks, so a page with a thousand unterminated
    openers costs a thousand searches, not a thousand full scans. An opener
    with no closer swallows the rest of the document, which is what a browser
    does with it too.
    """
    out: list[str] = []
    position = 0
    while (match := _BLOCK_OPEN.search(html, position)) is not None:
        out.append(html[position : match.start()])
        closer = f"</{match.group(1).lower()}"
        end = html.lower().find(closer, match.end())
        if end == -1:
            return "".join(out)
        position = html.find(">", end)
        if position == -1:
            return "".join(out)
        position += 1
    out.append(html[position:])
    return "".join(out)


def _text_of(html: str) -> str:
    match = _MAIN.search(html)
    body = _without_blocks(match.group(2) if match else html)
    lines = [
        " ".join(line.split())
        for line in _TAGS.sub(" ", _BLOCK.sub("\n", body)).splitlines()
    ]
    kept = [
        line
        for line in lines
        if len(line) >= _MIN_LINE or any(character.isdigit() for character in line)
    ]
    text = "\n".join(kept or [line for line in lines if line])
    return text[:MAX_TEXT_CHARS]


# --- spreadsheets, json, images ----------------------------------------------


def _from_excel(body: bytes) -> Extracted:
    out = Extracted(kind="excel")
    try:
        sheets = pd.read_excel(io.BytesIO(body), sheet_name=None)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        out.notes.append(f"Excel dosyasi okunamadi ({type(exc).__name__}).")
        return out
    tables = []
    for name, frame in sheets.items():
        cleaned = _clean(frame, min_rows=1)
        if cleaned is not None:
            tables.append(Table(str(name), cleaned))
    out.tables, out.skipped_tables = _keep(tables)
    return out


def _from_csv(body: bytes) -> Extracted:
    out = Extracted(kind="csv")
    text = _decode(body)
    try:
        # sep=None asks the sniffer: Turkish exports are as often ";" as ",".
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        out.notes.append(f"CSV okunamadi ({type(exc).__name__}).")
        out.text = text[:MAX_TEXT_CHARS]
        return out
    cleaned = _clean(frame, min_rows=1)
    out.tables = [Table("csv", cleaned)] if cleaned is not None else []
    return out


def _from_json(body: bytes) -> Extracted:
    out = Extracted(kind="json")
    try:
        payload = json.loads(_decode(body))
    except ValueError as exc:
        out.notes.append(f"JSON okunamadi ({exc}).")
        return out
    tables = [
        Table(f"json {index}", frame)
        for index, frame in enumerate(
            filter(
                None,
                (
                    _clean(pd.DataFrame(records), min_rows=1)
                    for records in _record_lists(payload)
                ),
            ),
            start=1,
        )
    ]
    out.tables, out.skipped_tables = _keep(tables)
    if not tables:
        out.text = json.dumps(payload, ensure_ascii=False)[:MAX_TEXT_CHARS]
    return out


def _from_image(body: bytes, ocr: OcrFn | None) -> Extracted:
    if ocr is None:
        return Extracted(kind="image", notes=["Gorsel icin OCR kullanilamadi."])
    text = ocr([body])[:MAX_TEXT_CHARS]
    return Extracted(kind="image", text=text, title=_first_line(text))


# --- shared ------------------------------------------------------------------


def _decode(body: bytes) -> str:
    for encoding in ("utf-8", "cp1254", "latin-1"):
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _frame_from_rows(rows: list[list]) -> pd.DataFrame | None:
    """A pdfplumber table (rows of cells) as a frame.

    The header is however many rows come before the first row of numbers,
    joined per column: a printed table's heading is routinely three lines
    ("AY/MONTH" over "Hacim/Volume" over "(TL)"), and taking only the first
    of them leaves columns named `0, 1, 2` -- which is what the model then
    has to work with.
    """
    rows = [[(cell or "").strip() for cell in row] for row in rows if any(row)]
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    # `None`, not `0`: the old default conflated "row 0 is already data" with
    # "no data row anywhere", and both fell through to the headerless frame.
    first_data = next(
        (index for index, row in enumerate(padded) if _mostly_numbers(row)), None
    )
    if not first_data:  # no numbers at all, or numbers from the very first row
        return _clean(pd.DataFrame(padded))
    header = [
        " ".join(part for part in column if part).strip() or f"sutun {index + 1}"
        for index, column in enumerate(zip(*padded[:first_data], strict=True))
    ]
    header = _unique(header)
    return _clean(pd.DataFrame(padded[first_data:], columns=header))


def _mostly_numbers(row: list[str]) -> bool:
    """A data row: at least half of its filled cells read as numbers.

    Half, not over half. A two-column (label, value) table -- the commonest
    printed shape there is -- has exactly one number in two cells, and a
    strict majority called it a header row. With no data row found, the whole
    table came back headerless and the header row itself became data, which
    left every value as text nothing could sum (measured).
    """
    filled = [cell for cell in row if cell]
    if len(filled) < 2:
        return False
    numeric = sum(1 for cell in filled if _NUMBER.fullmatch(cell.replace(" ", "")))
    return numeric * 2 >= len(filled)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} {seen[name]}")
    return out


def _clean(frame: pd.DataFrame, *, min_rows: int = 2) -> pd.DataFrame | None:
    """Drop blank rows and columns; return None when what is left is not a
    data table (see the module docstring).

    `min_rows` is 2 where the table was found among others -- a page's
    one-row `<table>` is a heading or a menu -- and 1 for a document that is
    a table, where a single row is simply a short export.
    """
    if frame is None or frame.empty:
        return None
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if frame.shape[0] < min_rows or frame.shape[1] < 2:
        return None
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.reset_index(drop=True)


def _keep(tables: list[Table]) -> tuple[list[Table], int]:
    return tables[:MAX_TABLES], max(0, len(tables) - MAX_TABLES)


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return None
