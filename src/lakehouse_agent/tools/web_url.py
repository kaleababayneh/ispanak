"""`read_url`: the tool for an address the question names.

The brief promises it outright -- on demo day a URL arrives in the prompt and
the system is expected to read a PDF, a spreadsheet, an image or a page and
carry on analysing. So this tool ends where the pool's own tools begin: every
table it extracts goes through `prepare_table`, the same cleaning the lake
applies, and comes out with a quality card. From there a URL's table and a
pool series are the same kind of thing to `transform_table` or
`detect_anomalies`.

Three decisions worth knowing when reading this:

* **Every data table becomes a table**, not just the first. A page's tables
  are separate answers to separate questions, and making the model ask again
  for the second one costs a step it may not have. Layout and empty tables
  are dropped before this point (`web/extract.py`), and past ten the rest are
  named rather than built.
* **The largest becomes active.** "Add a column to this" then means the table
  a person is most likely looking at, and the quality card describes that one.
* **Nothing is written to the pool.** Gold stays what it was; a demo-time
  document lives in the session and in `data/adhoc/url/`.

Fetching is behind the human approval gate (`approval="fetch"`): it leaves our
process, costs time and can fail mid-demo.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.agent.formatting import preview_rows
from lakehouse_agent.contracts import Source, ToolResult
from lakehouse_agent.tools.registry import ToolSpec
from lakehouse_agent.web import extract as extractor
from lakehouse_agent.web.fetch import WebFetchError, fetch

logger = logging.getLogger(__name__)

#: Characters of the document's own text handed to the model. Enough for a
#: PDF's caption and the rows around what was asked for; the table carries
#: the numbers, so this is context, not data.
EXCERPT_CHARS = 1200


@dataclass(frozen=True)
class Built:
    """One table this call put into the session."""

    ref: str
    name: str
    frame: pd.DataFrame
    quality: Any


_WORD = re.compile(r"[a-z0-9çğıöşü]{3,}", re.I)
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


class ReadUrlArgs(BaseModel):
    url: str = Field(
        ...,
        description=(
            "Okunacak adres, http veya https. Ornek: "
            "'https://www.borsaistanbul.com/dosyalar/kmtp/veriler/kmp_au.pdf'."
        ),
    )
    goal: str | None = Field(
        None,
        description=(
            "Bu adreste ne aradigin, kisa: 'altin islem hacmi tablosu'. Belgenin "
            "metninden bu konuya en yakin bolum ozete konur."
        ),
    )


def _fold(text: str) -> str:
    return str(text).replace("İ", "i").replace("I", "ı").lower().translate(_FOLD)


def excerpt(text: str, goal: str | None, limit: int = EXCERPT_CHARS) -> str:
    """The part of a document the model should read.

    With a `goal`, the lines that share words with it, in the document's own
    order and with the line before and after each -- a number means little
    without the row it sits in. Without one, the opening. Deterministic on
    purpose: a tool that asked a model what mattered would be a second
    opinion inside a step, and the model is already reading this.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if goal:
        wanted = {_fold(word) for word in _WORD.findall(goal)}
        hits = [
            index
            for index, line in enumerate(lines)
            if wanted & {_fold(word) for word in _WORD.findall(line)}
        ]
        if hits:
            keep = sorted({i for hit in hits for i in (hit - 1, hit, hit + 1)})
            chosen = [lines[i] for i in keep if 0 <= i < len(lines)]
            return "\n".join(chosen)[:limit]
    return "\n".join(lines)[:limit]


def _approval_preview(args: BaseModel, _ctx: ToolContext) -> dict[str, Any]:
    return {
        "adres": args.url,
        "amac": args.goal or "belirtilmedi",
        "not": "Adres indirilecek; havuza yazilmaz, oturuma tablo olarak girer.",
    }


def make_read_url() -> ToolSpec:
    """The tool the model calls with an address."""

    def run(args: ReadUrlArgs, ctx: ToolContext) -> ToolResult:
        try:
            document = fetch(args.url, root=ctx.settings.adhoc_dir / "url")
        except WebFetchError as exc:
            return ToolResult(ok=False, summary=f"Adres okunamadi: {exc}")

        def ocr(images: list[bytes]) -> str:
            return ctx.llm.ocr(images, tag="read_url").text

        try:
            found = extractor.extract(
                document.body,
                content_type=document.content_type,
                url=document.final_url,
                ocr=ocr,
            )
        except Exception as exc:  # noqa: BLE001 - a broken document is an answer
            logger.warning("read_url: %s okunamadi: %s", args.url, exc)
            return ToolResult(
                ok=False,
                summary=(
                    f"Adres indirildi ama icerigi okunamadi ({type(exc).__name__}). "
                    "Baska bir bicimde (PDF, Excel, CSV) bir adres varsa onu dene."
                ),
            )

        warnings = list(found.notes)
        if document.from_cache:
            warnings.append("Adres daha once okunmustu; onbellekteki kopya kullanildi.")
        if found.skipped_tables:
            warnings.append(
                f"Belgede {len(found.tables) + found.skipped_tables} tablo var; "
                f"ilk {len(found.tables)} tanesi alindi."
            )

        built: list[Built] = []
        for table in found.tables:
            try:
                frame, quality = ctx.lake.prepare_table(table.frame)
            except Exception as exc:  # noqa: BLE001 - one bad table is not the lot
                warnings.append(f"'{table.name}' hazirlanamadi ({type(exc).__name__}).")
                continue
            if frame.empty:
                continue
            ref = ctx.new_table_ref("read_url")
            ctx.put_table(ref, frame)
            built.append(Built(ref=ref, name=table.name, frame=frame, quality=quality))

        source = Source(
            url=document.final_url,
            name=found.title or document.filename or document.final_url,
            fetched_at=document.fetched_at,
        )

        if not built:
            return ToolResult(
                ok=True,
                summary=_no_table_summary(args, found, document),
                sources=[source],
                warnings=warnings,
            )

        active = max(built, key=lambda item: item.frame.size)
        ctx.set_active(active.ref)
        return ToolResult(
            ok=True,
            summary=_summary(args, found, document, built, active),
            table_ref=active.ref,
            # Ten from each end: a document's table is usually short (the
            # gold bulletin is twelve months) and then the model sees all of
            # it -- and an answer's numbers have to appear in a tool's output
            # (agent/verifier.py), so what is not previewed cannot be quoted.
            preview=preview_rows(active.frame, rows=10),
            sources=[source],
            warnings=warnings,
            quality=active.quality.model_copy(update={"table_ref": active.ref}),
        )

    return ToolSpec(
        name="read_url",
        description_tr=(
            "Verilen web adresini okur: PDF, Excel, CSV, JSON, gorsel ya da sayfa. "
            "Icindeki veri tablolarini temizleyip oturuma tablo olarak ekler, "
            "metninden ozet dondurur. Havuza yazmaz."
        ),
        args_model=ReadUrlArgs,
        fn=run,
        brief_name="Web URL",
        approval="fetch",
        approval_preview=_approval_preview,
    )


def _no_table_summary(args: ReadUrlArgs, found, document) -> str:  # noqa: ANN001
    """What to say when the document held no data table: what it was, what it
    said, and -- for a page that is a list of documents -- what to open."""
    parts = [
        f"{document.final_url} okundu ({found.kind}); veri tablosu cikmadi.",
    ]
    if found.links:
        listed = ", ".join(
            f"{text or 'baglanti'}: {url}" for url, text in found.links[:8]
        )
        parts.append(f"Sayfadaki veri dosyalari -> {listed}")
    text = excerpt(found.text, args.goal)
    if text:
        parts.append(f"Belgeden: {text}")
    return "\n".join(parts)


def _summary(args: ReadUrlArgs, found, document, built, active) -> str:  # noqa: ANN001
    lines = [
        f"{document.final_url} okundu ({found.kind})"
        + (f": {found.title}" if found.title else "")
        + f". {len(built)} tablo hazirlandi, aktif tablo {active.ref}."
    ]
    for table in built:
        columns = ", ".join(str(column) for column in table.frame.columns[:6])
        lines.append(
            f"- {table.ref} ({table.name}): {table.frame.shape[0]} satir x "
            f"{table.frame.shape[1]} sutun"
            + (f"; sutunlar: {columns}" if columns else "")
        )
    text = excerpt(found.text, args.goal, limit=600)
    if text:
        lines.append(f"Belgeden: {text}")
    return "\n".join(lines)
