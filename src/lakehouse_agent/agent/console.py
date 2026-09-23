"""The terminal interface: it draws the events and asks the questions.

The one file allowed to import `rich`. The engine publishes events and awaits
a callback; what any of that looks like on a screen is decided here and
nowhere else (`tests/test_import_discipline.py`). `loop.py` imports this
lazily, only when `--chat` is given, so a headless run never loads it.

Three seams, all on one object so they can share a console and an input
stream, and all injectable so the tests drive them without a terminal:

* `emit` satisfies `EventSink` (`agent/events.py`) -- the screen half;
* `approve` satisfies `ApproveFn` (`agent/loop.py`) -- the approval card;
* `ask` returns the next question, or `None` when the session is over.

Everything a model or a page wrote is printed with `markup=False`: rich reads
`[...]` as markup, and an answer citing `TP.KTF12 [aylik]` would lose the
bracket, or worse, be coloured by text we did not write.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lakehouse_agent.agent.formatting import json_safe, preview_rows
from lakehouse_agent.agent.labels import table_labels
from lakehouse_agent.agent.webui import chart_labels, note_table

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from lakehouse_agent.agent.events import Event
    from lakehouse_agent.agent.state import ApprovalDecision, ApprovalRequest

#: Rows drawn before a table is cut down to its two ends. A question asking
#: for sixty months must show sixty rows -- the screen is where a person
#: checks the data, and `docs/contracts.md` section 4 says the full table
#: goes to the interface for exactly this reason. The ceiling is only a guard
#: against a tool that built forty thousand rows.
TABLE_ROWS = 200
#: Decimals kept when a float is drawn. Binary arithmetic turns a monthly
#: difference into 0.4029999999999999, and a person reading that on a screen
#: cannot tell noise from precision. `tools/lakehouse._preview` rounds the
#: model's copy for the same reason.
DECIMALS = 4
#: Sources listed under a turn, matching the headless recap's own limit.
SOURCES_SHOWN = 12


class Terminal:
    """One screen and one keyboard, wired to the loop's three seams."""

    def __init__(
        self,
        console: Console | None = None,
        stream: Any = None,
        *,
        tables: Callable[[str], Any] | None = None,
        lineage: Mapping[str, Any] | None = None,
        charts: Iterable[Any] | None = None,
        quiet: bool = False,
    ) -> None:
        self.console = console or Console(stderr=True)
        #: Where a question or an approval is read from. `None` means the
        #: real stdin; the tests pass a `StringIO`.
        self.stream = stream
        #: The table a turn built, drawn when the turn ends rather than after
        #: every column added to it.
        self._pending_table = ""
        self._fresh: set[str] = set()
        #: Tables already announced, so a follow-up that only reads one
        #: (`top`) does not print its sixty rows again (`webui.note_table`).
        self._known: set[str] = set()
        #: Resolves a table reference to the whole frame. The event only
        #: carries a preview; `docs/contracts.md` section 4 keeps the full
        #: table in the session store for the interface to read.
        self._tables = tables
        #: The session's column provenance, read to put a person's label
        #: over each column name. The name stays: it is what the next
        #: question has to use.
        self._lineage = lineage if lineage is not None else {}
        #: The session's charts, read live: the model's word for a column
        #: on the legend is the word the table header uses too.
        self._charts = charts if charts is not None else ()
        #: True while an answer is streaming, so the next event starts on a
        #: fresh line instead of running into it.
        self._streaming = False
        self._quiet = quiet

    # -- the screen --------------------------------------------------------

    async def emit(self, event: Event) -> None:
        """Draw one event. `docs/contracts.md` section 6 is the mapping.

        An unknown type is ignored rather than raised on: the event list
        grows, and a demo must not die because the interface has not caught
        up with it yet.
        """
        payload = event.payload
        drawer = getattr(self, f"_draw_{event.type}", None)
        if drawer is None:
            return
        if event.type != "answer_delta":
            self._end_stream()
        drawer(payload)

    def _end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    def _say(self, text: str, **style: Any) -> None:
        """Print text we did not write, with markup off."""
        self.console.print(text, markup=False, highlight=False, **style)

    def _draw_plan(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "").strip()
        if text:
            self._say(text, style="dim")

    def _draw_step_start(self, payload: dict[str, Any]) -> None:
        tool = payload.get("tool")
        if not tool:
            return
        args = ", ".join(f"{k}={v}" for k, v in (payload.get("args") or {}).items())
        self._say(f"  > {tool}({args[:90]})", style="cyan")

    def _draw_step_end(self, payload: dict[str, Any]) -> None:
        """One line per finished step.

        Keyed on `node`, not on `tool`: `step_end` carries no tool name
        (`docs/contracts.md` section 6), which is why an earlier version of
        this drew nothing at all. The node is what a person wants anyway --
        it says whether the agent is planning, calling a tool, or checking
        its own answer.
        """
        ok = payload.get("ok") is not False
        mark = "ok  " if ok else "HATA"
        summary = str(payload.get("summary") or "").splitlines()[:1]
        self._say(
            f"    {mark} {str(payload.get('node') or ''):<8} "
            f"{payload.get('duration_ms', 0):>6} ms  "
            f"{(summary[0] if summary else '')[:80]}",
            style="green" if ok else "red",
        )

    def _draw_table_update(self, payload: dict[str, Any]) -> None:
        """A line while the table is being built; the table itself at the end.

        Drawn in full on every update, a turn that adds four columns to sixty
        months printed the same table five times, some three hundred lines,
        and the answer scrolled off the top. The person wants the finished
        table, not each intermediate. So an update notes what changed and the
        reference is remembered; `done` draws it once.
        """
        ref = str(payload.get("table_ref") or "")
        fresh = [str(name) for name in payload.get("new_columns") or []]
        if self._tables is None:
            # No store to read from at the end, so the event's own preview is
            # all there will ever be: drawn now rather than lost.
            self._draw_preview(payload)
            return
        is_new, changed = note_table(self._known, ref, fresh)
        if not changed:
            return
        self._pending_table = ref or self._pending_table
        self._fresh.update(fresh)
        if is_new:
            self._say(f"  tablo {ref} kuruldu", style="dim")
        else:
            self._say(f"  tablo {ref}: {', '.join(fresh)} eklendi", style="dim")

    def _draw_done(self, _payload: dict[str, Any]) -> None:
        """The turn is over: draw the table it ended with, once."""
        ref, self._pending_table = self._pending_table, ""
        fresh, self._fresh = set(self._fresh), set()
        if ref:
            self._draw_table(ref, fresh)

    def _draw_preview(self, payload: dict[str, Any]) -> None:
        """The event's ten rows, for a caller that wired no table store."""
        columns = [str(name) for name in payload.get("columns") or []]
        rows = list(payload.get("rows") or [])
        if not columns:
            return
        fresh = set(payload.get("new_columns") or [])
        table = Table(title=str(payload.get("table_ref") or ""), title_style="bold")
        for name in columns:
            table.add_column(
                name, style="yellow" if name in fresh else None, overflow="fold"
            )
        for row in rows:
            table.add_row(*[self._cell(row.get(name)) for name in columns])
        self.console.print(table)

    def _draw_table(self, ref: str, fresh: set[str]) -> None:
        """The table with its values, in full.

        The event carries only a preview -- ten rows, enough for the model.
        A person needs the table they asked for: sixty months of a sixty
        month question. `docs/contracts.md` section 4 keeps the whole frame
        in the session store precisely so the interface can draw it, so that
        is where this reads from, falling back to the preview when no store
        was wired (an older caller, or a test).
        """
        frame = self._tables(ref) if self._tables else None
        if frame is None:
            return
        else:
            columns = [str(name) for name in frame.columns]
            total = len(frame)
            shown = frame if total <= TABLE_ROWS else None
            rows = (
                [
                    {str(k): json_safe(v) for k, v in row.items()}
                    for row in shown.to_dict("records")
                ]
                if shown is not None
                else preview_rows(frame, rows=TABLE_ROWS // 2)
            )
        if not columns:
            return
        table = Table(title=ref, title_style="bold")
        labels = table_labels(
            ref, columns, self._lineage, chart_labels(self._charts, ref)
        )
        for name in columns:
            # Folded, not truncated: a generated name like
            # `aylik_tuketici_tuketici_kredileri_konut_toplam_sektor` is
            # unreadable cut off, and it is the name the next question has to
            # use. The label sits above it when the lineage knows one.
            header = (
                f"{labels[name]}\n[dim]{name}[/dim]" if labels[name] != name else name
            )
            table.add_column(
                header, style="yellow" if name in fresh else None, overflow="fold"
            )
        for row in rows:
            table.add_row(*[self._cell(row.get(name)) for name in columns])
        self.console.print(table)
        if len(rows) < total:
            self._say(
                f"  {total} satirin ilk ve son {TABLE_ROWS // 2}'i gosteriliyor",
                style="dim",
            )

    @staticmethod
    def _cell(value: Any) -> str:
        """One cell as text, floats rounded so the screen shows no noise."""
        if value is None:
            return ""
        if isinstance(value, float):
            if value != value:  # NaN
                return ""
            return f"{round(value, DECIMALS):g}"
        return str(value)

    def _draw_quality_report(self, payload: dict[str, Any]) -> None:
        bits = [f"{payload.get('rows', '?')} satir"]
        for key, label in (("missing_pct", "eksik %"), ("duplicates", "tekrar")):
            value = payload.get(key)
            if value:
                bits.append(f"{label} {value}")
        if payload.get("unit_guess") is None:
            bits.append("birim bilinmiyor")
        self._say("  kalite: " + ", ".join(str(bit) for bit in bits), style="dim")

    def _draw_answer_delta(self, payload: dict[str, Any]) -> None:
        self.console.print(
            str(payload.get("text") or ""),
            end="",
            markup=False,
            highlight=False,
        )
        self._streaming = True

    def _draw_verification(self, payload: dict[str, Any]) -> None:
        flags = payload.get("flags") or []
        if not flags:
            return
        lines = "\n".join(f"- {flag.get('text', '')}" for flag in flags)
        self.console.print(
            Panel(lines, title="Dogrulanamayan ifadeler", border_style="yellow"),
            markup=False,
            highlight=False,
        )

    def _draw_sources(self, payload: dict[str, Any]) -> None:
        sources = payload.get("sources") or []
        if not sources:
            return
        self._say("  KAYNAKLAR (bakilan)", style="dim")
        for source in sources[:SOURCES_SHOWN]:
            name = source.get("name") or ""
            self._say(f"  - {source.get('series_id') or source.get('url')}  {name}")
        more = len(sources) - SOURCES_SHOWN
        if more > 0:
            self._say(f"  ... ve {more} kaynak daha", style="dim")

    def _draw_chart(self, payload: dict[str, Any]) -> None:
        """A terminal cannot draw it; say that it exists and where it shows."""
        ref = payload.get("chart_ref") or "?"
        traces = len((payload.get("plotly_json") or {}).get("data") or [])
        self._say(
            f"  grafik {ref} hazir ({traces} seri) -- tarayici arayuzunde cizilir",
            style="dim",
        )

    def _draw_approval_decision(self, payload: dict[str, Any]) -> None:
        how = "otomatik" if payload.get("automatic") else "secildi"
        self._say(f"  onay [{payload.get('kind')}]: {payload.get('choice')} ({how})")

    def _draw_error(self, payload: dict[str, Any]) -> None:
        self._say(f"  HATA: {payload.get('message', '')}", style="red")

    # -- the keyboard ------------------------------------------------------

    def _read(self, prompt: str) -> str | None:
        """One line, or `None` at end of input.

        A real terminal raises `EOFError` on Ctrl-D; a stream just runs out
        and returns "". Both mean the same thing here.
        """
        try:
            line = self.console.input(prompt, stream=self.stream)
        except EOFError:
            return None
        if self.stream is not None:
            # A real terminal echoes the Enter that ended the line; a stream
            # does not, and without this the next thing drawn runs into the
            # prompt.
            self.console.print()
        return line if line else None

    def _read_question(self) -> str | None:
        """One question, however many lines it was pasted as.

        A question is a paragraph and a blank line ends it. `None` -- the end
        of input, Ctrl-D -- ends the session.

        Deliberately not clever. The first version stopped at the first line
        unless the terminal already held more, which read a pasted paragraph
        as several questions: measured 23 Sep, the brief's three-line
        scenario ran three turns, the second on the fragment "gosteriniz.
        Buna ek olarak ...". A paste arrives in chunks, so "is there more
        waiting" is a race with no safe answer -- a settling window of 150 ms
        covered a 120 ms gap and nothing covers a slow link. A blank line is
        the person saying they are done, and it cannot be raced.
        """
        parts: list[str] = []
        while True:
            line = self._read("\n[bold]soru>[/bold] " if not parts else "")
            if line is None:
                # Ctrl-D: ends the question if one is half-written, else the
                # session.
                break
            text = line.strip()
            if not text:
                break
            parts.append(text)
        return " ".join(parts).strip() or None

    async def ask(self) -> str | None:
        """The next question, or `None` to end the session.

        Read in a thread: the loop is async and `input` blocks, which is what
        `asyncio.to_thread` is already used for elsewhere in the loop.
        """
        self._end_stream()
        return await asyncio.to_thread(self._read_question)

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        """Draw the card, read the choice.

        Whatever comes back is validated by `approvals.settle` against the
        card, so a typo cannot pick an option that is not there -- it falls
        to the default and is marked automatic. That is why this can read a
        number without checking it against the option list itself.
        """
        from lakehouse_agent.agent.state import ApprovalDecision

        self._end_stream()
        lines = [request.question, ""]
        for number, option in enumerate(request.options, start=1):
            mark = " (varsayilan)" if option.id == request.default else ""
            lines.append(f"  {number}. {option.label}{mark}")
            if option.detail:
                lines.append(f"     {option.detail}")
        self.console.print(
            Panel(
                "\n".join(lines), title=f"ONAY [{request.kind}]", border_style="cyan"
            ),
            markup=False,
            highlight=False,
        )
        chosen = await asyncio.to_thread(self._choose, request)
        return ApprovalDecision(
            request_id=request.request_id, choice=chosen, note="terminalden secildi"
        )

    @staticmethod
    def _default_number(request: ApprovalRequest) -> str:
        """Which number Enter picks. "bos=varsayilan" made a person look for
        the word rather than the option."""
        for number, option in enumerate(request.options, start=1):
            if option.id == request.default:
                return str(number)
        return "1"

    def _choose(self, request: ApprovalRequest) -> str:
        """Block until a number is typed. Empty input takes the default."""
        while True:
            line = self._read(
                f"  secim [1-{len(request.options)}, bos={self._default_number(request)}]> "
            )
            if line is None:
                return request.default or ""
            answer = line.strip()
            if not answer:
                return request.default or ""
            if answer.isdigit() and 1 <= int(answer) <= len(request.options):
                return request.options[int(answer) - 1].id
            self._say("  Gecerli bir numara yaz.", style="red")
