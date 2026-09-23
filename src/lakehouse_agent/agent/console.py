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
import select
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from lakehouse_agent.agent.events import Event
    from lakehouse_agent.agent.state import ApprovalDecision, ApprovalRequest

#: Preview rows drawn per table. The payload already carries `preview_rows`'
#: head-and-tail selection (`agent/formatting.py`); this is the ceiling in
#: case a tool sends more.
TABLE_ROWS = 12
#: Sources listed under a turn, matching the headless recap's own limit.
SOURCES_SHOWN = 12


class Terminal:
    """One screen and one keyboard, wired to the loop's three seams."""

    def __init__(
        self,
        console: Console | None = None,
        stream: Any = None,
        *,
        pending: Callable[[], bool] | None = None,
        quiet: bool = False,
    ) -> None:
        self.console = console or Console(stderr=True)
        #: Where a question or an approval is read from. `None` means the
        #: real stdin; the tests pass a `StringIO`.
        self.stream = stream
        #: Is there more input waiting right now? Injectable because the
        #: real answer needs a file descriptor and a `StringIO` has none.
        self._pending = pending or self._input_waiting
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
        """The table with its values. The headless recap prints a one-line
        description; on a screen the numbers are the point."""
        columns = [str(name) for name in payload.get("columns") or []]
        rows = list(payload.get("rows") or [])
        if not columns:
            return
        fresh = set(payload.get("new_columns") or [])
        table = Table(title=str(payload.get("table_ref") or ""), title_style="bold")
        for name in columns:
            table.add_column(name, style="yellow" if name in fresh else None)
        for row in rows[:TABLE_ROWS]:
            table.add_row(
                *["" if row.get(c) is None else str(row.get(c)) for c in columns]
            )
        self.console.print(table)
        if len(rows) > TABLE_ROWS:
            self._say(f"  ... {len(rows) - TABLE_ROWS} satir daha", style="dim")

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

    def _input_waiting(self) -> bool:
        """Is another line already sitting in the terminal's buffer?

        True only for text that arrived without the person typing it, which
        in practice means a paste. `select` with a zero timeout asks exactly
        that and never blocks. A stream with no file descriptor -- the tests'
        `StringIO` -- answers no, so they stay line by line.
        """
        stream = self.stream if self.stream is not None else sys.stdin
        try:
            return bool(select.select([stream.fileno()], [], [], 0)[0])
        except Exception:  # noqa: BLE001 - no descriptor is simply "nothing waiting"
            return False

    def _read_question(self) -> str | None:
        """One question, however many lines it was pasted as.

        A question is a paragraph, not a line. Measured 23 Sep on the first
        chat run: the brief's own scenario is three lines long, and pasting
        it ran three turns -- the second on the fragment "gösteriniz. Buna ek
        olarak ...". Whatever the terminal already holds when the first line
        lands was pasted with it and belongs to the same question; a blank
        line or the end of input closes it.
        """
        first = self._read("\n[bold]soru>[/bold] ")
        if first is None:
            return None
        parts = [first.strip()]
        while self._pending():
            more = self._read("")
            if more is None or not more.strip():
                break
            parts.append(more.strip())
        return " ".join(part for part in parts if part).strip() or None

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

    def _choose(self, request: ApprovalRequest) -> str:
        """Block until a number is typed. Empty input takes the default."""
        while True:
            line = self._read(f"  secim [1-{len(request.options)}, bos=varsayilan]> ")
            if line is None:
                return request.default or ""
            answer = line.strip()
            if not answer:
                return request.default or ""
            if answer.isdigit() and 1 <= int(answer) <= len(request.options):
                return request.options[int(answer) - 1].id
            self._say("  Gecerli bir numara yaz.", style="red")
