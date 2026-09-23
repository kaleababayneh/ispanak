"""The browser interface: `uv run chainlit run app.py`.

Deliberately at the repository root rather than inside the package. The engine
promises it does not know about its interface, and
`tests/test_import_discipline.py` enforces that by refusing a `chainlit`
import anywhere under `src/lakehouse_agent/` with no exemption at all. Keeping
this file outside that tree leaves the rule untouched, and `chainlit run
app.py` is Chainlit's own way of being started.

The price is that pytest cannot import this file, so nothing here decides
anything: every choice is made in `lakehouse_agent.agent.webui`, which is
tested, and this file is the shell that calls `cl.*` with what it returns.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chainlit as cl
import plotly.graph_objects as go
from dotenv import find_dotenv, load_dotenv

from lakehouse_agent.agent import webui
from lakehouse_agent.agent.events import Event
from lakehouse_agent.agent.loop import AgentError, real_deps, run_turn
from lakehouse_agent.agent.state import ApprovalDecision, ApprovalRequest, SessionState
from lakehouse_agent.tools.lake_adapter import LakeAdapter

# Before the adapter, not after: it reads the configured embedding model once
# at construction to compare with the one the index was built with, and with
# no environment loaded it would read nothing and skip that check quietly.
load_dotenv(find_dotenv(usecwd=True))

#: One adapter for every chat session. It holds the opened pool index -- about
#: half a gigabyte -- so building one per session would reopen it each time.
#: Its lazy initialisation is guarded by a lock for exactly this reason
#: (`LakeAdapter._pool_index`).
LAKE = LakeAdapter()

logger = logging.getLogger(__name__)

#: How long an approval card waits for a person. Longer than Chainlit's 90s
#: default: a jury reads the card, and `SessionState.approval_wait_s` keeps
#: that time out of the turn's budget anyway.
APPROVAL_TIMEOUT_S = 600


class ChainlitSink:
    """Draws the loop's events. The other half of `agent/console.py`."""

    def __init__(self, state: SessionState, log: Any = None) -> None:
        self._state = state
        self._log = log
        self._answer: cl.Message | None = None
        self._steps: dict[int, cl.Step] = {}
        # Tables are drawn once, at `done`, for the reason the terminal does
        # it: a turn that adds four columns fires five updates, and five
        # copies of a sixty-row table bury the answer.
        self._touched: dict[str, set[str]] = {}
        #: The tables the last finished turn built or changed. Read after
        #: the turn to decide whether a fallback chart is about this
        #: question at all (`webui.fallback_table`).
        self.last_touched: set[str] = set()
        #: Every table an update has ever named, so a later update can be
        #: told apart from a first sighting (`webui.note_table`).
        self._known: set[str] = set()
        self._new: set[str] = set()
        #: The tables the last finished turn created; the fallback chart is
        #: for those alone.
        self.last_new: set[str] = set()

    async def emit(self, event: Event) -> None:
        if self._log is not None:
            self._log.write(event.model_dump_json() + "\n")
            self._log.flush()
        draw = getattr(self, f"_draw_{event.type}", None)
        if draw is not None:
            await draw(event.payload)

    # -- the turn ----------------------------------------------------------

    async def _draw_plan(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "").strip()
        if text:
            await cl.Message(content=text, author="plan").send()

    async def _draw_step_start(self, payload: dict[str, Any]) -> None:
        tool = payload.get("tool")
        if not tool:
            return
        step = cl.Step(name=str(tool), type="tool")
        step.input = payload.get("args") or {}
        await step.send()
        self._steps[int(payload.get("step_no") or 0)] = step

    async def _draw_step_end(self, payload: dict[str, Any]) -> None:
        step = self._steps.pop(int(payload.get("step_no") or 0), None)
        if step is None:
            return
        step.name = webui.step_line(payload)
        step.output = str(payload.get("summary") or payload.get("error") or "")
        await step.update()

    async def _draw_table_update(self, payload: dict[str, Any]) -> None:
        ref = str(payload.get("table_ref") or "")
        if not ref:
            return
        removed = webui.removed_note(payload)
        if removed:
            # The quality gate took it away; nothing to draw, and it must
            # not be drawn from an earlier update either.
            self._touched.pop(ref, None)
            self._new.discard(ref)
            await cl.Message(content=removed).send()
            return
        fresh = {str(name) for name in payload.get("new_columns") or []}
        is_new, changed = webui.note_table(self._known, ref, fresh)
        if is_new:
            self._new.add(ref)
        if changed:
            self._touched.setdefault(ref, set()).update(fresh)

    async def _draw_done(self, _payload: dict[str, Any]) -> None:
        """The turn is over: the tables it touched, each once, in full.

        `docs/contracts.md` section 4: the model sees a summary, the interface
        gets the frame itself. `cl.Dataframe` paginates, so there is no reason
        to cut rows the way the terminal has to.
        """
        touched, self._touched = self._touched, {}
        self.last_touched = set(touched)
        self.last_new, self._new = self._new, set()
        for ref, fresh in touched.items():
            frame = self._state.tables.get(ref)
            if frame is None:
                continue
            shown = webui.display_frame(
                frame,
                ref,
                self._state.lineage,
                webui.chart_labels(self._state.charts, ref),
            )
            labels = dict(zip(frame.columns, shown.columns, strict=True))
            added = ", ".join(labels.get(name, name) for name in sorted(fresh))
            note = f" — yeni: {added}" if fresh else ""
            await cl.Message(
                content=f"**{ref}** · {len(frame)} satır × {len(frame.columns)} sütun{note}",
                elements=[cl.Dataframe(data=shown, display="inline", name=ref)],
            ).send()

    async def _draw_chart(self, payload: dict[str, Any]) -> None:
        figure = payload.get("plotly_json")
        if figure:
            await _send_figure(str(payload.get("chart_ref") or "grafik"), figure)

    async def _draw_answer_delta(self, payload: dict[str, Any]) -> None:
        if self._answer is None:
            self._answer = cl.Message(content="")
            await self._answer.send()
        await self._answer.stream_token(str(payload.get("text") or ""))

    async def _draw_verification(self, payload: dict[str, Any]) -> None:
        lines = webui.flag_lines(payload)
        if lines:
            await cl.Message(
                content="**Doğrulanamayan ifadeler**\n" + "\n".join(lines),
                author="doğrulayıcı",
            ).send()

    async def _draw_sources(self, payload: dict[str, Any]) -> None:
        lines = webui.source_lines(payload)
        if lines:
            await cl.Message(
                content="**Kaynaklar**\n" + "\n".join(lines), author="kaynaklar"
            ).send()

    async def _draw_error(self, payload: dict[str, Any]) -> None:
        user, log = webui.error_message(payload)
        logger.error(log)
        await cl.Message(content=f"**Hata:** {user}").send()

    async def finish(self) -> None:
        """Commit the streamed answer; Chainlit only persists it on update.

        The ids are fenced here, on the whole text, not per token: a token
        boundary can fall inside `evds:TP.KTF12`, and fencing halves would
        leave the join unprotected.
        """
        if self._answer is not None:
            self._answer.content = webui.protect_series_ids(self._answer.content)
            await self._answer.update()
            self._answer = None


async def _send_figure(name: str, figure: dict[str, Any]) -> None:
    """`cl.Plotly` wants a real `Figure`; the engine only ever built a dict."""
    await cl.Message(
        content="",
        elements=[cl.Plotly(name=name, figure=go.Figure(figure), display="inline")],
    ).send()


async def _approve(request: ApprovalRequest) -> ApprovalDecision:
    """Ask a person, with the card's own options as buttons."""
    response = await cl.AskActionMessage(
        content=f"**{request.question}**",
        actions=[cl.Action(**spec) for spec in webui.action_specs(request)],
        timeout=APPROVAL_TIMEOUT_S,
    ).send()
    value = ((response or {}).get("payload") or {}).get("value")
    return webui.decision(request, value)


@cl.on_chat_start
async def start() -> None:
    session_id = str(cl.user_session.get("id") or "web")
    state = SessionState(session_id=session_id)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    try:
        deps = real_deps(lake=LAKE, sink=None)
    except Exception:
        # A missing key is the usual cause. Without this the tab opens onto
        # a silent page; the log has the detail, the person gets a sentence.
        logger.exception("agent baslatilamadi")
        await cl.Message(
            content="Agent başlatılamadı; sunucu logunu kontrol edin."
        ).send()
        return
    path = (
        Path(deps.settings.adhoc_dir) / "sessions" / f"{stamp}-{session_id[:8]}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.open("a", encoding="utf-8")

    deps.sink = ChainlitSink(state, log)
    deps.approve = _approve

    cl.user_session.set("state", state)
    cl.user_session.set("deps", deps)
    cl.user_session.set("log", log)
    cl.user_session.set("turn_running", False)


@cl.on_message
async def message(msg: cl.Message) -> None:
    # The browser drew this bubble before this handler ran, bare ids and all;
    # `update()` re-sends it and the browser replaces it in place.
    protected = webui.protect_series_ids(msg.content)
    if protected != msg.content:
        msg.content = protected
        await msg.update()

    state: SessionState | None = cl.user_session.get("state")
    deps = cl.user_session.get("deps")
    if state is None or deps is None:
        await cl.Message(content="Oturum hazır değil; sayfayı yenileyin.").send()
        return
    if cl.user_session.get("turn_running"):
        # One `SessionState` per tab and `run_turn` mutates it; a second
        # turn on it at the same time would interleave two histories.
        await cl.Message(content="Önceki soru hâlâ işleniyor, lütfen bekleyin.").send()
        return

    sink: ChainlitSink = deps.sink
    charts_before = len(state.charts)
    cl.user_session.set("turn_running", True)
    try:
        result = await run_turn(state, deps, msg.content)
    except AgentError as exc:
        await sink.finish()
        await cl.Message(content=f"**Tur tamamlanamadı:** {exc}").send()
        return
    finally:
        cl.user_session.set("turn_running", False)

    await sink.finish()
    if not result.answer:
        await cl.Message(
            content="Bu tur cevapsız bitti. " + (result.truncated_reason or "")
        ).send()
    ref = webui.fallback_table(
        result.active_table, sink.last_new, len(state.charts) - charts_before
    )
    if ref is not None:
        await _chart_fallback(state, ref)


async def _chart_fallback(state: SessionState, ref: str) -> None:
    """Draw the obvious chart when the model did not ask for one this turn.

    `make_chart` costs a step and a turn has twelve, so a long analysis can
    spend them all and end table-only. The fallback is marked in its title:
    it is the interface's picture of a table this turn touched, not
    something the agent chose to show.
    """
    frame = state.tables.get(ref)
    if frame is None:
        return
    spec = webui.default_chart(
        frame,
        ref,
        state.charts,
        lineage=state.lineage,
        overrides=webui.chart_labels(state.charts, ref),
    )
    if spec is None or not spec.plotly_json:
        return
    state.charts.append(spec)
    await _send_figure(spec.chart_ref, spec.plotly_json)


@cl.on_chat_end
async def end() -> None:
    log = cl.user_session.get("log")
    if log is not None:
        log.close()
