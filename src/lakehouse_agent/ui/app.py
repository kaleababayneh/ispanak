"""The Chainlit adapter: draws the agent's event stream and asks approval
cards in a browser instead of a terminal.

The second consumer of the contract `agent/console.py`'s `Terminal` already
proved: `EventSink.emit` for the screen, `AgentDeps.approve` for the approval
card. No engine file changes for this to exist, and this module can be
deleted without the engine noticing.

State lives per chat session, in `cl.user_session` (`on_chat_start`), not at
module level -- a module-level `SessionState` would let one browser tab's
tables and history leak into another's.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any

import chainlit as cl

from lakehouse_agent.agent.events import Event
from lakehouse_agent.agent.loop import AgentDeps, run_turn
from lakehouse_agent.agent.state import (
    ApprovalDecision,
    ApprovalRequest,
    SessionState,
    initial_state,
)
from lakehouse_agent.ui.render import (
    render_approval_card,
    render_approval_decision,
    render_chart,
    render_error,
    render_sources,
    render_step_end,
    render_step_start,
    render_table_note,
    render_verification,
)

logger = logging.getLogger(__name__)

#: Seconds an approval card waits before falling back to its default, the
#: same outcome `agent/console.py._choose` gives an empty Enter.
APPROVAL_TIMEOUT_S = 120


def _build_deps() -> AgentDeps:
    """Real model, real tools, real data -- what the demo runs.

    Mirrors `agent/loop._real_deps` rather than importing it: that function
    is a module-private detail of the CLI entrypoint, and keeping the wiring
    local here means this adapter's `sink` and `approve` can change without
    touching `loop.py`, and `loop.py` can change its CLI wiring without
    touching this file.
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    from lakehouse_agent.config import get_agent_settings
    from lakehouse_agent.llm import MiaLLM
    from lakehouse_agent.tools.lake_adapter import LakeAdapter
    from lakehouse_agent.tools.toolset import make_registry

    settings = get_agent_settings()
    return AgentDeps(
        llm=MiaLLM(settings),
        registry=make_registry(),
        settings=settings,
        lake=LakeAdapter(),
    )


class ChainlitSink:
    """Draws one session's events on its Chainlit page.

    Mirrors `agent/console.py`'s `Terminal`: a table is only recorded on
    `table_update`, and drawn in full exactly once, when the turn's `done`
    arrives -- read from the session's own table store rather than the
    event's preview (`docs/contracts.md` section 4). An event type this
    class has no `_draw_*` for is ignored, same dispatch as `Terminal.emit`.
    """

    def __init__(self, state: SessionState) -> None:
        self._state = state
        self._steps: dict[int, cl.Step] = {}
        self._pending_table = ""
        self._fresh: set[str] = set()
        self._pending_sources: dict[str, Any] | None = None
        self._pending_verification: dict[str, Any] | None = None
        self._answer_msg: cl.Message | None = None

    async def emit(self, event: Event) -> None:
        handler = getattr(self, f"_draw_{event.type}", None)
        if handler is None:
            return
        await handler(event.payload)

    async def _draw_step_start(self, payload: dict[str, Any]) -> None:
        view = render_step_start(payload)
        step = cl.Step(name=view["label"], type="tool")
        step.input = view["input_text"]
        await step.send()
        step_no = payload.get("step_no")
        if step_no is not None:
            self._steps[int(step_no)] = step

    async def _draw_step_end(self, payload: dict[str, Any]) -> None:
        step_no = payload.get("step_no")
        step = self._steps.pop(int(step_no), None) if step_no is not None else None
        if step is None:
            return
        view = render_step_end(payload)
        step.output = view["output_text"]
        step.is_error = view["is_error"]
        if view["is_error"] and view["error_text"]:
            step.output = f"{view['output_text']} ({view['error_text']})".strip()
        await step.update()

    async def _draw_table_update(self, payload: dict[str, Any]) -> None:
        if payload.get("removed"):
            self._pending_table = ""
            self._fresh.clear()
            return
        ref = str(payload.get("table_ref") or "")
        self._pending_table = ref or self._pending_table
        self._fresh.update(str(name) for name in payload.get("new_columns") or [])

    async def _draw_answer_delta(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "")
        if self._answer_msg is None:
            self._answer_msg = cl.Message(content="")
        await self._answer_msg.stream_token(text)

    async def _draw_sources(self, payload: dict[str, Any]) -> None:
        self._pending_sources = payload

    async def _draw_verification(self, payload: dict[str, Any]) -> None:
        self._pending_verification = payload

    async def _draw_chart(self, payload: dict[str, Any]) -> None:
        chart = render_chart(payload)
        if chart is None:
            return
        try:
            import plotly.graph_objects as go

            figure = go.Figure(chart["plotly_json"])
        except Exception:
            logger.exception("grafik cizilemedi: %s", chart.get("chart_ref"))
            return
        await cl.Message(
            content=str(chart.get("title") or ""),
            elements=[cl.Plotly(name=chart["chart_ref"] or "grafik", figure=figure)],
        ).send()

    async def _draw_approval_decision(self, payload: dict[str, Any]) -> None:
        await cl.Message(content=render_approval_decision(payload)).send()

    async def _draw_error(self, payload: dict[str, Any]) -> None:
        view = render_error(payload)
        logger.error(view["log_message"])
        await cl.Message(content=view["user_message"]).send()

    async def _draw_done(self, _payload: dict[str, Any]) -> None:
        if self._answer_msg is not None:
            await self._answer_msg.send()
            self._answer_msg = None

        ref, self._pending_table = self._pending_table, ""
        fresh, self._fresh = self._fresh, set()
        if ref and ref in self._state.tables:
            note = render_table_note({"table_ref": ref, "new_columns": sorted(fresh)})
            frame = self._state.tables[ref]
            await cl.Message(
                content=note,
                elements=[cl.Dataframe(data=frame, name=ref, display="inline")],
            ).send()

        sources_text = render_sources(self._pending_sources)
        self._pending_sources = None
        if sources_text:
            await cl.Message(content=sources_text).send()

        verification_text = render_verification(self._pending_verification)
        self._pending_verification = None
        if verification_text:
            await cl.Message(content=verification_text).send()


async def _chainlit_approve(request: ApprovalRequest) -> ApprovalDecision:
    """The approval card, as an `cl.AskActionMessage`.

    A timed-out card -- nobody answers within `APPROVAL_TIMEOUT_S` -- takes
    the card's default, the same outcome `agent/console.py._choose` gives an
    empty Enter or end of input.
    """
    card = render_approval_card(request.model_dump(mode="json"))
    actions = [
        cl.Action(
            name=option["id"],
            payload={"choice": option["id"]},
            label=option["label"] + (" (varsayilan)" if option["is_default"] else ""),
            tooltip=option["detail"] or "",
        )
        for option in card["options"]
    ]
    response = await cl.AskActionMessage(
        content=card["question"], actions=actions, timeout=APPROVAL_TIMEOUT_S
    ).send()
    if response is None:
        return ApprovalDecision(
            request_id=request.request_id,
            choice=request.default or "",
            note="chainlit zaman asimi: varsayilan secildi",
        )
    choice = response.get("name") or (response.get("payload") or {}).get("choice")
    return ApprovalDecision(
        request_id=request.request_id,
        choice=str(choice or request.default or ""),
        note="chainlit'te secildi",
    )


@cl.password_auth_callback
async def _auth(username: str, password: str) -> cl.User | None:
    """Compares against `CHAINLIT_USERNAME`/`CHAINLIT_PASSWORD` in constant
    time, read here rather than at import so a missing `.env` does not stop
    the module from loading."""
    expected_user = os.environ.get("CHAINLIT_USERNAME", "")
    expected_pass = os.environ.get("CHAINLIT_PASSWORD", "")
    if not expected_user or not expected_pass:
        return None
    if hmac.compare_digest(username, expected_user) and hmac.compare_digest(
        password, expected_pass
    ):
        return cl.User(identifier=username)
    return None


@cl.on_chat_start
async def _on_chat_start() -> None:
    session_id = str(cl.user_session.get("id") or "chainlit")
    state = initial_state(session_id)
    try:
        deps = _build_deps()
    except Exception:
        logger.exception("agent baglantisi kurulamadi")
        await cl.Message(
            content="Agent baslatilamadi. Sunucu logunu kontrol edin."
        ).send()
        return

    deps.sink = ChainlitSink(state)
    deps.approve = _chainlit_approve
    cl.user_session.set("state", state)
    cl.user_session.set("deps", deps)
    cl.user_session.set("turn_running", False)


@cl.on_message
async def _on_message(message: cl.Message) -> None:
    state: SessionState | None = cl.user_session.get("state")
    deps: AgentDeps | None = cl.user_session.get("deps")
    if state is None or deps is None:
        await cl.Message(content="Oturum hazir degil. Sayfayi yenileyin.").send()
        return

    if cl.user_session.get("turn_running"):
        await cl.Message(content="Onceki soru hala isleniyor, lutfen bekleyin.").send()
        return

    cl.user_session.set("turn_running", True)
    try:
        await run_turn(state, deps, message.content)
    except Exception:
        # `run_turn` already publishes an `error` event -- which
        # `ChainlitSink._draw_error` turns into a message -- for every path
        # it raises through. This is only a net for what escapes before that:
        # `_answer_unanswered_calls` runs ahead of `run_turn`'s own try block.
        logger.exception("tur calisirken beklenmeyen hata")
        await cl.Message(
            content="Beklenmeyen bir hata olustu. Lutfen tekrar deneyin."
        ).send()
    finally:
        cl.user_session.set("turn_running", False)
