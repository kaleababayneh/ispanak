"""What the agent tells the outside world while it works.

The loop never draws anything. It publishes events and something else decides
what they look like. That boundary is why the engine can be tested with a list
and demoed in a chat interface without either knowing about the other.

`emit` is async because the consumer is. In a chat interface, showing a step
means awaiting a network send; a synchronous sink would either block the event
loop or need a queue and a second task to drain it. Making the sink async
removes that machinery and keeps events in the order they happened.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol, get_args

from pydantic import BaseModel, Field

EventType = Literal[
    "plan",
    "step_start",
    "step_end",
    "table_update",
    "chart",
    "answer_delta",
    "verification",
    "sources",
    "quality_report",
    "approval_request",
    "approval_decision",
    "done",
    "error",
]

#: The same list, available at runtime so a test can compare it with the table
#: in docs/contracts.md. A written contract nothing checks drifts within days.
#: Read off the `Literal` rather than written out again: two hand-kept copies
#: of one list are a drift guard that can itself drift.
EVENT_TYPES: tuple[str, ...] = get_args(EventType)


class Event(BaseModel):
    """One thing that happened, timestamped and addressed to a session.

    `payload` must be JSON-serialisable. DataFrames never go in: a table
    reaches the interface as a preview plus a reference it can fetch in full.
    Putting a frame here would both break serialisation and quietly send
    megabytes through a channel meant for progress updates.
    """

    type: EventType
    ts: datetime
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EventSink(Protocol):
    """Anything that can receive events. The interface implements one."""

    async def emit(self, event: Event) -> None: ...


class ListSink:
    """Collects events in a list. Used by tests and the headless CLI."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [event.type for event in self.events]

    def of_type(self, type_: str) -> list[Event]:
        return [event for event in self.events if event.type == type_]

    def last(self, type_: str) -> Event | None:
        matches = self.of_type(type_)
        return matches[-1] if matches else None


async def emit(
    sink: EventSink | None,
    session_id: str,
    type_: EventType,
    **payload: Any,
) -> Event:
    """Build and publish one event.

    A `None` sink drops it silently and on purpose: the loop is often run
    headless, in tests and in the evaluation runner, and it should not have to
    care whether anyone is watching.
    """
    event = Event(
        type=type_, ts=datetime.now(UTC), session_id=session_id, payload=payload
    )
    if sink is not None:
        await sink.emit(event)
    return event
