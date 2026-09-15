"""The event channel between the engine and whatever draws it.

Two things matter here and nothing else does. Every payload has to survive
`json.dumps`, because the headless runner writes them one per line and the
interface sends them over a socket. And the list of event names has to match
the one written down in docs/contracts.md, because that table is what the data
lane and the interface build against.
"""

import asyncio
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from pydantic_core import PydanticSerializationError

from lakehouse_agent.agent.events import EVENT_TYPES, Event, ListSink, emit

DOCS = Path(__file__).resolve().parents[1] / "docs" / "contracts.md"


def test_documented_event_names_match_the_code():
    """The drift guard.

    docs/contracts.md is what the other lane implements against. If someone
    adds an event in code and forgets the table, the interface silently never
    handles it; if someone edits the table and not the code, they build against
    something that never fires. Neither shows up until integration day.
    """
    text = DOCS.read_text(encoding="utf-8")
    documented = set(re.findall(r"`(" + "|".join(EVENT_TYPES) + r")`", text))
    missing = sorted(set(EVENT_TYPES) - documented)
    assert missing == [], (
        f"docs/contracts.md'de gecmeyen olay turleri: {missing}. "
        "Belge ile kod ayrisirsa arayuz olmayan bir olaya gore yazilir."
    )


def test_emit_without_a_sink_is_a_no_op():
    """The loop runs headless in tests and in the eval runner."""
    event = asyncio.run(emit(None, "s1", "plan", text="merhaba"))
    assert event.type == "plan"
    assert event.payload == {"text": "merhaba"}


def test_events_arrive_in_the_order_they_were_emitted():
    """An async sink was chosen so ordering needs no queue; this pins that."""
    sink = ListSink()

    async def scenario():
        await emit(sink, "s1", "plan", text="a")
        await emit(sink, "s1", "step_start", step_no=1)
        await emit(sink, "s1", "step_end", step_no=1, ok=True)
        await emit(sink, "s1", "done")

    asyncio.run(scenario())
    assert sink.types() == ["plan", "step_start", "step_end", "done"]
    assert sink.last("step_end").payload["ok"] is True
    assert sink.of_type("missing") == []
    assert sink.last("missing") is None


def test_every_payload_survives_json_dumps():
    """The headless runner writes one JSON line per event; a DataFrame in a
    payload would kill the run at the worst possible moment."""
    sink = ListSink()
    frame = pd.DataFrame({"period": pd.date_range("2021-01-01", periods=2, freq="MS")})

    async def scenario():
        from lakehouse_agent.agent.formatting import preview_rows

        await emit(
            sink, "s1", "table_update", table_ref="t01_x", rows=preview_rows(frame)
        )

    asyncio.run(scenario())
    line = json.dumps(sink.events[0].model_dump(mode="json"), ensure_ascii=False)
    assert "2021-01-01" in line


def test_a_dataframe_in_a_payload_is_caught_not_shipped():
    """Deliberately the failure mode above, so the rule is visible.

    It fails at `model_dump`, before a single byte is written, which is the
    behaviour we want: a table reaches the interface as a preview plus a
    reference it can fetch, never as megabytes on a progress channel.
    """
    event = Event(
        type="table_update",
        ts=pd.Timestamp("2026-09-15T00:00:00Z").to_pydatetime(),
        session_id="s1",
        payload={"frame": pd.DataFrame({"a": [1]})},
    )
    with pytest.raises(PydanticSerializationError):
        json.dumps(event.model_dump(mode="json"))
