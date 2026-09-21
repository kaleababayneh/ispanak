"""The view a tool gets of the session, and the wall around it.

`ToolContext` is the only thing a tool can touch. It reads through a copy and
writes into a buffer the loop merges afterwards, which is what makes a failed or
abandoned step harmless. These tests pin that wall.
"""

import pandas as pd
import pytest

from lakehouse_agent.agent.context import TableNotFound, ToolContext
from lakehouse_agent.testing.fake_llm import FakeLLM


def make_ctx(settings, tables=None, active=None) -> ToolContext:
    return ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM(),
        lake=None,
        settings=settings,
        active_table=active,
        _tables=tables or {},
    )


def test_a_tool_can_read_back_a_table_it_just_wrote(agent_settings):
    """Write then read inside one step has to work.

    It did not: the lookup used `written.get(k) or tables.get(k)`, and a
    DataFrame has no truth value, so `or` raised `ValueError` the moment the
    first branch actually found something. Nothing in the fake tools happened
    to write and then read, so it stayed invisible until a real tool would.
    """
    ctx = make_ctx(agent_settings, active="t01_a")
    ctx.put_table("t01_a", pd.DataFrame({"period": [1], "v": [2.0]}))

    frame = ctx.table("t01_a")

    assert list(frame.columns) == ["period", "v"]
    assert ctx.table().equals(frame)  # same thing through the active table


def test_reading_gives_a_copy_so_a_tool_cannot_reach_the_session_table(agent_settings):
    """The user is looking at that frame; a tool may not edit it in place."""
    original = pd.DataFrame({"period": [1, 2], "v": [1.0, 2.0]})
    ctx = make_ctx(agent_settings, {"t01_a": original}, active="t01_a")

    borrowed = ctx.table()
    borrowed["v"] = 99.0
    borrowed["yeni"] = 1.0

    assert original["v"].tolist() == [1.0, 2.0]
    assert "yeni" not in original.columns


def test_an_unknown_reference_says_which_ones_exist(agent_settings):
    """The model reads this message and picks a real name on the next try."""
    ctx = make_ctx(agent_settings, {"t01_a": pd.DataFrame()}, active="t01_a")
    with pytest.raises(TableNotFound, match="t01_a"):
        ctx.table("t09_uydurma")


def test_asking_for_the_active_table_when_there_is_none_says_what_to_do(agent_settings):
    ctx = make_ctx(agent_settings)
    with pytest.raises(TableNotFound, match="get_series"):
        ctx.table()


def test_generated_references_account_for_what_this_step_already_wrote(agent_settings):
    """Two tables in one step must not collide with each other."""
    ctx = make_ctx(agent_settings, {"t01_a": pd.DataFrame()})
    first = ctx.new_table_ref("x")
    ctx.put_table(first, pd.DataFrame())
    second = ctx.new_table_ref("y")

    assert first == "t02_x"
    assert second == "t03_y"


def test_a_tool_cannot_write_into_the_session_through_the_read_only_halves(
    agent_settings,
):
    """`_tables` and `lineage` were typed `Mapping` and handed over as `dict`.

    The loop passes the session's own dictionaries, so the annotation described
    an intention nothing enforced: a tool could insert a table or a lineage
    entry straight into the session, bypassing the write buffer entirely. An
    abandoned or timed-out step would then leave exactly the trace the buffer
    exists to prevent, and no test would have failed.
    """
    tables = {"t01_a": pd.DataFrame({"v": [1.0]})}
    lineage = {}
    ctx = ToolContext(
        session_id="s1",
        step_no=1,
        llm=FakeLLM(),
        lake=None,
        settings=agent_settings,
        active_table="t01_a",
        lineage=lineage,
        _tables=tables,
    )

    with pytest.raises(TypeError):
        ctx._tables["t02_b"] = pd.DataFrame()
    with pytest.raises(TypeError):
        ctx.lineage["t01_a.v"] = "anything"

    assert "t02_b" not in tables
    assert lineage == {}

    ctx.put_table("t02_b", pd.DataFrame({"v": [2.0]}))
    assert "t02_b" not in tables  # the buffer, not the session
    assert ctx.table("t02_b")["v"].tolist() == [2.0]


def test_the_view_is_a_snapshot_so_a_late_step_cannot_see_the_session_move(
    agent_settings,
):
    """A tool that overran its timeout is still holding this object.

    It should not observe a session that carried on without it, any more than
    it should be able to write to one.
    """
    tables = {"t01_a": pd.DataFrame({"v": [1.0]})}
    ctx = make_ctx(agent_settings, tables, active="t01_a")

    tables["t02_added_later"] = pd.DataFrame()

    assert not ctx.has_table("t02_added_later")
    with pytest.raises(TableNotFound):
        ctx.table("t02_added_later")


def test_the_write_side_records_what_the_loop_merges_back(agent_settings):
    """`set_active` and `add_chart` only ever mark intent.

    A tool that fails or times out must leave the session's active table and
    chart list exactly as they were, so neither of these touches anything but
    this object.
    """
    ctx = make_ctx(agent_settings, {"t01_a": pd.DataFrame()}, active="t01_a")

    assert ctx.has_table("t01_a")
    assert ctx.has_table() is True
    assert ctx.has_table("t09_yok") is False

    ctx.set_active("t02_b")
    ctx.add_chart({"chart_ref": "c01", "table_ref": "t01_a"})

    assert ctx.active_table == "t01_a"  # unchanged; the loop reads new_active
    assert ctx.new_active == "t02_b"
    assert ctx.written_charts == [{"chart_ref": "c01", "table_ref": "t01_a"}]


def test_has_table_says_no_when_there_is_no_active_table(agent_settings):
    assert make_ctx(agent_settings).has_table() is False
