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
