"""The tool list the model sees, and the machinery that runs one.

The schema this produces is the only description of a tool the model ever gets.
If it is wrong the model writes wrong arguments, and the failure looks like the
model being stupid rather than like a bug in here.
"""

from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.contracts import ToolResult
from lakehouse_agent.testing.fake_llm import FakeLLM
from lakehouse_agent.tools.registry import ToolRegistry, ToolSpec, UnknownTool


class Window(BaseModel):
    """A nested model, which is what makes Pydantic emit `$defs`."""

    start: str = Field(..., pattern=r"^\d{4}-\d{2}$", examples=["2021-01"])
    end: str = Field(..., pattern=r"^\d{4}-\d{2}$", examples=["2025-12"])


class NestedArgs(BaseModel):
    series_ids: list[str] = Field(..., examples=[["evds:TP.KTF12"]])
    window: Window
    into: Literal["new", "active"] = "new"


def run(args: BaseModel, ctx: ToolContext) -> ToolResult:
    return ToolResult(ok=True, summary="ok")


def spec(**kwargs) -> ToolSpec:
    return ToolSpec(
        name=kwargs.pop("name", "nested_tool"),
        description_tr="Test araci.",
        args_model=kwargs.pop("args_model", NestedArgs),
        fn=run,
        **kwargs,
    )


def function_schema(registry: ToolRegistry, name: str) -> dict:
    for tool in registry.openai_tools():
        if tool["function"]["name"] == name:
            return tool["function"]["parameters"]
    raise AssertionError(f"{name} semada yok")


def test_a_nested_model_arrives_as_one_self_contained_schema():
    """Pydantic hoists nested models into `$defs` and points at them by `$ref`.

    Not every server resolves those, and a tool whose arguments silently arrive
    as an unconstrained object is worse than one that fails loudly: the model
    would then be free to put anything in `window`, and we would only find out
    from a wrong answer. None of the fake tools has a nested model, so nothing
    else exercises this path.
    """
    parameters = function_schema(ToolRegistry.of([spec()]), "nested_tool")

    assert "$defs" not in parameters
    assert "$ref" not in repr(parameters)

    window = parameters["properties"]["window"]
    assert window["properties"]["start"]["pattern"] == r"^\d{4}-\d{2}$"
    assert window["properties"]["start"]["examples"] == ["2021-01"]


def test_the_schema_keeps_the_constraints_the_loop_validates_against():
    """Same class on both sides, so the model cannot be told one thing and
    judged by another."""
    parameters = function_schema(ToolRegistry.of([spec()]), "nested_tool")

    assert parameters["required"] == ["series_ids", "window"]
    assert parameters["properties"]["into"]["enum"] == ["new", "active"]
    assert parameters["properties"]["series_ids"]["examples"] == [["evds:TP.KTF12"]]


def test_a_tool_whose_data_half_has_not_landed_is_hidden_and_refuses(agent_settings):
    """How we ship before the gold layer exists.

    The tool stays registered so `docs/tools.md` and the interface still know
    about it, but the model is not offered something that cannot work, and if
    it somehow asks anyway it gets a sentence it can act on rather than
    "unknown tool".
    """
    registry = ToolRegistry.of(
        [spec(name="run_sql", unavailable_reason="SQL araci gold katmani ile gelecek.")]
    )

    assert registry.names() == ["run_sql"]
    assert registry.openai_tools() == []

    result = registry.execute(
        "run_sql",
        NestedArgs(series_ids=[], window=Window(start="2021-01", end="2025-12")),
        ToolContext(
            session_id="s1",
            step_no=1,
            llm=FakeLLM(),
            lake=None,
            settings=agent_settings,
        ),
    )
    assert result.ok is False
    assert "gold katmani" in result.summary


def test_an_unknown_tool_names_the_real_ones():
    """This message goes straight back to the model, so it has to be useful."""
    registry = ToolRegistry.of([spec(name="get_series")])
    with pytest.raises(UnknownTool, match="get_series"):
        registry.require("sql_query")


def test_tools_are_listed_in_a_stable_order():
    """The tool list is part of the prompt; reordering it between calls would
    make otherwise identical requests look different to a cache."""
    registry = ToolRegistry.of([spec(name="z_tool"), spec(name="a_tool")])
    assert registry.names() == ["a_tool", "z_tool"]


class FreeFormArgs(BaseModel):
    """A tool with a field the model is meant to fill freely."""

    table_ref: str = Field(..., pattern=r"^t\d{2,}_[a-z][a-z0-9_]{0,24}$")
    options: dict[str, str] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


def test_a_key_the_model_made_up_is_refused_not_dropped():
    """The silent failure this registry existed to make impossible.

    Pydantic ignores unknown keys by default. A model writing `tablo` instead
    of `table_ref` -- and the whole schema reaches it in Turkish, so that slip
    is the expected one -- would have its key dropped, `table_ref` left at
    `None`, and `None` means the active table. The tool then analyses a
    different table than the one asked for and reports success. Nothing in the
    result, the summary or the sources says which table it actually read.
    """
    registry = ToolRegistry.of([spec(args_model=FreeFormArgs)])

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"tablo": "t02_deflate"})

    assert "tablo" in str(excinfo.value)
    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_the_schema_says_the_same_thing_the_validation_enforces():
    """Both halves of the rule, pinned together.

    Closing only the validation would let the model be blamed for a field list
    it was never given; closing only the schema would be a promise nothing
    keeps.
    """
    parameters = function_schema(ToolRegistry.of([spec()]), "nested_tool")

    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["window"]["additionalProperties"] is False


def test_a_free_form_field_is_left_open():
    """A mapping field has no `properties`, and closing it would forbid every
    key it exists to carry.

    That is why the rule is "objects that declare properties", not "every
    object". `dict[str, str]` already carries an `additionalProperties` of its
    own -- the value type -- and that must survive untouched; `dict[str, Any]`
    carries nothing and must stay untouched too.
    """
    registry = ToolRegistry.of([spec(args_model=FreeFormArgs)])
    parameters = function_schema(registry, "nested_tool")

    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["options"]["additionalProperties"] == {
        "type": "string"
    }
    assert parameters["properties"]["context"]["additionalProperties"] is True

    args = registry.validate(
        "nested_tool",
        {"table_ref": "t01_a", "options": {"ne": "olursa"}, "context": {"x": [1]}},
    )
    assert args.options == {"ne": "olursa"}
    assert args.context == {"x": [1]}


@pytest.mark.parametrize(
    "extra, in_schema, keeps_unknown",
    [("allow", True, True), ("forbid", False, False), (None, False, False)],
)
def test_the_two_halves_agree_whatever_the_tool_author_chose(
    extra, in_schema, keeps_unknown
):
    """The rule is "schema and validation say the same thing", not "everything
    is closed".

    A tool author who deliberately opens their model keeps it open on both
    sides; one who says nothing gets it closed on both sides. Pydantic writes
    `additionalProperties` itself for the first two, which is what the schema
    pass keys on -- it only fills in the case where nothing was said.
    """
    config = {} if extra is None else {"model_config": ConfigDict(extra=extra)}
    model = type(
        "Args", (BaseModel,), {"__annotations__": {"a": int}, "a": 1, **config}
    )
    registry = ToolRegistry.of([spec(args_model=model)])

    assert function_schema(registry, "nested_tool")["additionalProperties"] is in_schema

    if keeps_unknown:
        assert registry.validate("nested_tool", {"b": 2}).b == 2
    else:
        with pytest.raises(ValidationError):
            registry.validate("nested_tool", {"b": 2})
