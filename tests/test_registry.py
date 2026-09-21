"""The tool list the model sees, and the machinery that runs one.

The schema this produces is the only description of a tool the model ever gets.
If it is wrong the model writes wrong arguments, and the failure looks like the
model being stupid rather than like a bug in here.
"""

import json
from typing import Any, Literal

import pytest
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

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


class Leaf(BaseModel):
    leaf: str = "x"


class Mid(BaseModel):
    mid: Leaf | None = None


class DeepArgs(BaseModel):
    outer: Mid | None = None


def test_a_made_up_key_inside_a_nested_object_is_refused_too():
    """`additionalProperties: false` is written on every modelled object.

    Enforcing it only at the top level would be the same promise broken one
    level down, and a nested model whose fields all have defaults would take
    the bad key silently and use its defaults.
    """
    registry = ToolRegistry.of([spec()])

    with pytest.raises(ValidationError) as excinfo:
        registry.validate(
            "nested_tool",
            {"series_ids": ["a"], "window": {"baslangic": "2024-01", "end": "2024-06"}},
        )

    assert excinfo.value.errors()[0]["loc"] == ("window", "baslangic")


def test_the_accepted_keys_are_the_ones_the_schema_advertises():
    """Aliases used to be got wrong in both directions at once.

    The accepted set was rebuilt from `model_fields`, which meant it both let
    through a key Pydantic would silently drop and refused the key the schema
    itself advertised. Reading the schema instead makes the two halves the same
    object, which is what this module claims in its first paragraph.
    """

    class Aliased(BaseModel):
        table_ref: str | None = Field(None, alias="table")

    registry = ToolRegistry.of([spec(args_model=Aliased)])
    assert list(function_schema(registry, "nested_tool")["properties"]) == ["table"]

    assert registry.validate("nested_tool", {"table": "t02_x"}).table_ref == "t02_x"
    with pytest.raises(ValidationError):
        registry.validate("nested_tool", {"table_ref": "t02_x"})


def test_a_field_with_alias_choices_accepts_the_key_it_advertised():
    """The inverse failure at the same line: the gate refused its own schema."""

    class Choice(BaseModel):
        table_ref: str | None = Field(
            None, validation_alias=AliasChoices("table", "tabela")
        )

    registry = ToolRegistry.of([spec(args_model=Choice)])
    advertised = list(function_schema(registry, "nested_tool")["properties"])

    assert registry.validate("nested_tool", {advertised[0]: "t02_x"}).table_ref == (
        "t02_x"
    )


def test_ordinary_nesting_is_not_mistaken_for_a_cycle():
    """The guard used to count every step of the walk, not every reference.

    Three levels of `Optional[Model]` -- which `ChartSpec` is one field away
    from -- hit the cap, and because `Optional[X]` puts the pointer inside an
    `anyOf` the cut let the dangling `$ref` out anyway. That is precisely the
    unresolvable schema this function exists to prevent.
    """
    parameters = function_schema(
        ToolRegistry.of([spec(args_model=DeepArgs)]), "nested_tool"
    )

    assert "$ref" not in json.dumps(parameters)
    assert "$defs" not in parameters

    leaf = parameters["properties"]["outer"]["anyOf"][0]["properties"]["mid"]["anyOf"][
        0
    ]
    assert leaf["properties"]["leaf"]["type"] == "string"
    assert leaf["additionalProperties"] is False


def test_a_model_that_refers_to_itself_ends_as_an_open_object():
    """Unconstrained is a schema a server can read; a dangling pointer is not."""

    class Node(BaseModel):
        name: str = "n"
        child: "Node | None" = None

    Node.model_rebuild()
    parameters = function_schema(
        ToolRegistry.of([spec(args_model=Node)]), "nested_tool"
    )

    assert "$ref" not in json.dumps(parameters)
    assert parameters["properties"]["child"]["anyOf"][0]["type"] == "object"


def test_get_returns_none_for_a_tool_that_is_not_registered():
    """`get` is the soft lookup; `require` is the one that raises."""
    registry = ToolRegistry.of([spec(name="get_series")])

    assert registry.get("get_series").name == "get_series"
    assert registry.get("sql_query") is None
    assert registry.require("get_series").name == "get_series"


def test_a_registered_tool_runs_and_a_union_argument_is_checked_in_its_branch():
    """The registry's actual job, end to end, plus the union path in the gate.

    `Optional[Model]` renders as `anyOf`, so a bad key inside one has to be
    found by trying the branches rather than by looking at the node itself.
    """
    registry = ToolRegistry.of([spec(args_model=DeepArgs)])

    args = registry.validate("nested_tool", {"outer": {"mid": {"leaf": "x"}}})
    assert args.outer.mid.leaf == "x"
    assert registry.execute("nested_tool", args, None).ok is True

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"outer": {"mid": {"yaprak": "x"}}})
    assert excinfo.value.errors()[0]["loc"] == ("outer", "mid", "yaprak")

    assert registry.validate("nested_tool", {"outer": None}).outer is None


def test_a_tool_can_be_added_after_the_registry_was_built():
    registry = ToolRegistry.of([spec(name="get_series")])
    registry.add(spec(name="run_sql"))

    assert registry.names() == ["get_series", "run_sql"]


def test_the_gate_steps_aside_when_the_value_is_the_wrong_shape_entirely():
    """It reports made-up keys; it is not a second type checker.

    When no branch of a union could take this value at all, the gate says
    nothing and lets Pydantic produce the type error it is much better at
    phrasing. Reporting both would bury the real one.
    """

    class Shapes(BaseModel):
        name: str | int = "x"
        rows: list[Window] | None = None
        loose: list = Field(default_factory=list)

    registry = ToolRegistry.of([spec(args_model=Shapes)])

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"name": {"uydurma": 1}})
    assert {error["type"] for error in excinfo.value.errors()} == {
        "string_type",
        "int_type",
    }

    # A list against a union is checked inside its element model...
    with pytest.raises(ValidationError) as excinfo:
        registry.validate(
            "nested_tool", {"rows": [{"start": "2021-01", "end": "2021-02", "bas": 1}]}
        )
    assert excinfo.value.errors()[0]["loc"] == ("rows", 0, "bas")

    # ...and an untyped list has nothing to check its elements against.
    assert registry.validate("nested_tool", {"loose": [{"ne": "olursa"}]}).loose == [
        {"ne": "olursa"}
    ]


class Cell(BaseModel):
    a: int = 1


class Deflate(BaseModel):
    op: Literal["deflate"] = "deflate"
    base: str = "2021"


class Rebase(BaseModel):
    op: Literal["rebase"] = "rebase"
    anchor: str = "2021-01"


def test_the_gate_follows_the_two_edges_a_plain_walk_misses():
    """`dict[str, Model]` and `tuple[Model, ...]` are shaped unlike everything
    else Pydantic emits.

    The first renders with no `properties` at all and the value model under
    `additionalProperties`; the second with `prefixItems` and no `items`. The
    schema closed the inner model in both cases while the walk went straight
    past it -- the schema forbidding a key that validation then accepted and
    dropped, which is the exact divergence this module exists to prevent.
    """

    class Shapes(BaseModel):
        cells: dict[str, Cell] = Field(default_factory=dict)
        pair: tuple[Cell, Cell] | None = None

    registry = ToolRegistry.of([spec(args_model=Shapes)])

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"cells": {"x": {"uydurma": 1}}})
    assert excinfo.value.errors()[0]["loc"] == ("cells", "x", "uydurma")

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"pair": [{"a": 1}, {"uydurma": 2}]})
    assert excinfo.value.errors()[0]["loc"] == ("pair", 1, "uydurma")


def test_a_tagged_union_is_read_by_its_tag_not_by_whichever_branch_agrees():
    """Asking every branch is wrong once there is a discriminator.

    The tag is a field of every branch, so `{op: "deflate", anchor: ...}` is
    accepted whole by the `rebase` branch and the gate says nothing -- while
    Pydantic routes strictly by the tag and drops `anchor` onto its default.
    That is the wrong-value-silently failure in a different costume.
    """

    class Tagged(BaseModel):
        step: Deflate | Rebase = Field(Deflate(), discriminator="op")

    registry = ToolRegistry.of([spec(args_model=Tagged)])

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"step": {"op": "deflate", "anchor": "x"}})
    assert excinfo.value.errors()[0]["loc"] == ("step", "anchor")

    assert (
        registry.validate(
            "nested_tool", {"step": {"op": "rebase", "anchor": "2021-01"}}
        ).step.anchor
        == "2021-01"
    )

    # An unknown tag is Pydantic's to phrase, not ours.
    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"step": {"op": "uydurma"}})
    assert excinfo.value.errors()[0]["type"] == "union_tag_invalid"


def test_no_unresolvable_pointer_leaves_the_process_in_any_shape():
    """A discriminator stores its pointers as plain string VALUES of `mapping`.

    The walk only ever rewrote a dict that had a `$ref` KEY, so `$defs` was
    deleted and the mapping was left pointing into it. The old assertion
    (`"$ref" not in ...`) could not see it: the document contains the pointer,
    just not under that name.
    """

    class Tagged(BaseModel):
        step: Deflate | Rebase = Field(Deflate(), discriminator="op")

    parameters = function_schema(
        ToolRegistry.of([spec(args_model=Tagged)]), "nested_tool"
    )

    assert "#/$defs/" not in json.dumps(parameters)
    assert "$defs" not in parameters


def test_a_non_string_argument_key_still_raises_a_validation_error():
    """`validate` is documented to raise ValidationError. Sorting the extra
    keys for a deterministic message must not turn that into a TypeError that
    escapes the loop's retry path."""
    registry = ToolRegistry.of([spec(args_model=Cell)])

    with pytest.raises(ValidationError) as excinfo:
        registry.validate("nested_tool", {"b": 1, 2: "x"})
    assert len(excinfo.value.errors()) == 2
