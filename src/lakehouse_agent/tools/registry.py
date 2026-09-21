"""The list of tools the model sees, and the machinery to run them.

Two jobs. It turns Pydantic argument models into the JSON schema MIA expects,
and it runs a named tool with validated arguments. Keeping both here means the
schema the model reads and the validation the loop applies can never disagree:
they come from the same class.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails

from lakehouse_agent.agent.context import ToolContext
from lakehouse_agent.contracts import ToolResult

ToolFn = Callable[[BaseModel, ToolContext], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """One tool: what the model sees and what actually runs."""

    name: str
    description_tr: str
    args_model: type[BaseModel]
    fn: ToolFn

    #: Brief's name for this tool, so docs/tools.md can map our twelve
    #: functions onto the six tools the presentation names.
    brief_name: str = ""

    #: Set when running this tool must be confirmed by a human first. Only
    #: "fetch" exists today: pulling data that is not in the pool.
    approval: str | None = None

    #: Builds the preview shown on the approval card: what will be fetched,
    #: over what range, roughly how many calls.
    approval_preview: Callable[[BaseModel, ToolContext], dict[str, Any]] | None = None

    #: Set when `preview` comes back as a ranked candidate list. The loop reads
    #: it to decide whether the series-choice gate applies. Declared here rather
    #: than matched on the tool's name so renaming a tool cannot silently
    #: disable a human approval.
    returns_candidates: bool = False

    #: When set, the tool is registered but refuses to run and says why. Used
    #: for tools whose data-lane half has not landed yet, so the model gets a
    #: usable explanation instead of "unknown tool".
    unavailable_reason: str | None = None


class UnknownTool(KeyError):
    """The model asked for a tool that is not registered."""


@dataclass
class ToolRegistry:
    """The set of tools available for one run."""

    specs: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def of(cls, specs: Iterable[ToolSpec]) -> ToolRegistry:
        return cls({spec.name: spec for spec in specs})

    def add(self, spec: ToolSpec) -> None:
        self.specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def require(self, name: str) -> ToolSpec:
        spec = self.specs.get(name)
        if spec is None:
            raise UnknownTool(
                f"Bilinmeyen arac: {name!r}. "
                f"Kullanilabilir araclar: {', '.join(self.names()) or 'yok'}."
            )
        return spec

    def names(self) -> list[str]:
        return sorted(self.specs)

    def openai_tools(self) -> list[dict[str, Any]]:
        """Render the tool list MIA receives.

        The schema comes straight from the Pydantic model, so a field the loop
        validates is a field the model was told about, with the same pattern
        and the same description. MIA was observed writing SQL into an
        unexplained field, which is why every field carries an example.
        """
        rendered: list[dict[str, Any]] = []
        for name in self.names():
            spec = self.specs[name]
            if spec.unavailable_reason:
                continue
            rendered.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description_tr,
                        "parameters": tool_schema(spec.args_model),
                    },
                }
            )
        return rendered

    def validate(self, name: str, raw_args: dict[str, Any]) -> BaseModel:
        """Validate arguments against the tool's model. Raises ValidationError.

        A key the model made up is rejected, not dropped. Pydantic's default is
        `extra="ignore"`, which would silently discard it and leave the field it
        was aiming at on its default -- and for `table_ref` that default is the
        active table, so a misspelled key turns into a confident answer about
        the wrong table with nothing anywhere saying so. The schema now says
        `additionalProperties: false`, and this is the half that enforces it.
        """
        model = self.require(name).args_model
        _reject_unknown_keys(model, raw_args)
        return model.model_validate(raw_args)

    def execute(self, name: str, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """Run the tool. Synchronous: the loop calls this off the event loop."""
        spec = self.require(name)
        if spec.unavailable_reason:
            return ToolResult(ok=False, summary=spec.unavailable_reason)
        return spec.fn(args, ctx)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten `$ref`/`$defs` into a self-contained schema.

    Pydantic hoists nested models into `$defs` and points at them with `$ref`.
    Not every server resolves those, and a tool whose arguments silently arrive
    as an unconstrained object is worse than one that fails loudly, so the
    references are expanded before the schema leaves the process.

    Termination is by the chain of references being followed, not by how deep
    the walk has gone. A depth counter cannot tell three levels of nesting from
    a cycle: it fired on ordinary acyclic models, cut them off, and -- because
    `Optional[X]` puts the pointer one level down inside `anyOf` -- let the
    dangling `$ref` out anyway, which is the one thing this function exists to
    prevent. A model that genuinely refers back to itself ends as an open
    object: unconstrained, but resolvable.
    """
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve(node: Any, chain: frozenset[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                around = {
                    key: resolve(value, chain)
                    for key, value in node.items()
                    if key != "$ref"
                }
                if name in chain:  # pragma: no cover - cyclic argument model
                    return {"type": "object", **around}
                target = resolve(defs.get(name, {}), chain | {name})
                return {**target, **around}
            resolved = {key: resolve(value, chain) for key, value in node.items()}
            tag = resolved.get("discriminator")
            if isinstance(tag, dict) and "mapping" in tag:
                # A discriminator stores its `#/$defs/...` pointers as plain
                # string VALUES of `mapping`, so the walk never sees them as
                # references and they would leave here pointing at a `$defs`
                # that no longer exists. Every branch carries its tag inline
                # already, so the mapping adds nothing once the refs are gone.
                resolved["discriminator"] = {
                    key: value for key, value in tag.items() if key != "mapping"
                }
            return resolved
        if isinstance(node, list):
            return [resolve(item, chain) for item in node]
        return node

    return resolve(schema, frozenset())


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema for one tool's arguments, as the model receives it."""
    return _forbid_extra_properties(_inline_refs(model.model_json_schema()))


def _reject_unknown_keys(model: type[BaseModel], raw_args: Any) -> None:
    """Raise the same `extra_forbidden` error Pydantic would.

    Checked against the schema the model was actually shown, not against the
    class. That is the whole promise of this module stated as code: the field
    list in the schema and the field list validation accepts are the same
    object, so they cannot describe different things. An earlier version
    rebuilt the accepted set from `model_fields` and got aliases wrong in both
    directions at once -- it let through a key Pydantic would silently drop,
    and refused the key the schema itself advertised.

    It also follows the value down. `additionalProperties: false` is written on
    every modelled object in the schema, including nested ones, so enforcing it
    only at the top level would be another promise made and not kept.
    """
    errors = _extra_key_errors(tool_schema(model), raw_args, ())
    if errors:
        raise ValidationError.from_exception_data(model.__name__, errors)


def _branch_fits(schema: Any, value: Any) -> bool:
    """Whether a union branch is one this value could have been meant for."""
    declared = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(value, dict):
        return declared in (None, "object")
    if isinstance(value, list):
        return declared in (None, "array")
    return True


def _extra_key_errors(
    schema: Any, value: Any, loc: tuple[str | int, ...]
) -> list[InitErrorDetails]:
    """Every key in `value` that the schema said would not be accepted."""
    if not isinstance(schema, dict):
        # Pydantic does not emit a boolean or absent sub-schema, so nothing
        # reaches this today; it is here because the walk is recursive and a
        # sub-schema that is not a mapping must stop it rather than raise.
        return []  # pragma: no cover - not produced by model_json_schema()

    branches = schema.get("anyOf") or schema.get("oneOf")
    if branches:
        # A union -- `Optional[Model]` is the common one. Only the branches
        # whose shape the value could take are asked: `{"type": "null"}` has no
        # `properties`, so it "accepts" an object with a made-up key in it, and
        # a single branch accepting is enough to let the whole value through.
        tagged = _tagged_branch(schema, branches, value)
        if tagged is not None:
            # A discriminated union routes strictly by its tag. Asking every
            # branch would let the fields of the wrong one through -- the tag
            # is present in each branch, so the wrong branch accepts them --
            # and Pydantic would then drop every one onto its default.
            return _extra_key_errors(tagged, value, loc)
        candidates = [b for b in branches if _branch_fits(b, value)]
        if not candidates:
            return []
        per_branch = [_extra_key_errors(b, value, loc) for b in candidates]
        if any(not errors for errors in per_branch):
            return []
        return min(per_branch, key=len)

    if isinstance(value, dict):
        properties = schema.get("properties")
        # `dict[str, Model]` renders with no `properties` at all and the value
        # model under `additionalProperties`. Reading only `properties` walked
        # straight past it, so the schema closed the inner model and the
        # validation never looked inside.
        rest = schema.get("additionalProperties")
        if not isinstance(properties, dict):
            if not isinstance(rest, dict):
                return []
            properties = {}
        errors: list[InitErrorDetails] = []
        if rest is False:
            errors = [
                InitErrorDetails(
                    type="extra_forbidden", loc=(*loc, key), input=value[key]
                )
                # Sorted only to make the order deterministic; `key=str` because
                # a non-string key would otherwise raise TypeError out of a
                # method documented to raise ValidationError.
                for key in sorted(set(value) - set(properties), key=str)
            ]
        for key, item in value.items():
            if key in properties:
                errors.extend(_extra_key_errors(properties[key], item, (*loc, key)))
            elif isinstance(rest, dict):
                errors.extend(_extra_key_errors(rest, item, (*loc, key)))
        return errors

    if isinstance(value, list):
        # `tuple[A, B]` renders as `prefixItems` with no `items`, so a fixed
        # tuple of models was another edge the walk did not follow.
        prefix = schema.get("prefixItems") or []
        items = schema.get("items")
        return [
            error
            for index, item in enumerate(value)
            for error in _extra_key_errors(
                prefix[index] if index < len(prefix) else items, item, (*loc, index)
            )
        ]

    return []


def _tagged_branch(schema: Any, branches: list[Any], value: Any) -> Any:
    """The one branch a discriminated union's tag selects, if it selects one.

    `None` when there is no discriminator, or when the tag is missing or names
    nothing -- Pydantic phrases `union_tag_invalid` better than we would.
    """
    discriminator = schema.get("discriminator")
    if not isinstance(discriminator, dict) or not isinstance(value, dict):
        return None
    name = discriminator.get("propertyName")
    tag = value.get(name)
    for branch in branches:
        if not isinstance(branch, dict):  # pragma: no cover - not emitted
            continue
        declared = (branch.get("properties") or {}).get(name)
        if not isinstance(declared, dict):
            # Every branch of a discriminated union carries the tag as a field,
            # so nothing Pydantic emits reaches this.
            continue  # pragma: no cover - not produced by model_json_schema()
        if declared.get("const") == tag or tag in (declared.get("enum") or []):
            return branch
    return None


def _forbid_extra_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Close every modelled object in the schema.

    Only objects that declare `properties` are closed. A field typed
    `dict[str, Any]` renders as an object with no `properties`, and closing that
    would forbid the free-form payload it exists to carry.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            closed = {key: walk(value) for key, value in node.items()}
            if (
                closed.get("type") == "object"
                and "properties" in closed
                and "additionalProperties" not in closed
            ):
                closed["additionalProperties"] = False
            return closed
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(schema)
