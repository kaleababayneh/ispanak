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
        _reject_unknown_keys(name, model, raw_args)
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
    """
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 10:  # pragma: no cover - defensive against a cyclic model
            # Drop the pointer rather than let it out. A dangling `$ref` is the
            # unresolvable schema this function exists to prevent.
            if isinstance(node, dict):
                return {key: value for key, value in node.items() if key != "$ref"}
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**resolve(target, depth + 1)}
                merged.update({k: v for k, v in node.items() if k != "$ref"})
                return merged
            return {key: resolve(value, depth + 1) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item, depth + 1) for item in node]
        return node

    return resolve(schema)


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema for one tool's arguments, as the model receives it."""
    return _forbid_extra_properties(_inline_refs(model.model_json_schema()))


def _accepted_keys(model: type[BaseModel]) -> set[str]:
    """Every key the model is allowed to send: field names and their aliases."""
    keys: set[str] = set()
    for name, info in model.model_fields.items():
        keys.add(name)
        for alias in (info.alias, info.validation_alias, info.serialization_alias):
            if isinstance(alias, str):
                keys.add(alias)
    return keys


def _reject_unknown_keys(
    tool: str, model: type[BaseModel], raw_args: dict[str, Any]
) -> None:
    """Raise the same `extra_forbidden` error Pydantic would.

    Built by hand rather than by forcing `extra="forbid"` onto every argument
    model, because those models belong to whoever wrote the tool; the rule that
    the schema and the validation agree belongs here, and it has to hold for a
    tool registered tomorrow by someone who never read this file.
    """
    if not isinstance(raw_args, dict):
        return
    if model.model_config.get("extra") == "allow":
        return
    unknown = sorted(set(raw_args) - _accepted_keys(model))
    if not unknown:
        return
    raise ValidationError.from_exception_data(
        model.__name__,
        [
            InitErrorDetails(type="extra_forbidden", loc=(key,), input=raw_args[key])
            for key in unknown
        ],
    )


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
