"""Versioned prompts, loaded from Markdown files next to this module.

Prompts live in files rather than string literals for two reasons. A prompt
change is reviewable as a diff, and the version recorded on every step and in
every usage-log line answers "which prompt produced this answer" months later.

Substitution is `{{name}}` only. `str.format` would choke on the braces that
appear in Turkish examples and in the JSON we show the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def tag(self) -> str:
        """Goes on the LLM call and into the step log: 'planner@1'."""
        return f"{self.name}@{self.version}"


@cache
def load_prompt(name: str) -> Prompt:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt bulunamadi: {path}")
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    version = "0"
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "version":
                version = value.strip()
        raw = raw[match.end() :]
    return Prompt(name=name, version=version, text=raw.strip())


def render(prompt: Prompt, **values: str) -> str:
    """Fill `{{name}}` placeholders. An unknown one is left as-is, visibly."""

    def replace(match: re.Match[str]) -> str:
        return str(values.get(match.group(1), match.group(0)))

    return _PLACEHOLDER.sub(replace, prompt.text)


def placeholders(prompt: Prompt) -> set[str]:
    return set(_PLACEHOLDER.findall(prompt.text))
