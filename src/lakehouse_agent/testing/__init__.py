"""Test doubles that ship inside the package.

These live under `src/` rather than `tests/` on purpose: the `--fake` CLI mode
and (later) the eval runner import them from outside the test suite, and with a
src layout `tests/` is not part of the installed package. Keeping them here
means no `pythonpath` tricks and no surprise when running inside a container.

Production modules must not import this package. The only exception is a lazy
import inside a CLI entrypoint, guarded by a `--fake` flag.
"""

from lakehouse_agent.testing.fake_lake import FakeLake
from lakehouse_agent.testing.fake_llm import FakeLLM, ScriptExhausted

__all__ = ["FakeLLM", "FakeLake", "ScriptExhausted"]
