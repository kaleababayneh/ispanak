"""Test doubles that ship inside the package.

These live under `src/` rather than `tests/` on purpose: the `--fake` CLI mode
and (later) the eval runner import them from outside the test suite, and with a
src layout `tests/` is not part of the installed package. Keeping them here
means no `pythonpath` tricks and no surprise when running inside a container.

Production modules must not import this package. The only exception is a lazy
import inside a CLI entrypoint, guarded by a `--fake` flag.

`FakeLake` is exported lazily. Importing it eagerly pulls pandas, pyarrow and
numpy into every process that only wanted `FakeLLM` -- measured at +165 ms
(311 -> 476) on `from lakehouse_agent.testing import FakeLLM`. The fake model
has no use for a DataFrame, so it should not pay for one.
"""

from typing import TYPE_CHECKING, Any

from lakehouse_agent.testing.fake_llm import FakeLLM, ScriptExhausted

if TYPE_CHECKING:  # pragma: no cover - import-time only, for type checkers
    from lakehouse_agent.testing.fake_lake import FakeLake

__all__ = ["FakeLLM", "FakeLake", "ScriptExhausted"]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access, so `FakeLake` costs nothing until used."""
    if name == "FakeLake":
        from lakehouse_agent.testing.fake_lake import FakeLake

        return FakeLake
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
