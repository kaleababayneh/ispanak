"""What a tool gets besides its own validated arguments.

Tools never touch `SessionState`. They read through this view and write into a
buffer that the loop merges back only if the step succeeded. That separation is
what makes a timed-out or cancelled step harmless: its writes were never in the
session to begin with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pandas as pd

from lakehouse_agent.agent.state import new_chart_ref, new_table_ref
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.contracts import ColumnInfo
from lakehouse_agent.llm import LLM


def _detach(value: Any) -> Any:
    """A copy a tool cannot edit its way back into the session through."""
    copy = getattr(value, "model_copy", None)
    return copy(deep=True) if callable(copy) else value


class TableNotFound(KeyError):
    """A tool asked for a table reference the session does not have."""


@dataclass
class ToolContext:
    """A per-step view of the session, handed to exactly one tool call.

    Rebuilt for every step rather than reused. A tool that overruns its timeout
    keeps running in its thread; because it holds this object and not the
    session, whatever it writes afterwards lands in an orphan and the session
    stays consistent.
    """

    session_id: str
    step_no: int
    llm: LLM
    lake: Any
    settings: AgentSettings
    active_table: str | None = None
    lineage: Mapping[str, ColumnInfo] = field(default_factory=dict)
    user_message: str = ""

    #: Read-only view of the session's tables. Access goes through `table()`.
    _tables: Mapping[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    #: References already handed out for tables that no longer exist. A removed
    #: table must not free its number; see `SessionState.retired_tables`.
    retired_refs: frozenset[str] = frozenset()

    #: The charts the session already holds, so `new_chart_ref` can see them.
    #: A tuple rather than the session's list: a tool must not be able to
    #: append through it, which is the same reason `_tables` is proxied.
    charts: tuple[Any, ...] = ()

    #: Write buffer. The loop merges this into the session on success.
    written: dict[str, pd.DataFrame] = field(default_factory=dict)
    written_charts: list[Any] = field(default_factory=list)
    new_active: str | None = None

    def __post_init__(self) -> None:
        """Make the read-only halves actually read-only.

        The loop hands over the session's own `tables` and `lineage` dicts, so
        typing them `Mapping` documented an intention that nothing enforced: a
        tool could write straight into the session through either, and the
        whole point of the write buffer -- that an abandoned step leaves no
        trace -- would be gone without a single failing test.

        Snapshotted before wrapping, not proxied onto the live dict. A tool
        that overran its timeout is still holding this object, and it should
        not observe a session that moved on without it either.

        A proxy stops keys being added or removed, not the objects behind them
        from being edited. The frames are handled by `table()`, which copies.
        The `lineage` values are ordinary mutable models shared by identity
        with the session, so they are copied here instead -- otherwise a tool
        could rewrite a column's provenance in place, the write buffer would
        stay empty, and the traceability panel would show the edit as if it had
        always been there.
        """
        self._tables = MappingProxyType(dict(self._tables))
        self.lineage = MappingProxyType(
            {key: _detach(value) for key, value in self.lineage.items()}
        )

    def table(self, ref: str | None = None) -> pd.DataFrame:
        """Return a table for reading. `ref=None` means the active table.

        The frame comes back as a shallow copy. A tool that assigns a column on
        what it was handed then mutates its own object, not the session's, so a
        half-finished step cannot corrupt a table the user already saw. Under
        pandas copy-on-write this costs nothing until something is written.

        Shallow is exact, not loose: an object stored inside a cell is the same
        object the session holds. No table in the gold contract nests one, and
        a tool that starts stashing frames in cells has to copy them itself.
        """
        key = ref or self.active_table
        if key is None:
            raise TableNotFound(
                "Aktif tablo yok. Once get_series ile bir tablo olusturun."
            )
        # Membership, not `or`: a DataFrame has no truth value, so
        # `self.written.get(key) or ...` raises the moment a tool reads back a
        # table it wrote earlier in the same step.
        if key in self.written:
            source = self.written[key]
        elif key in self._tables:
            source = self._tables[key]
        else:
            source = None
        if source is None:
            available = sorted(set(self._tables) | set(self.written))
            raise TableNotFound(
                f"Bilinmeyen tablo anahtari: {key!r}. "
                f"Kullanilabilir tablolar: {', '.join(available) or 'yok'}."
            )
        return source.copy(deep=False)

    def has_table(self, ref: str | None = None) -> bool:
        key = ref or self.active_table
        return key is not None and (key in self.written or key in self._tables)

    def new_table_ref(self, suffix: str) -> str:
        """Generate the next reference. Tools never invent one.

        Retired references count as taken. A table the quality gate removed is
        gone from the session but its number is still on a card the person
        saw, and it is always the highest one, so reissuing it is the default
        outcome rather than an edge case.
        """
        known: dict[str, Any] = dict.fromkeys(self.retired_refs)
        known.update(self._tables)
        known.update(self.written)
        return new_table_ref(known, suffix)

    def new_chart_ref(self) -> str:
        """Generate the next chart reference. Tools never invent one.

        Counts the charts written in this step too. A step that draws two
        charts would otherwise ask twice, see the same session list both
        times, and get `c01` back for both -- and the second would overwrite
        the first in the interface, which looks up charts by key.
        """
        return new_chart_ref([*self.charts, *self.written_charts])

    def put_table(self, ref: str, frame: pd.DataFrame) -> None:
        self.written[ref] = frame

    def set_active(self, ref: str) -> None:
        self.new_active = ref

    def add_chart(self, chart: Any) -> None:
        self.written_charts.append(chart)
