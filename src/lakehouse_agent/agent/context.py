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

from lakehouse_agent.agent.state import new_table_ref
from lakehouse_agent.config import AgentSettings
from lakehouse_agent.contracts import ColumnInfo
from lakehouse_agent.llm import LLM


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
        not observe a session that moved on without it either. Only the mapping
        is copied; the frames inside are not, and `table()` copies those.
        """
        self._tables = MappingProxyType(dict(self._tables))
        self.lineage = MappingProxyType(dict(self.lineage))

    def table(self, ref: str | None = None) -> pd.DataFrame:
        """Return a table for reading. `ref=None` means the active table.

        The frame comes back as a shallow copy. A tool that assigns a column on
        what it was handed then mutates its own object, not the session's, so a
        half-finished step cannot corrupt a table the user already saw. Under
        pandas copy-on-write this costs nothing until something is written.
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
        """Generate the next reference. Tools never invent one."""
        known = {**dict(self._tables), **self.written}
        return new_table_ref(known, suffix)

    def put_table(self, ref: str, frame: pd.DataFrame) -> None:
        self.written[ref] = frame

    def set_active(self, ref: str) -> None:
        self.new_active = ref

    def add_chart(self, chart: Any) -> None:
        self.written_charts.append(chart)
