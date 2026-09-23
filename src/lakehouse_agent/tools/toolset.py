"""The production tool set: what the agent is handed when it runs for real.

The counterpart of `testing.fake_tools.make_fake_registry`. Kept in its own
module rather than in `registry.py`, which the tool modules themselves import
for `ToolSpec` -- building the list there would be an import cycle.
"""

from __future__ import annotations

from lakehouse_agent.tools import analysis, charts, lakehouse, web_search, web_url
from lakehouse_agent.tools.registry import ToolRegistry


def make_registry() -> ToolRegistry:
    """Every tool with a real implementation behind it."""
    return ToolRegistry.of(
        [
            *lakehouse.TOOLS,
            *analysis.TOOLS,
            *charts.TOOLS,
            web_url.make_read_url(),
            web_search.make_web_search(),
        ]
    )
