"""Searching the open web, behind one interface.

What this is for: a break date the lake can find but not explain. The tools
say the credit rate changed in 2023-07; what happened that month is not in
any series, it is in a central bank's press release. So the agent looks it
up and cites where it read it.

Two things are deliberate here.

**The provider is swappable.** Today it is DuckDuckGo through `ddgs`, which
needs no key and no budget. It also rate-limits, and a demo is exactly when
that would show. `search` takes a `provider`, so moving to a paid API
(Serper, Tavily, Brave) is one function, not a change to the tool, the
prompt or the tests.

**Results are data, never instructions.** A page can say "ignore your
previous instructions"; the text comes back labelled for what it is
(`tools/web_search.py` puts the label in front of the model) and nothing
here evaluates, executes or stores what it reads. No figure in an answer may
come from here either -- the verifier only accepts numbers that a lake tool
returned.

Results are cached per query under `data/adhoc/search/`, with the time they
were fetched: a demo that asks the same question twice does not spend a
second call, and a rate limit mid-run does not erase what was already found.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

logger = logging.getLogger(__name__)

ADHOC_ROOT = Path("data/adhoc/search")
#: Turkish results for Turkish questions: the sources that matter here
#: (TCMB, BDDK, the press) publish in Turkish.
REGION = "tr-tr"
DEFAULT_RESULTS = 5
MAX_RESULTS = 10
TIMEOUT_S = 20.0


class WebSearchError(RuntimeError):
    """The search could not be run (no network, rate limit, provider down)."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


Provider = Callable[[str, int], list[SearchResult]]


def ddg_provider(query: str, k: int) -> list[SearchResult]:
    """DuckDuckGo through `ddgs`. No key, no cost, and a rate limit that a
    busy afternoon can reach -- which is why `search` takes a provider."""
    from ddgs import DDGS

    try:
        rows = DDGS(timeout=TIMEOUT_S).text(query, region=REGION, max_results=k)
    except Exception as exc:  # noqa: BLE001 - every provider failure reads alike
        raise WebSearchError(
            f"Arama yapilamadi ({type(exc).__name__}): {str(exc)[:120]}"
        ) from exc
    return [
        SearchResult(
            title=" ".join(str(row.get("title") or "").split())[:200],
            url=str(row.get("href") or ""),
            snippet=" ".join(str(row.get("body") or "").split())[:500],
        )
        for row in rows
        if row.get("href")
    ]


def cache_path(query: str, k: int, *, root: Path = ADHOC_ROOT) -> Path:
    digest = sha256(f"{query.strip().lower()}|{k}".encode()).hexdigest()[:16]
    return root / f"{digest}.json"


def search(
    query: str,
    k: int = DEFAULT_RESULTS,
    *,
    provider: Provider | None = None,
    root: Path = ADHOC_ROOT,
    force: bool = False,
) -> tuple[list[SearchResult], str, bool]:
    """(results, when they were fetched, whether they came from the cache).

    `provider` defaults to `ddg_provider` but is read at call time, not bound
    when this function is defined: a default argument would be the real
    provider forever, and a test that replaced the module's attribute would
    quietly go to the network instead (measured -- three tests did).
    """
    provider = provider or ddg_provider
    query = " ".join(query.split())
    if not query:
        raise WebSearchError("Bos sorgu ile arama yapilmaz.")
    k = max(1, min(k, MAX_RESULTS))

    path = cache_path(query, k, root=root)
    if not force and path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        results = [SearchResult(**row) for row in stored["results"]]
        return results, stored.get("fetched_at", ""), True

    results = provider(query, k)
    fetched_at = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query": query,
                "k": k,
                "fetched_at": fetched_at,
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("web arama: %r -> %d sonuc", query, len(results))
    return results, fetched_at, False
