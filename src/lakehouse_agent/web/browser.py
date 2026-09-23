"""Render a page with a headless browser, for the one case the other two
layers cannot help with.

`web/extract.py` already tries a page's own `<table>`s and then the JSON it
may carry inside itself (`_embedded_json`) before giving up -- see
`docs/decisions.md`'s entry for `read_url`. This module is the third layer,
used only when both of those come back empty: a page that builds its data
with JavaScript and ships none of it in the initial response.

Optional on purpose (`pyproject.toml`'s `browser` extra, Playwright/Chromium):
importing this module never requires Playwright to be installed, only
`render()` calling it does, and it fails with `BrowserUnavailable` rather
than `ImportError` so `tools/web_url.py` can tell "not installed" apart from
"failed to render" and word the warning accordingly.

A single GET has one address to check (`web/fetch.py`). A rendered page runs
the *page's own JavaScript*, which can issue any number of requests to any
number of hosts on its own -- the threat model this module has to cover is
materially larger, not the same one repeated. So every HTTP sub-request the
page makes is checked exactly as the first address was (`check_url`, imported
from `web/fetch.py`, not copied), every WebSocket is refused outright, and the
connection's real peer address is checked again after the page loads
(`_refuse_if_internal`, mirroring `web/fetch.py::_peer_address`'s defence
against DNS rebinding between the pre-flight check and the actual connection).

`render()`, the public entry point, runs the actual Playwright call
(`_render_sync`) in a fresh child process with a hard wall-clock deadline --
see its own docstring for why: a page that opens a WebSocket measurably wedged
the whole call, past `NAV_TIMEOUT_S`, in a place no exception or timeout
inside this same process could reach.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import socket
from typing import Any
from urllib.parse import urlparse

from lakehouse_agent.web.fetch import (
    MAX_BYTES,
    WebFetchError,
    _refuse_if_internal,
    check_url,
    redact_url,
)

#: Playwright's own ceiling on navigation. Pages loading normally finish well
#: inside this; `HARD_TIMEOUT_S` below is the backstop for when they do not.
NAV_TIMEOUT_S = 20.0

#: Wall-clock ceiling on the whole render, subprocess spawn and teardown
#: included. A page that opens a WebSocket to a refused address measurably
#: hung past `NAV_TIMEOUT_S` with no exception raised anywhere in this process
#: (see docs/decisions.md) -- Playwright's sync API dispatches every call,
#: including one made from inside a route handler, on a single fiber, and a
#: handler that blocks that fiber blocks everything waiting on it, forever.
#: Nothing catchable *in this process* can end that. A process can always be
#: killed from outside it, which is the only reason `render()` runs
#: `_render_sync` in a child rather than calling it directly. A little over
#: `NAV_TIMEOUT_S` -- Playwright's own timeouts should end an ordinary slow
#: page well before this fires, so this is a last resort, not the usual exit.
HARD_TIMEOUT_S = NAV_TIMEOUT_S + 10.0

#: Grace period after a kill signal for the OS to actually reclaim the
#: process before giving up on `join()`-ing it. Not counted against the
#: deadline above -- the deadline is when we stop waiting for a normal exit,
#: not the total time `render()` may take.
KILL_GRACE_S = 5.0

#: Request kinds a rendered page never needs read for its data, and that a
#: browser fetches most of on its own once the document loads. Blocking them
#: is speed and a smaller attack surface, not correctness.
#: WebSocket is *not* in this set even though nothing here reads one: measured
#: against a real Chromium, `context.route`/`page.route` never sees a
#: WebSocket handshake at all -- Playwright has a separate API for that
#: (`context.route_web_socket`, wired up in `_render_sync` below) -- so a
#: "websocket" entry here would sit dead, matching nothing and blocking
#: nothing, while a page whose JS opened `ws://127.0.0.1` sailed straight
#: through this route handler unchecked.
_BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font"})


class BrowserUnavailable(RuntimeError):
    """Playwright is not installed."""


class RenderError(RuntimeError):
    """The page could not be rendered, and why, in a sentence the model can
    pass on to the person."""


def _import_sync_playwright():
    """Isolated so a test can make this fail without Playwright installed."""
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _allow_subrequest(url: str, *, resolve=socket.getaddrinfo) -> bool:
    """Pure: would `web/fetch.py`'s own guard let this address through?

    Used for every request the rendered page's own script makes, not only
    the page's own address -- see the module docstring.
    """
    try:
        check_url(url, resolve=resolve)
    except WebFetchError:
        return False
    return True


def _validate_peer(server_addr: dict[str, Any] | None, url: str) -> None:
    """Raise `RenderError` if the connection's real peer address is not a
    public, routable one. Pure; `server_addr` is whatever Playwright's
    `Response.server_addr()` returned (or None -- best effort, mirroring
    `web/fetch.py::_peer_address`)."""
    if not server_addr:
        return
    address = server_addr.get("ipAddress")
    if not address:
        return
    try:
        _refuse_if_internal(address, urlparse(url).hostname or url)
    except WebFetchError as exc:
        raise RenderError(str(exc)) from exc


def _render_sync(url: str, *, resolve=socket.getaddrinfo) -> bytes:
    """The actual Playwright call. Not called directly outside this module or
    a test -- `render()` below is what everything else uses, so that a hang
    in here (measured: real) cannot outlast `HARD_TIMEOUT_S`.

    Raises `BrowserUnavailable` if Playwright is not installed, or
    `RenderError` if the page could not be rendered safely. A fresh,
    non-persistent context per call; a single page, closed if the page itself
    opens another tab.
    """
    try:
        sync_playwright = _import_sync_playwright()
    except ImportError as exc:
        raise BrowserUnavailable(
            "Sayfa JavaScript gerektiriyor, tarayici katmani kurulu degil "
            "(uv sync --extra browser && uv run playwright install chromium)."
        ) from exc

    def _route(route) -> None:  # noqa: ANN001 - playwright.Route
        request = route.request
        if request.resource_type in _BLOCKED_RESOURCE_TYPES:
            route.abort()
            return
        if not _allow_subrequest(request.url, resolve=resolve):
            route.abort()
            return
        route.continue_()

    def _reject_websocket(_ws) -> None:  # noqa: ANN001 - playwright.WebSocketRoute
        # Deliberately empty. Per Playwright's own docs: "By default, the
        # routed WebSocket will not connect to the server" -- a handler that
        # does nothing already means the page's WebSocket never reaches a
        # real network address, mocked rather than merely blocked, which is
        # everything this needs. Calling `ws.close()` from inside this
        # handler was tried and measured to hang: the sync API dispatches
        # every call, including a synchronous one made from inside a handler
        # it is itself running, on one fiber, and `close()` here tries to
        # switch back into that same fiber to wait for the result -- of
        # itself. `page.goto` then blocked past `NAV_TIMEOUT_S` with nothing
        # in this process able to interrupt it (see `HARD_TIMEOUT_S` and
        # `render()`, the backstop this measurement led to).
        pass

    try:
        with sync_playwright() as playwright:
            browser_instance = playwright.chromium.launch()
            try:
                context = browser_instance.new_context(
                    accept_downloads=False,
                    service_workers="block",
                )
                context.set_default_navigation_timeout(NAV_TIMEOUT_S * 1000)
                context.set_default_timeout(NAV_TIMEOUT_S * 1000)
                # Context-level, set up before any page exists: it then
                # applies to the page below *and* to anything the page's own
                # JS opens (a popup), with no gap either could slip through.
                context.route("**/*", _route)
                context.route_web_socket("**/*", _reject_websocket)

                page = context.new_page()
                # Popup handler on the *page*, not `context.on("page", ...)`:
                # that event also fires for this call's own `new_page()`
                # above, and closing it out from under `page.goto` below
                # crashed with TargetClosedError against a real Chromium
                # (measured). A page that opens a second tab (a link with
                # target="_blank") gets nothing read from it regardless --
                # this call only ever wants the one document it navigated to.
                page.on("popup", lambda popup: popup.close())

                try:
                    response = page.goto(url, wait_until="networkidle")
                except Exception as exc:  # noqa: BLE001 - any nav failure is reported, not raised
                    raise RenderError(
                        f"Sayfa render edilemedi ({type(exc).__name__}): "
                        f"{redact_url(url)}"
                    ) from exc

                if response is None:
                    raise RenderError(f"Sayfa yanit vermedi: {redact_url(url)}")

                _validate_peer(response.server_addr(), url)

                html = page.content()
            finally:
                browser_instance.close()
    except (BrowserUnavailable, RenderError):
        raise
    except Exception as exc:  # noqa: BLE001 - a browser-layer failure is an answer
        raise RenderError(f"Tarayici hatasi ({type(exc).__name__}).") from exc

    body = html.encode("utf-8")
    if len(body) > MAX_BYTES:
        raise RenderError(
            f"Render edilen sayfa {MAX_BYTES // (1024 * 1024)} MB sinirini asiyor."
        )
    return body


def _render_worker(url: str, resolve, conn) -> None:  # noqa: ANN001
    """Runs inside the child process `render()` spawns. Sends back
    `(status, payload)`: `("ok", bytes)`, `("unavailable", message)`, or
    `("error", message)` -- never lets an exception escape into
    `multiprocessing`'s own (much less specific) reporting."""
    try:
        html = _render_sync(url, resolve=resolve)
    except BrowserUnavailable as exc:
        conn.send(("unavailable", str(exc)))
    except RenderError as exc:
        conn.send(("error", str(exc)))
    except Exception as exc:  # noqa: BLE001 - the child must always answer
        conn.send(("error", f"Tarayici hatasi ({type(exc).__name__})."))
    else:
        conn.send(("ok", html))
    finally:
        conn.close()


def _in_new_process_group(target, args, conn) -> None:  # noqa: ANN001
    """The actual target every `_run_in_subprocess` child runs.

    A process group of its own (`os.setsid`) is what lets the parent kill
    this *and* any process it starts -- Chromium, from Playwright's own
    driver -- in one signal (`_kill_process_group`) rather than leaving a
    browser process orphaned when only its immediate child is killed.
    POSIX only; harmless best-effort elsewhere (`hasattr` guard).
    """
    if hasattr(os, "setsid"):
        os.setsid()
    target(*args, conn)


def _kill_process_group(process: mp.process.BaseProcess) -> None:
    pid = process.pid
    if pid is None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX fallback
            process.kill()
    except ProcessLookupError:
        pass  # already gone


def _run_in_subprocess(target, args: tuple, timeout_s: float):  # noqa: ANN001
    """Run `target(*args, conn)` in a fresh child process; return what it
    sent back over `conn`, or raise `TimeoutError` if it outlives
    `timeout_s` -- killing the whole process group, not just asking nicely.

    `target` (and everything in `args`) must be picklable by reference, so a
    module-level function, not a closure -- `multiprocessing`'s `spawn` start
    method re-imports it in the child rather than inheriting this process'
    memory (deliberate: the safe way to start a process from a thread inside
    an asyncio loop, which is how `agent/loop.py` calls tools).
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_in_new_process_group, args=(target, args, child_conn), daemon=True
    )
    process.start()
    child_conn.close()
    process.join(timeout_s)
    if process.is_alive():
        _kill_process_group(process)
        process.join(KILL_GRACE_S)
        raise TimeoutError(f"{timeout_s:g} saniyede bitmedi, surec sonlandirildi.")
    if not parent_conn.poll():
        raise RuntimeError("Alt surec sonuc gondermeden kapandi.")
    return parent_conn.recv()


def render(url: str, *, resolve=socket.getaddrinfo) -> bytes:
    """The page's HTML after its own JavaScript has run.

    Runs `_render_sync` in a fresh child process under a hard wall-clock
    deadline (`HARD_TIMEOUT_S`): a page that opens a WebSocket to a refused
    address measurably wedged Playwright's sync dispatcher with nothing in
    this process able to interrupt it (see `_render_sync`'s note on
    `_reject_websocket`). A stuck thread in this process cannot be killed; a
    process always can be, and killing the process also ends whatever
    Chromium process it started, isolating a Chromium crash from the agent
    process as a side effect.

    Raises `BrowserUnavailable` if Playwright is not installed, or
    `RenderError` if the page could not be rendered safely, including when
    the deadline is what ends it.
    """
    try:
        status, payload = _run_in_subprocess(
            _render_worker, (url, resolve), HARD_TIMEOUT_S
        )
    except (TimeoutError, RuntimeError) as exc:
        raise RenderError(f"{exc} ({redact_url(url)})") from exc

    if status == "unavailable":
        raise BrowserUnavailable(payload)
    if status == "error":
        raise RenderError(payload)
    return payload
