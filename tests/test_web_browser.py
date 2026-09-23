"""Tests for the third, optional layer of `read_url`: the headless browser.

No test here launches Chromium -- by choice (see `docs/decisions.md`'s entry
for this module): the pure decision functions and `_render_sync` (the actual
Playwright call) are tested directly against fakes, "Playwright not
installed" is tested by making the import fail, and the subprocess/hard-
timeout mechanism (`_run_in_subprocess`) is tested with a plain function that
hangs -- none of that needs a browser. Only the `live`-marked tests at the
bottom use a real one.
"""

from __future__ import annotations

import base64
import time

import pytest

from lakehouse_agent.web import browser


def _resolve_to(address: str):
    def resolve(host, port, *args, **kwargs):  # noqa: ANN001, ARG001
        return [(2, 1, 6, "", (address, port))]

    return resolve


# --- _allow_subrequest --------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # the machine itself
        "10.1.2.3",  # a private network
        "169.254.169.254",  # the cloud metadata service
        "::1",  # IPv6 loopback
    ],
)
def test_a_subrequest_into_our_own_network_is_refused(address):
    assert not browser._allow_subrequest(
        "https://veri.example.com/x.json", resolve=_resolve_to(address)
    )


def test_a_subrequest_to_a_global_address_is_allowed():
    assert browser._allow_subrequest(
        "https://veri.example.com/x.json", resolve=_resolve_to("93.184.216.34")
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,<b>x</b>",
    ],
)
def test_a_subrequest_with_a_non_http_scheme_is_refused(url):
    assert not browser._allow_subrequest(url, resolve=_resolve_to("93.184.216.34"))


# --- _validate_peer ------------------------------------------------------------


def test_a_rendered_response_from_an_internal_address_is_refused():
    with pytest.raises(browser.RenderError, match="ic aga"):
        browser._validate_peer(
            {"ipAddress": "169.254.169.254", "port": 443}, "https://x/e"
        )


def test_a_rendered_response_from_a_global_address_passes():
    browser._validate_peer({"ipAddress": "93.184.216.34", "port": 443}, "https://x/e")


@pytest.mark.parametrize("server_addr", [None, {}, {"ipAddress": None}])
def test_a_missing_server_address_is_not_treated_as_internal(server_addr):
    """Best effort, mirroring `web/fetch.py::_peer_address`: nothing to check
    is not the same as something refused."""
    browser._validate_peer(server_addr, "https://x/e")


# --- graceful fallback when Playwright is not installed -----------------------


def test_render_sync_reports_browser_unavailable_when_playwright_is_missing(
    monkeypatch,
):
    def _raise():
        raise ImportError("no module named playwright")

    monkeypatch.setattr(browser, "_import_sync_playwright", _raise)

    with pytest.raises(browser.BrowserUnavailable, match="kurulu degil"):
        browser._render_sync("https://x/e")


# --- how _render_sync wires a (fake) Playwright --------------------------------
#
# A fake browser/context/page, standing in for Playwright's own objects, so
# the wiring itself -- what gets registered where, on the page or on the
# context -- can be checked without a real Chromium. These tests call
# `_render_sync` directly, not the public `render()`: `render()` now runs
# `_render_sync` in a spawned child process (see the subprocess/hard-timeout
# section below), and a `monkeypatch` in this process would not reach that
# child's own fresh import of this module.
#
# Two bugs measured against a real Chromium (see `docs/decisions.md`) started
# here: `context.on("page", ...)` closing `_render_sync`'s own page before it
# was ever used, and "websocket" sitting in `_BLOCKED_RESOURCE_TYPES` where
# `page.route`/`context.route` never sees one at all. A third, `ws.close()`
# inside the WebSocket handler itself, is not wiring but a real hang -- it is
# covered by the subprocess-level tests further down, since a fake Playwright
# would not reproduce a dispatcher-fiber deadlock.


class _FakeRequest:
    def __init__(self, url: str, resource_type: str):
        self.url = url
        self.resource_type = resource_type


class _FakeRoute:
    def __init__(self, request: _FakeRequest):
        self.request = request
        self.aborted = False
        self.continued = False

    def abort(self):
        self.aborted = True

    def continue_(self):
        self.continued = True


class _FakeWebSocketRoute:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeResponse:
    def __init__(self, server_addr=None):
        self._server_addr = server_addr

    def server_addr(self):
        return self._server_addr


class _FakePage:
    def __init__(self):
        self.popup_handlers: list = []
        self.closed = False

    def on(self, event, handler):
        if event == "popup":
            self.popup_handlers.append(handler)

    def goto(self, _url, **_kwargs):
        return _FakeResponse({"ipAddress": "93.184.216.34", "port": 443})

    def content(self):
        return "<html><body>ok</body></html>"

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self):
        self.routes: list = []
        self.ws_routes: list = []
        # Any `context.on(...)` call at all is the bug: the fix registers the
        # popup handler on the page, never on the context.
        self.page_event_registrations: list = []
        self.pages: list[_FakePage] = []

    def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    def route_web_socket(self, pattern, handler):
        self.ws_routes.append((pattern, handler))

    def on(self, event, handler):
        self.page_event_registrations.append((event, handler))

    def set_default_navigation_timeout(self, _ms):
        pass

    def set_default_timeout(self, _ms):
        pass

    def new_page(self):
        page = _FakePage()
        self.pages.append(page)
        return page


class _FakeBrowser:
    def __init__(self, context: _FakeContext):
        self._context = context
        self.closed = False

    def new_context(self, **_kwargs):
        return self._context

    def close(self):
        self.closed = True


class _FakePlaywright:
    def __init__(self, fake_browser: _FakeBrowser):
        self.chromium = _FakeChromium(fake_browser)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeChromium:
    def __init__(self, fake_browser: _FakeBrowser):
        self._browser = fake_browser

    def launch(self, **_kwargs):
        return self._browser


def _wire_fake_playwright(monkeypatch) -> tuple[_FakeContext, _FakeBrowser]:
    context = _FakeContext()
    fake_browser = _FakeBrowser(context)
    monkeypatch.setattr(
        browser,
        "_import_sync_playwright",
        lambda: lambda: _FakePlaywright(fake_browser),
    )
    return context, fake_browser


def test_render_sync_never_closes_its_own_main_page(monkeypatch):
    """The bug, measured against real Chromium: `context.on("page", ...)`,
    registered before `new_page()`, fired for that very page and closed it,
    and the route handler set up on it then raised TargetClosedError."""
    context, fake_browser = _wire_fake_playwright(monkeypatch)

    browser._render_sync("https://x/e")

    assert len(context.pages) == 1
    assert context.pages[0].closed is False
    assert (
        fake_browser.closed is True
    )  # the browser is torn down; the page was not, mid-flight


def test_the_popup_handler_is_on_the_page_not_the_context(monkeypatch):
    context, _ = _wire_fake_playwright(monkeypatch)

    browser._render_sync("https://x/e")

    assert context.page_event_registrations == []
    assert len(context.pages[0].popup_handlers) == 1


def test_every_websocket_is_routed_but_left_unconnected(monkeypatch):
    """Not closed either: per Playwright's own docs, a routed WebSocket that
    never calls `connect_to_server()` already never reaches the real network
    -- it is mocked, not merely blocked. Calling `close()` from inside this
    handler was the second bug (see `docs/decisions.md`): it hung the sync
    API's dispatcher fiber against a real Chromium."""
    context, _ = _wire_fake_playwright(monkeypatch)

    browser._render_sync("https://x/e")

    assert len(context.ws_routes) == 1
    pattern, handler = context.ws_routes[0]
    assert pattern == "**/*"

    fake_ws = _FakeWebSocketRoute()
    assert handler(fake_ws) is None
    assert fake_ws.closed is False


def test_the_route_handler_blocks_images_and_addresses_check_url_refuses(monkeypatch):
    context, _ = _wire_fake_playwright(monkeypatch)

    browser._render_sync("https://x/e", resolve=_resolve_to("169.254.169.254"))

    assert len(context.routes) == 1
    pattern, handler = context.routes[0]
    assert pattern == "**/*"

    image = _FakeRoute(_FakeRequest("https://x/e/logo.png", "image"))
    handler(image)
    assert image.aborted is True

    # Every host resolves to the cloud metadata address here, so even a
    # request the resource type would allow is refused by `check_url`.
    xhr = _FakeRoute(_FakeRequest("https://x/e/data.json", "xhr"))
    handler(xhr)
    assert xhr.aborted is True


# --- the subprocess / hard-timeout mechanism -----------------------------------
#
# `render()` runs `_render_sync` in a child process (`_run_in_subprocess`) so
# that a hang inside Playwright's own dispatcher -- measured, see
# `docs/decisions.md` -- cannot outlast a deadline no exception in this
# process could otherwise enforce. That mechanism is generic and does not
# need Playwright to test: a plain function that never returns proves the
# kill is real, and one that returns or raises normally proves the ordinary
# path still works. The worker functions below must be module-level (not
# defined inside a test) -- `multiprocessing`'s spawn context re-imports them
# by name in the child rather than inheriting this process' memory.


def _worker_returns_ok(conn):
    conn.send(("ok", b"hello"))
    conn.close()


def _worker_returns_an_error(conn):
    conn.send(("error", "boom"))
    conn.close()


def _worker_hangs_forever(*_args):
    """Ignores whatever it is called with (a bare `conn`, or the `(url,
    resolve, conn)` shape `_render_worker` gets) -- used both directly and as
    a stand-in for `_render_worker`."""
    time.sleep(999)


def test_run_in_subprocess_returns_the_workers_result():
    result = browser._run_in_subprocess(_worker_returns_ok, (), timeout_s=10.0)

    assert result == ("ok", b"hello")


def test_run_in_subprocess_relays_an_error_the_worker_reports():
    result = browser._run_in_subprocess(_worker_returns_an_error, (), timeout_s=10.0)

    assert result == ("error", "boom")


def test_a_hung_subprocess_is_killed_within_the_deadline():
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="saniyede bitmedi"):
        browser._run_in_subprocess(_worker_hangs_forever, (), timeout_s=1.0)

    elapsed = time.monotonic() - started
    # Generous margin for process spawn/kill overhead; the point is that this
    # returns at all, in seconds, not that it hung for `HARD_TIMEOUT_S` or
    # longer the way the bug this replaces did.
    assert elapsed < 10.0


def test_render_reports_the_deadline_when_the_subprocess_never_answers(monkeypatch):
    """`render()`'s own wiring of `_run_in_subprocess`, without needing a
    hang inside Playwright to prove it: a worker that outlives the deadline
    turns into `RenderError`, not an uncaught `TimeoutError`."""
    monkeypatch.setattr(browser, "HARD_TIMEOUT_S", 1.0)
    monkeypatch.setattr(browser, "_render_worker", _worker_hangs_forever)

    with pytest.raises(browser.RenderError, match="saniyede bitmedi"):
        browser.render("https://x/e")


# --- live (skipped by default) -------------------------------------------


@pytest.mark.live
def test_render_reads_a_real_javascript_rendered_page():
    """Needs the `browser` extra installed and `playwright install chromium`
    run once. Run manually: `uv run pytest -m live tests/test_web_browser.py`.
    """
    html = browser.render("https://quotes.toscrape.com/js/")

    assert b'<div class="quote"' in html


@pytest.mark.live
def test_rendering_a_page_that_opens_a_websocket_does_not_hang():
    """The exact shape of the measured hang: a page whose own script opens a
    WebSocket to a refused address. Must finish well inside `HARD_TIMEOUT_S`
    rather than wedge for minutes the way it did before the fix."""
    page = b'<p>ok</p><script>new WebSocket("ws://127.0.0.1:9999/")</script>'
    encoded = base64.urlsafe_b64encode(page).decode()

    started = time.monotonic()
    html = browser.render(f"https://httpbin.org/base64/{encoded}")
    elapsed = time.monotonic() - started

    assert b"ok" in html
    assert elapsed < browser.HARD_TIMEOUT_S + 15.0
