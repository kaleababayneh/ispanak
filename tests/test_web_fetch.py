"""Tests for the one part of `read_url` that touches the network.

No test here makes a request: the resolver and the HTTP session are stood in
for, which is also the only way to test what the guards refuse -- resolving a
name to 127.0.0.1 for real would depend on the developer's DNS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from lakehouse_agent.web.fetch import (
    MAX_BYTES,
    WebFetchError,
    cache_path,
    check_url,
    fetch,
    redact_url,
)


def _resolve_to(address: str):
    def resolve(host, port, *args, **kwargs):  # noqa: ANN001, ARG001
        return [(2, 1, 6, "", (address, port))]

    return resolve


class _Response:
    def __init__(self, *, status=200, headers=None, body=b"", chunks=None):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [body]
        self.closed = False

    def iter_content(self, _size):
        yield from self._chunks

    def close(self):
        self.closed = True


class _Session:
    """A requests.Session stand-in that answers from a list."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.urls: list[str] = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        if not self._responses:
            raise AssertionError(f"beklenmeyen istek: {url}")
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


# --- the guards --------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("file:///etc/passwd", "http ve https"),
        ("data:text/html,<b>x</b>", "http ve https"),
        ("https://", "alan adi yok"),
    ],
)
def test_only_http_addresses_are_read(url, match):
    with pytest.raises(WebFetchError, match=match):
        check_url(url, resolve=_resolve_to("93.184.216.34"))


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # the machine itself
        "10.1.2.3",  # a private network
        "169.254.169.254",  # the cloud metadata service
    ],
)
def test_an_address_inside_our_own_network_is_refused(address):
    """A URL comes from the question, so it can point at us. The check is on
    the resolved address, not on the name: "metadata.example.com" can resolve
    to 169.254.169.254 as easily as "localhost" does to 127.0.0.1."""
    with pytest.raises(WebFetchError, match="ic aga"):
        check_url("https://veri.example.com/x.csv", resolve=_resolve_to(address))


def test_a_name_that_does_not_resolve_says_so():
    def resolve(*_args, **_kwargs):
        raise OSError("nodename nor servname provided")

    with pytest.raises(WebFetchError, match="cozulemedi"):
        check_url("https://yok.example.com", resolve=resolve)


# --- fetching ----------------------------------------------------------------


def test_the_body_and_the_request_are_stored_side_by_side(tmp_path):
    session = _Session(
        _Response(
            headers={"Content-Type": "text/csv; charset=utf-8"}, body=b"a,b\n1,2\n"
        )
    )

    got = fetch(
        "https://veri.example.com/x.csv",
        root=tmp_path,
        session=session,
        resolve=_resolve_to("93.184.216.34"),
    )

    assert got.body == b"a,b\n1,2\n"
    assert got.content_type == "text/csv"  # the charset is not part of it
    assert got.path is not None and got.path.read_bytes() == got.body
    record = json.loads(got.path.with_suffix(got.path.suffix + ".json").read_text())
    assert record["url"] == "https://veri.example.com/x.csv"
    assert record["http_status"] == 200 and record["bytes"] == len(got.body)
    assert record["fetched_at"]


def test_a_second_read_of_the_same_address_makes_no_request(tmp_path):
    """What makes a follow-up question about a document free -- and what lets
    the tool hand back another of its tables without downloading again."""
    url = "https://veri.example.com/x.csv"
    session = _Session(
        _Response(headers={"Content-Type": "text/csv"}, body=b"a,b\n1,2\n")
    )
    first = fetch(
        url, root=tmp_path, session=session, resolve=_resolve_to("93.184.216.34")
    )

    empty = _Session()  # raises on any request
    second = fetch(
        url, root=tmp_path, session=empty, resolve=_resolve_to("93.184.216.34")
    )

    assert second.body == first.body
    assert second.from_cache and not first.from_cache
    assert empty.urls == []


def test_a_file_past_the_cap_is_refused_while_it_is_still_arriving(tmp_path):
    """Checked chunk by chunk, not from Content-Length: that header is
    optional, and a server that under-reports it would stream us anything."""
    chunks = [b"x" * (1024 * 1024)] * 11
    session = _Session(_Response(headers={"Content-Type": "text/plain"}, chunks=chunks))

    with pytest.raises(WebFetchError, match="MB sinirini asiyor"):
        fetch(
            "https://veri.example.com/big.csv",
            root=tmp_path,
            session=session,
            resolve=_resolve_to("93.184.216.34"),
        )

    assert not list(tmp_path.iterdir())  # nothing half-written is kept
    assert sum(len(chunk) for chunk in chunks) > MAX_BYTES


def test_a_redirect_is_followed_and_checked_again(tmp_path):
    """The guard is per hop: a public address is allowed to redirect, and the
    place it redirects to is a new address that has to pass the same check."""
    session = _Session(
        _Response(status=302, headers={"Location": "/sonraki.csv"}),
        _Response(headers={"Content-Type": "text/csv"}, body=b"a,b\n1,2\n"),
    )

    got = fetch(
        "https://veri.example.com/ilk.csv",
        root=tmp_path,
        session=session,
        resolve=_resolve_to("93.184.216.34"),
    )

    assert session.urls == [
        "https://veri.example.com/ilk.csv",
        "https://veri.example.com/sonraki.csv",
    ]
    assert got.final_url.endswith("/sonraki.csv")
    assert got.url.endswith("/ilk.csv")  # what was asked for is kept


def test_a_redirect_into_our_own_network_is_refused(tmp_path):
    def resolve(host, port, *_args, **_kwargs):
        address = "127.0.0.1" if host == "ic.example.com" else "93.184.216.34"
        return [(2, 1, 6, "", (address, port))]

    session = _Session(
        _Response(status=302, headers={"Location": "https://ic.example.com/x"})
    )

    with pytest.raises(WebFetchError, match="ic aga"):
        fetch(
            "https://veri.example.com/ilk.csv",
            root=tmp_path,
            session=session,
            resolve=resolve,
        )


def test_an_error_status_is_reported_not_stored(tmp_path):
    session = _Session(_Response(status=404, body=b"yok"))

    with pytest.raises(WebFetchError, match="404"):
        fetch(
            "https://veri.example.com/yok.csv",
            root=tmp_path,
            session=session,
            resolve=_resolve_to("93.184.216.34"),
        )

    assert not list(tmp_path.iterdir())


def test_a_connection_failure_becomes_a_sentence(tmp_path):
    session = _Session(requests.ConnectionError("dns"))

    with pytest.raises(WebFetchError, match="okunamadi"):
        fetch(
            "https://veri.example.com/x.csv",
            root=tmp_path,
            session=session,
            resolve=_resolve_to("93.184.216.34"),
        )


def test_two_addresses_never_share_a_cache_file(tmp_path):
    """A page and the PDF it links to can both be called `veriler.pdf`, and a
    query string cannot be a path."""
    first = cache_path("https://a.example.com/veriler.pdf", root=tmp_path)
    second = cache_path("https://b.example.com/veriler.pdf", root=tmp_path)
    query = cache_path("https://a.example.com/rapor?yil=2026", root=tmp_path)

    assert first != second
    assert first.suffix == ".pdf" and query.suffix == ""


# --- what the 23 Sep review found --------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "100.64.1.1",  # carrier-grade NAT: the default pod network on EKS
        "100.127.255.254",
        "192.0.0.1",  # IETF protocol assignments
        "198.18.0.1",  # benchmarking range
    ],
)
def test_addresses_that_are_not_globally_routable_are_refused(address):
    """The guard used to be five separate predicates (private, loopback,
    link-local, reserved, multicast). Every one of them returns False for
    100.64.1.1, so carrier-grade NAT went straight through. `is_global` is
    the property that actually means "on the public internet"."""
    with pytest.raises(WebFetchError, match="ic aga"):
        check_url(f"http://{address}/x.csv", resolve=_resolve_to(address))


def test_a_host_that_resolves_to_nothing_is_refused():
    """An empty answer skipped the loop body entirely and returned the host
    as checked."""

    def resolve(host, port, *args, **kwargs):  # noqa: ANN001, ARG001
        return []

    with pytest.raises(WebFetchError, match="hicbir adrese cozulmedi"):
        check_url("http://bos.example.com/x.csv", resolve=resolve)


def test_a_connection_that_lands_somewhere_else_is_refused_before_the_body():
    """DNS rebinding: the guard resolves the name, then hands the *name* to
    the HTTP client, which resolves it again. A record that changes between
    the two defeats the pre-flight check entirely, so the socket's actual
    peer is checked once more before a byte of the body is read."""

    class _Sock:
        def getpeername(self):
            return ("169.254.169.254", 80)

    class _Raw:
        _connection = type("c", (), {"sock": _Sock()})()

    response = _Response(body=b"secret")
    response.raw = _Raw()

    with pytest.raises(WebFetchError, match="ic aga"):
        fetch(
            "http://rebind.example.com/x.csv",
            root=Path("/nonexistent"),
            session=_Session(response),
            resolve=_resolve_to("93.184.216.34"),
        )


def test_a_key_in_the_address_is_redacted_everywhere_it_is_stored(tmp_path):
    """CLAUDE.md: an address that is logged or stored has its key removed.
    `read_url` takes its address from a prompt, so a key-bearing one is the
    plausible case, and the record sits in the repo tree."""
    url = "https://veri.example.com/rapor.csv?key=SUPERSECRET&yil=2026"
    document = fetch(
        url,
        root=tmp_path,
        session=_Session(_Response(body=b"a,b\n1,2\n")),
        resolve=_resolve_to("93.184.216.34"),
    )

    record = json.loads(
        document.path.with_suffix(document.path.suffix + ".json").read_text()
    )
    assert "SUPERSECRET" not in json.dumps(record)
    assert record["url"].endswith("key=***&yil=2026")
    # The raw address is still what was fetched, and still what is cached.
    assert document.url == url


def test_redaction_leaves_ordinary_parameters_alone():
    assert redact_url("https://x/y?yil=2026") == "https://x/y?yil=2026"
    assert redact_url("https://x/y") == "https://x/y"
    assert redact_url("https://x/y?token=abc&a=1") == "https://x/y?token=***&a=1"
