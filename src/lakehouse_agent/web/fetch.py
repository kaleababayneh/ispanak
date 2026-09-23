"""Download what a URL points at, once, with the limits a demo needs.

The one place in the project that fetches an address a *person typed*, rather
than an endpoint we wrote code against. That changes the threat model, so the
guards here are not decoration:

* only http and https -- `file://` would read the developer's disk and
  `data:` would let a prompt smuggle content in as if it came from the web;
* the host is resolved before the request and every redirect is checked the
  same way, so an address that resolves to localhost, a private network or a
  link-local address (169.254.169.254 is the cloud metadata service) is
  refused rather than fetched from inside our own network;
* the body is read in chunks against a byte cap, so a 2 GB file cannot take
  the process down. The timeouts are requests' own (connect, then per-read);
  they are not a deadline on the whole call, so a server that trickles bytes
  can outlast the step that started it. The step timeout is what ends the
  turn; it does not stop the thread (`loop.py`), which is why the byte cap
  matters more than the clock here.

What comes back is stored under `data/adhoc/url/` with the request beside it
(url, status, content type, time) -- bronze discipline for demo-time data: the
bytes are kept as they arrived, and a second read of the same URL makes no
call at all.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

ADHOC_ROOT = Path("data/adhoc/url")

#: 10 MB. The demo's own targets are a 140 KB PDF and an 85 KB page; a file
#: past this is not something an answer is going to be built on in one turn.
MAX_BYTES = 10 * 1024 * 1024
#: The whole fetch, redirects included. `read_url` runs inside a step that is
#: itself capped at 60 s (AgentConfig.step_timeout_s), so this must not exceed
#: it: a fetch that dies with the step tells the model nothing.
TIMEOUT_S = 50.0
CONNECT_TIMEOUT_S = 10.0
MAX_REDIRECTS = 5

CHUNK = 64 * 1024

#: Sent so the request is not mistaken for a scraper hiding itself. Honest
#: about what it is; no cookies and no credentials are ever attached.
USER_AGENT = "lakehouse-agent/1.0 (KKB Hackathon 2026; +https://github.com/)"


class WebFetchError(RuntimeError):
    """The address could not be fetched, and why, in a sentence the model can
    pass on to the person."""


@dataclass(frozen=True)
class Fetched:
    """One downloaded document."""

    url: str
    final_url: str
    status: int
    content_type: str
    body: bytes
    fetched_at: str
    path: Path | None
    from_cache: bool = False

    @property
    def filename(self) -> str:
        return Path(urlparse(self.final_url).path).name


#: Query parameters whose value is a secret. A person pasting an address into
#: the prompt may well paste one with a key in it, and this module stores the
#: address next to the bytes and hands it to the model as a source.
_SECRET_PARAMS = frozenset(
    {
        "key",
        "apikey",
        "api_key",
        "token",
        "access_token",
        "auth",
        "password",
        "secret",
        "signature",
        "sig",
    }
)


_ANY_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+")
#: Edits a model may make to an address it was copying before we stop
#: trusting that it meant the one in the question. Three covers the case
#: measured 23 Sep (one character) with room for a dropped or doubled slash;
#: a bigger distance starts matching sibling pages.
URL_SNAP_EDITS = 3


def _edits(a: str, b: str) -> int:
    """Levenshtein distance. Addresses are short; the plain table is fine."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def nearest_url(url: str, text: str, *, max_edits: int = URL_SNAP_EDITS) -> str | None:
    """The address in `text` that `url` is a slightly wrong copy of, or None.

    The model retypes an address from the question rather than copying it,
    and a long one comes back altered. Measured 23 Sep: TCMB's
    `.../main+menu/duyurular/...` was passed to `read_url` as
    `.../main+menu+duyurular/...` -- the address already held two plus
    signs and the model continued the pattern -- and the fetch failed. The
    person's own text is the ground truth for what they asked to be read.

    None when `url` appears in `text` as written (nothing to correct), when
    `text` holds no address on the same host, or when the nearest one is
    further than `max_edits` away. Host is compared exactly: a one-letter
    slip in a host name is a different site, not a typo to be helpful about.
    """
    if not text or url in text:
        return None
    host = urlparse(url).netloc.lower()
    best: tuple[int, str] | None = None
    for candidate in _ANY_URL.findall(text):
        candidate = candidate.rstrip(".,;:")
        if candidate == url or urlparse(candidate).netloc.lower() != host:
            continue
        distance = _edits(url, candidate)
        if distance <= max_edits and (best is None or distance < best[0]):
            best = (distance, candidate)
    return best[1] if best else None


def redact_url(url: str) -> str:
    """`?key=abc123` -> `?key=***`, for anything stored, logged or shown.

    CLAUDE.md's rule is written for `fetch_log`, but it is about what lands on
    disk and in front of a person, and this module does both. The raw address
    is still used to make the request and to resolve relative links; only the
    copies leave redacted.
    """
    parsed = urlparse(url)
    if not parsed.query:
        return url
    parts = []
    for piece in parsed.query.split("&"):
        name, sep, _value = piece.partition("=")
        parts.append(f"{name}=***" if sep and name.lower() in _SECRET_PARAMS else piece)
    return urlunparse(parsed._replace(query="&".join(parts)))


def _refuse_if_internal(address: str, host: str) -> None:
    """Raise unless `address` is a public, routable one.

    `not is_global` rather than a list of `is_private`/`is_loopback`/...:
    those five miss carrier-grade NAT (100.64.0.0/10, measured -- every one of
    them returns False for 100.64.1.1), which is the default pod network on
    several container platforms, and they would miss whatever range comes
    next. `is_global` is the property that means "routable on the public
    internet", which is the only thing this tool has any business reaching.
    """
    parsed = ipaddress.ip_address(address)
    if not parsed.is_global:
        raise WebFetchError(f"{host} ic aga ({parsed}) cozuluyor; bu adres okunmaz.")


def check_url(url: str, *, resolve=socket.getaddrinfo) -> str:
    """Return the host, or raise `WebFetchError`. See the module docstring."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise WebFetchError(
            f"Yalnizca http ve https adresleri okunabilir; verilen sema: "
            f"{parsed.scheme or 'yok'}."
        )
    host = parsed.hostname
    if not host:
        raise WebFetchError("Adreste alan adi yok.")
    try:
        infos = resolve(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError as exc:
        raise WebFetchError(f"Alan adi cozulemedi: {host}") from exc
    if not infos:
        raise WebFetchError(f"Alan adi hicbir adrese cozulmedi: {host}")
    for info in infos:
        try:
            _refuse_if_internal(info[4][0], host)
        except ValueError as exc:  # not an IP literal at all
            raise WebFetchError(f"{host} icin gecersiz adres dondu.") from exc
    return host


def _peer_address(response: requests.Response) -> str | None:
    """The address the socket actually connected to, if it can be read.

    `check_url` resolves the name itself and then hands the *name* to the HTTP
    client, which resolves it again. A DNS answer that changes between the two
    lookups defeats the guard entirely (classic rebinding), so the connection
    is checked once more here, before a single byte of the body is read.
    Best effort: with a stubbed session, or a urllib3 that keeps its socket
    elsewhere, there is nothing to read and the pre-flight check stands alone.
    """
    connection = getattr(getattr(response, "raw", None), "_connection", None)
    sock = getattr(connection, "sock", None)
    try:
        peer = sock.getpeername() if sock is not None else None
    except OSError:  # pragma: no cover - closed between reading and asking
        return None
    return str(peer[0]) if peer else None


def cache_path(url: str, *, root: Path = ADHOC_ROOT) -> Path:
    """Where this URL's bytes live. Named by the hash of the address: a page
    and its PDF can share a filename, and a query string cannot be a path."""
    digest = sha256(url.strip().encode()).hexdigest()[:16]
    suffix = Path(urlparse(url).path).suffix.lower()
    if len(suffix) > 6 or not suffix[1:].isalnum():
        suffix = ""
    return root / f"{digest}{suffix}"


def fetch(
    url: str,
    *,
    root: Path = ADHOC_ROOT,
    force: bool = False,
    session: requests.Session | None = None,
    resolve=socket.getaddrinfo,
) -> Fetched:
    """Download `url` (or read what an earlier call stored) and return it.

    A stored document is returned without a request, which is what makes a
    second look at the same PDF free -- and what lets the tool hand back
    another table from a page it already read.
    """
    url = url.strip()
    path = cache_path(url, root=root)
    record_path = path.with_suffix(path.suffix + ".json")
    if not force and path.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return Fetched(
            url=url,
            final_url=record.get("final_url", url),
            status=int(record.get("http_status", 200)),
            content_type=record.get("content_type", ""),
            body=path.read_bytes(),
            fetched_at=record.get("fetched_at", ""),
            path=path,
            from_cache=True,
        )

    http = session or requests.Session()
    current = url
    response = None
    try:
        for _hop in range(MAX_REDIRECTS + 1):
            check_url(current, resolve=resolve)
            response = http.get(
                current,
                headers={"User-Agent": USER_AGENT},
                timeout=(CONNECT_TIMEOUT_S, TIMEOUT_S),
                allow_redirects=False,
                stream=True,
            )
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise WebFetchError("Yonlendirme adresi bos.")
                current = requests.compat.urljoin(current, location)
                continue
            break
        else:
            raise WebFetchError(f"{MAX_REDIRECTS} yonlendirmeden sonra durulmadi.")

        if response is None:  # pragma: no cover - the loop always assigns
            raise WebFetchError("Istek yapilamadi.")
        if response.status_code != 200:
            raise WebFetchError(
                f"Adres {response.status_code} dondurdu: {redact_url(current)}",
            )
        peer = _peer_address(response)
        if peer is not None:
            _refuse_if_internal(peer, urlparse(current).hostname or current)
        body = _read_capped(response)
        content_type = (
            (response.headers.get("Content-Type") or "").split(";")[0].strip()
        )
    except requests.RequestException as exc:
        raise WebFetchError(
            f"Adres okunamadi ({type(exc).__name__}): {redact_url(url)}"
        ) from exc
    finally:
        if response is not None:
            response.close()

    fetched_at = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    record_path.write_text(
        json.dumps(
            {
                # Redacted on the way to disk: this file sits in the repo tree
                # and a pasted address can carry a key.
                "url": redact_url(url),
                "final_url": redact_url(current),
                "http_status": response.status_code,
                "content_type": content_type,
                "fetched_at": fetched_at,
                "bytes": len(body),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        "web: %s okundu (%d bayt, %s)",
        redact_url(url),
        len(body),
        content_type or "?",
    )
    return Fetched(
        url=url,
        final_url=current,
        status=200,
        content_type=content_type,
        body=body,
        fetched_at=fetched_at,
        path=path,
    )


def _read_capped(response: requests.Response) -> bytes:
    """The body, refused as soon as it passes MAX_BYTES.

    Checked while reading rather than from Content-Length: that header is
    optional, and a server that under-reports it would otherwise stream us
    whatever it liked.
    """
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(CHUNK):
        total += len(chunk)
        if total > MAX_BYTES:
            raise WebFetchError(
                f"Dosya {MAX_BYTES // (1024 * 1024)} MB sinirini asiyor; okunmadi."
            )
        chunks.append(chunk)
    return b"".join(chunks)
