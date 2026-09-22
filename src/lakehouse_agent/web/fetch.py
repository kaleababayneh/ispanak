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
  the process down, and the whole call is held to one deadline.

What comes back is stored under `data/adhoc/url/` with the request beside it
(url, status, content type, time) -- bronze discipline for demo-time data: the
bytes are kept as they arrived, and a second read of the same URL makes no
call at all.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

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
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise WebFetchError(
                f"{host} ic aga ({address}) cozuluyor; bu adres okunmaz."
            )
    return host


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
                f"Adres {response.status_code} dondurdu: {current}",
            )
        body = _read_capped(response)
        content_type = (
            (response.headers.get("Content-Type") or "").split(";")[0].strip()
        )
    except requests.RequestException as exc:
        raise WebFetchError(f"Adres okunamadi ({type(exc).__name__}): {url}") from exc
    finally:
        if response is not None:
            response.close()

    fetched_at = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    record_path.write_text(
        json.dumps(
            {
                "url": url,
                "final_url": current,
                "http_status": 200,
                "content_type": content_type,
                "fetched_at": fetched_at,
                "bytes": len(body),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("web: %s okundu (%d bayt, %s)", url, len(body), content_type or "?")
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
