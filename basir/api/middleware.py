"""Request IDs, Host/Origin checks and log redaction for the Console and the phone listener.

Request IDs: one per HTTP request or WebSocket connection, readable anywhere via
`get_request_id()`. Pure ASGI (no body buffering, so streaming and WebSockets pass straight
through). A client-supplied `X-Request-ID` is kept when it is short and printable; otherwise a
new one is generated. The ID is echoed in the response (or WebSocket accept) headers.

`OriginGuard` refuses cross-origin WebSockets and POSTs (and optionally checks the Host);
`RedactSecretsFilter` keeps device tokens out of uvicorn's log lines.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
import ipaddress
import logging
import re
from urllib.parse import urlsplit
import uuid

from starlette.applications import Starlette
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("basir.security")

HEADER = b"x-request-id"
_VALID = re.compile(rb"^[A-Za-z0-9._:-]{1,128}$")
_request_id: ContextVar[str | None] = ContextVar("basir_request_id", default=None)


def get_request_id() -> str | None:
    """The current request's ID, or None outside a request."""
    return _request_id.get()


class RequestIDMiddleware:
    """Assign each HTTP request and WebSocket connection a request ID."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        supplied = dict(scope["headers"]).get(HEADER, b"")
        request_id = supplied.decode() if _VALID.match(supplied) else uuid.uuid4().hex
        start_type = "http.response.start" if scope["type"] == "http" else "websocket.accept"

        async def send_with_id(message: Message) -> None:
            if message["type"] == start_type:
                message["headers"] = [*message.get("headers", []), (HEADER, request_id.encode())]
            await send(message)

        token = _request_id.set(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _request_id.reset(token)


# ---- Host and Origin checks ------------------------------------------------------------------

LOOPBACK_HOSTS = ["127.0.0.1", "localhost"]
# Starlette's TestClient sends `Host: testserver`. A browser only sends it if the bare name
# `testserver` resolves on this laptop, which a website cannot arrange through public DNS.
TEST_HOSTS = ["testserver"]
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
SAFE_FETCH_SITES = frozenset({b"same-origin", b"none"})


def _host_only(netloc: str) -> str:
    """`127.0.0.1:8000` -> `127.0.0.1`, `[::1]:8000` -> `[::1]` (lower-case)."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[: netloc.find("]") + 1]
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def ip_or_local_host(host: str) -> bool:
    """A Host for the phone listener: an IP literal (phones connect by LAN address), localhost or `*.local`.

    DNS rebinding needs a domain name, so allowing only IP literals keeps phones working
    while a rebinding name such as `evil.example` is refused.
    """
    name = _host_only(host).strip("[]")
    if name == "localhost" or name.endswith(".local"):
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


class OriginGuard:
    """Refuse cross-origin browser requests: WebSocket handshakes and unsafe methods (POST, PUT, PATCH, DELETE).

    Browsers always send `Origin` on WebSocket handshakes and on cross-origin POSTs, so a
    request whose `Origin` is not this server's own origin (the same host and port as the
    `Host` header) is refused with 403 (a WebSocket is closed before accept, which the client
    also sees as 403). `Origin: null` (sandboxed frames, file://) is refused too. A request
    with no `Origin` comes from a non-browser client (the phone simulator, curl, tests) and
    passes unless `Sec-Fetch-Site` says another site sent it. HTMX posts and the Console's
    own sockets are same-origin and pass.

    `allowed_host`, when given, must accept the `Host` header or the request gets 400; it is
    for listeners where `TrustedHostMiddleware`'s fixed list does not fit (phones: any IP).
    """

    def __init__(self, app: ASGIApp, allowed_host: Callable[[str], bool] | None = None) -> None:
        self.app = app
        self.allowed_host = allowed_host

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin-1")
        if self.allowed_host is not None and not self.allowed_host(host):
            logger.warning("Refused host %r (request %s)", host, get_request_id())
            await _refuse(scope, receive, send, 400, "invalid host header")
            return
        checked = scope["type"] == "websocket" or scope["method"] in UNSAFE_METHODS
        if checked and not same_origin(headers, host):
            logger.warning(
                "Refused cross-origin %s %s from origin %r (request %s)",
                scope.get("method", "WebSocket"),
                scope.get("path", ""),
                headers.get(b"origin", b"").decode("latin-1"),
                get_request_id(),
            )
            await _refuse(scope, receive, send, 403, "cross-origin request refused")
            return
        await self.app(scope, receive, send)


def same_origin(headers: dict[bytes, bytes], host: str) -> bool:
    """True for a request with no Origin (non-browser) or an Origin whose host:port equals `host`."""
    origin = headers.get(b"origin")
    if origin is None:
        return headers.get(b"sec-fetch-site", b"none") in SAFE_FETCH_SITES
    parts = urlsplit(origin.decode("latin-1"))
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False  # includes "null"
    return parts.netloc.lower() == host.strip().lower()


def add_standard_middleware(
    app: Starlette, *, trusted_hosts: list[str] | None = None, allowed_host: Callable[[str], bool] | None = None
) -> None:
    """The stack the Console, the phone listener and the embed proxies share, outermost first:
    request ID, HTTP metrics, `TrustedHostMiddleware(trusted_hosts)` (when given), `OriginGuard`.

    Starlette runs the middleware added last first, hence the reverse order. Refused requests
    are still counted in the metrics and carry a request ID.
    """
    from basir.api.metrics import HTTPMetricsMiddleware

    app.add_middleware(OriginGuard, allowed_host=allowed_host)
    if trusted_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    app.add_middleware(HTTPMetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)


async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, reason: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008, "reason": reason})
        return
    await PlainTextResponse(reason, status_code=status)(scope, receive, send)


# ---- Log redaction ---------------------------------------------------------------------------

_SECRET = re.compile(r"(?i)(token=)[^&\s\"']+|bsr_[A-Za-z0-9_-]+")


class RedactSecretsFilter(logging.Filter):
    """Replace device tokens (`bsr_...`, `token=...`) in a log record's message with `[redacted]`.

    `scripts/serve.py` installs it on uvicorn's loggers: uvicorn logs every WebSocket path
    with its query string, so an old client sending `?token=` would otherwise leave its
    token in the log even though the server refuses it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _SECRET.sub(lambda m: (m.group(1) or "") + "[redacted]", message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def redact_uvicorn_logs() -> None:
    """Add `RedactSecretsFilter` to uvicorn's loggers (idempotent). Logger filters only see records
    logged on that logger itself, hence each name."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log = logging.getLogger(name)
        if not any(isinstance(f, RedactSecretsFilter) for f in log.filters):
            log.addFilter(RedactSecretsFilter())
