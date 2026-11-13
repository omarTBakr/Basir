"""Frameable copies of the MLflow and Temporal UIs: a loopback proxy that lets only the Console frame them.

Both UIs send `X-Frame-Options: SAMEORIGIN`, so the Console cannot iframe them directly, and
both use absolute asset paths, so they cannot live under a path prefix. Each one therefore
gets its own loopback listener (MLFLOW_EMBED_PORT, TEMPORAL_EMBED_PORT) that forwards requests
and replaces the frame-blocking headers with `Content-Security-Policy: frame-ancestors` naming
the Console's origins only (`http://127.0.0.1:<port>` and `http://localhost:<port>`; the proxy
is another origin, so `'self'` would block the Console). `scripts/serve.py` starts them.

Security:
- Host: only loopback names are accepted (DNS rebinding), and the upstream always gets its own
  Host (e.g. `127.0.0.1:5000`), Origin and Referer, so MLflow's host and CORS checks see the
  same request they would from a local browser tab; nothing host-like comes from the client
  (`Forwarded`, `X-Forwarded-*` are dropped).
- Origin: cross-site POSTs are refused (`OriginGuard`).
- Methods: GET, HEAD and POST, plus PATCH and DELETE, which the MLflow model registry and
  Temporal schedules use; anything else (PUT, OPTIONS, TRACE, ...) is a 405.
- Redirects: `Location` is rewritten to a path on the proxy itself; a redirect to any other
  origin becomes `/`.
- Request bodies over MAX_BODY_BYTES are refused (413).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
import logging
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import httpx
from starlette.background import BackgroundTask

from basir.api.middleware import LOOPBACK_HOSTS, TEST_HOSTS, add_standard_middleware

logger = logging.getLogger("basir.embed")

TIMEOUT_S = 30.0
MAX_BODY_BYTES = 16 * 2**20
METHODS = ["GET", "HEAD", "POST", "PATCH", "DELETE"]
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
# Request headers the client must not choose: the upstream's Host/Origin/Referer are set by the proxy.
DROPPED_REQUEST = HOP_BY_HOP | {
    "host",
    "origin",
    "referer",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-port",
    "x-real-ip",
    "content-length",
}
DROPPED_RESPONSE = HOP_BY_HOP | {"x-frame-options", "content-length"}


def frame_ancestors(console_port: int) -> str:
    """The CSP directive that lets only the Console (on either loopback name) frame a page."""
    return f"frame-ancestors http://127.0.0.1:{console_port} http://localhost:{console_port}"


def _rewrite_csp(value: str, ancestors: str) -> str:
    """`value` with its frame-ancestors directive replaced by `ancestors` (added when missing)."""
    directives = [d.strip() for d in value.split(";") if d.strip()]
    kept = [d for d in directives if d.split()[0].lower() != "frame-ancestors"]
    return "; ".join([*kept, ancestors])


def _rewrite_location(value: str, upstream: str) -> str:
    """A redirect target as a path on the proxy: the upstream's own URLs keep their path, any other
    origin (absolute or `//host`) becomes `/`, so the proxy is never an open redirect."""
    # Read it the way a browser does: tabs and newlines removed, `\` as `/` (so `/\evil` is `//evil`).
    cleaned = "".join(c for c in value.strip() if c not in "\t\r\n").replace("\\", "/")
    parts = urlsplit(cleaned)
    if not parts.scheme and not parts.netloc:  # relative; `///evil` would still leave, so one leading /
        return "/" + cleaned.lstrip("/") if cleaned.startswith("/") else cleaned
    own = urlsplit(upstream)
    if (parts.scheme, parts.netloc.lower()) != (own.scheme, own.netloc.lower()):
        logger.warning("Upstream redirect to another origin replaced with /: %s", value)
        return "/"
    return "/" + urlunsplit(("", "", parts.path, parts.query, parts.fragment)).lstrip("/")


def _response_headers(headers: httpx.Headers, upstream: str, ancestors: str) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    has_ancestors = False
    for name, value in headers.multi_items():
        lower = name.lower()
        if lower in DROPPED_RESPONSE:
            continue
        if lower == "content-security-policy":
            value, has_ancestors = _rewrite_csp(value, ancestors), True
        elif lower == "location":
            value = _rewrite_location(value, upstream)
        kept.append((name, value))
    if not has_ancestors:
        kept.append(("content-security-policy", ancestors))
    return kept


def _request_headers(items: Iterable[tuple[str, str]], upstream: str) -> list[tuple[str, str]]:
    own = urlsplit(upstream)
    headers = [(k, v) for k, v in items if k.lower() not in DROPPED_REQUEST]
    headers.append(("host", own.netloc))
    return headers


def create_proxy_app(
    upstream: str, console_port: int = 8000, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """An app that forwards requests to `upstream` (e.g. http://127.0.0.1:5000), frameable only by the
    Console on `console_port`. `transport` is for tests."""
    upstream = upstream.rstrip("/")
    ancestors = frame_ancestors(console_port)
    client = httpx.AsyncClient(base_url=upstream, timeout=TIMEOUT_S, follow_redirects=False, transport=transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    add_standard_middleware(app, trusted_hosts=[*LOOPBACK_HOSTS, *TEST_HOSTS])

    @app.api_route("/{path:path}", methods=METHODS)
    async def forward(request: Request, path: str) -> Response:
        declared = request.headers.get("content-length", "0")
        if not declared.isdigit() or int(declared) > MAX_BODY_BYTES:
            return Response("request body too large", status_code=413)
        chunks, size = [], 0
        async for chunk in request.stream():  # bounded even without a Content-Length (chunked)
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return Response("request body too large", status_code=413)
            chunks.append(chunk)
        body = b"".join(chunks)
        headers = _request_headers(request.headers.items(), upstream)
        if request.method not in ("GET", "HEAD"):
            headers += [("origin", upstream), ("referer", upstream + "/")]
        outgoing = client.build_request(
            request.method,
            "/" + path.lstrip("/"),  # `//evil.example/x` stays a path on the upstream
            params=str(request.query_params),
            headers=headers,
            content=body,
        )
        try:
            answer = await client.send(outgoing, stream=True)
        except (httpx.HTTPError, httpx.InvalidURL) as e:  # InvalidURL: an unparsable upstream Location
            return Response(f"{upstream} is not reachable ({type(e).__name__})", status_code=502)
        response = StreamingResponse(
            answer.aiter_raw(), status_code=answer.status_code, background=BackgroundTask(answer.aclose)
        )
        response.raw_headers = [  # a list, so repeated headers such as Set-Cookie survive
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in _response_headers(answer.headers, upstream, ancestors)
        ]
        return response

    return app
