"""Frameable copies of the MLflow and Temporal UIs: a pass-through proxy that drops the frame-blocking headers.

Both UIs send `X-Frame-Options: SAMEORIGIN`, so the Console cannot iframe them directly, and
both use absolute asset paths, so they cannot live under a path prefix. Each one therefore
gets its own loopback listener (MLFLOW_EMBED_PORT, TEMPORAL_EMBED_PORT) that forwards every
request unchanged and strips only those headers. `scripts/serve.py` starts them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import httpx
from starlette.background import BackgroundTask

TIMEOUT_S = 30.0
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
DROPPED_RESPONSE = HOP_BY_HOP | {"x-frame-options", "content-length"}


def _response_headers(headers: httpx.Headers, upstream: str, public: str) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for name, value in headers.multi_items():
        lower = name.lower()
        if lower in DROPPED_RESPONSE:
            continue
        if lower == "content-security-policy" and "frame-ancestors" in value:
            continue
        if lower == "location":
            value = value.replace(upstream, public)
        kept.append((name, value))
    return kept


def create_proxy_app(upstream: str) -> FastAPI:
    """An app that forwards every request to `upstream` (e.g. http://127.0.0.1:5000) and is frameable."""
    upstream = upstream.rstrip("/")
    upstream_host = urlsplit(upstream).netloc
    client = httpx.AsyncClient(base_url=upstream, timeout=TIMEOUT_S, follow_redirects=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def forward(request: Request, path: str) -> Response:
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP | {"host"}]
        headers.append(("host", upstream_host))
        outgoing = client.build_request(
            request.method,
            "/" + path,
            params=str(request.query_params),
            headers=headers,
            content=await request.body(),
        )
        try:
            answer = await client.send(outgoing, stream=True)
        except httpx.HTTPError as e:
            return Response(f"{upstream} is not reachable ({type(e).__name__})", status_code=502)
        public = f"{request.url.scheme}://{request.url.netloc}"
        response = StreamingResponse(
            answer.aiter_raw(), status_code=answer.status_code, background=BackgroundTask(answer.aclose)
        )
        response.raw_headers = [  # a list, so repeated headers such as Set-Cookie survive
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in _response_headers(answer.headers, upstream, public)
        ]
        return response

    return app
