"""Request context middleware for Basir.

Gives every HTTP request and WebSocket session a request ID and one timing log line.
Bodies are never read or buffered, so streaming responses and the phone's frame
WebSocket pass through untouched. Detailed tracing belongs to Langfuse (queries,
LLM / VLM calls) and MLflow (evaluation runs); attach `get_request_id()` to those.

    app.add_middleware(RequestContextMiddleware)
"""

from __future__ import annotations

from contextvars import ContextVar
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("basir.request")

REQUEST_ID_HEADER = b"x-request-id"
_request_id: ContextVar[str | None] = ContextVar("basir_request_id", default=None)


def get_request_id() -> str | None:
    """ID of the request or WebSocket session being handled, or None outside one."""
    return _request_id.get()


class RequestContextMiddleware:
    """Attach a request ID and timing to every HTTP request and WebSocket session."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope)
        token = _request_id.set(request_id)
        start = time.perf_counter()
        status: int | None = None

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                elapsed_ms = (time.perf_counter() - start) * 1000
                message["headers"] = [
                    *message.get("headers", []),
                    (REQUEST_ID_HEADER, request_id.encode("latin-1")),
                    (b"x-process-time-ms", f"{elapsed_ms:.2f}".encode()),
                ]
            await send(message)

        label = f"[{request_id[:8]}] {scope.get('method', 'WS')} {scope.get('path')}"
        try:
            await self.app(scope, receive, send_with_headers if scope["type"] == "http" else send)
        except Exception:
            logger.exception("%s failed after %.1f ms", label, _ms_since(start))
            raise
        else:
            outcome = f"-> {status}" if scope["type"] == "http" else "closed"
            logger.info("%s %s (%.1f ms)", label, outcome, _ms_since(start))
        finally:
            _request_id.reset(token)


def _incoming_request_id(scope: Scope) -> str:
    """The client's X-Request-ID (truncated to 128 chars), or a new uuid4 hex."""
    for name, value in scope.get("headers") or []:
        if name.lower() == REQUEST_ID_HEADER and value:
            return value.decode("latin-1")[:128]
    return uuid.uuid4().hex


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
