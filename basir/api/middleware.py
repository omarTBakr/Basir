"""Request IDs: one per HTTP request or WebSocket connection, readable anywhere via `get_request_id()`.

Pure ASGI (no body buffering, so streaming and WebSockets pass straight through). A
client-supplied `X-Request-ID` is kept when it is short and printable; otherwise a new
one is generated. The ID is echoed in the response (or WebSocket accept) headers.
"""

from __future__ import annotations

from contextvars import ContextVar
import re
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
