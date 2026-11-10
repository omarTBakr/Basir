"""Tests for basir/api/middleware.py (request IDs over HTTP and WebSocket)."""

from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from basir.api.middleware import RequestIDMiddleware, get_request_id


def make_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/id")
    def read_id() -> dict[str, str | None]:
        return {"id": get_request_id()}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text(get_request_id() or "")
        await websocket.close()

    return TestClient(app)


def test_new_id_per_request_echoed_in_header() -> None:
    client = make_client()
    first, second = client.get("/id"), client.get("/id")
    assert first.json()["id"] == first.headers["x-request-id"]
    assert len(first.json()["id"]) == 32 and first.json()["id"] != second.json()["id"]


def test_valid_client_id_is_kept_and_unsafe_one_replaced() -> None:
    client = make_client()
    assert client.get("/id", headers={"X-Request-ID": "phone-1.42"}).json()["id"] == "phone-1.42"
    replaced = client.get("/id", headers={"X-Request-ID": "bad id\twith spaces"}).json()["id"]
    assert replaced != "bad id\twith spaces" and len(replaced) == 32


def test_websocket_gets_an_id() -> None:
    with make_client().websocket_connect("/ws", headers={"X-Request-ID": "ws-7"}) as ws:
        assert ws.receive_text() == "ws-7"


def test_no_id_outside_a_request() -> None:
    assert get_request_id() is None
