"""Tests for utilities/middleware.py (request ID + timing, HTTP and WebSocket)."""

from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
import pytest

from utilities.middleware import RequestContextMiddleware, get_request_id


@pytest.fixture
def stream_gate() -> threading.Event:
    return threading.Event()


@pytest.fixture
def client(stream_gate: threading.Event) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str | None]:
        return {"request_id": get_request_id()}

    @app.get("/error")
    def error() -> None:
        raise RuntimeError("Simulated internal error")

    @app.get("/stream")
    def stream() -> StreamingResponse:
        async def events():
            yield "data: first\n\n"
            # Finishes only after the test has read "first"; a buffering middleware
            # would never deliver it and the test would hang.
            while not stream_gate.is_set():
                await asyncio.sleep(0.01)
            yield "data: second\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.websocket("/ws/stream")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"request_id": get_request_id()})
        while (msg := await websocket.receive_text()) != "bye":
            await websocket.send_text(f"echo:{msg}")
        await websocket.close()

    return TestClient(app, raise_server_exceptions=False)


def test_http_response_gets_request_id_and_timing(client: TestClient) -> None:
    res = client.get("/ping")
    rid = res.headers["x-request-id"]
    assert res.status_code == 200 and len(rid) == 32
    assert float(res.headers["x-process-time-ms"]) >= 0.0
    assert res.json()["request_id"] == rid  # same ID inside the handler


def test_incoming_request_id_is_echoed(client: TestClient) -> None:
    res = client.get("/ping", headers={"X-Request-ID": "phone-42"})
    assert res.headers["x-request-id"] == "phone-42"
    assert res.json()["request_id"] == "phone-42"


def test_ids_differ_and_do_not_leak(client: TestClient) -> None:
    assert client.get("/ping").headers["x-request-id"] != client.get("/ping").headers["x-request-id"]
    assert get_request_id() is None


def test_streaming_response_is_not_buffered(client: TestClient, stream_gate: threading.Event) -> None:
    with client.stream("GET", "/stream") as res:
        assert "x-request-id" in res.headers
        lines = (line for line in res.iter_lines() if line)
        assert next(lines) == "data: first"  # arrived while the generator was still running
        stream_gate.set()
        assert list(lines) == ["data: second"]


def test_websocket_passes_through_with_request_id(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream", headers={"X-Request-ID": "ws-session-1"}) as ws:
        assert ws.receive_json() == {"request_id": "ws-session-1"}
        ws.send_text("frame-1")
        assert ws.receive_text() == "echo:frame-1"
        ws.send_text("bye")


def test_websocket_gets_generated_id(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        assert len(ws.receive_json()["request_id"]) == 32
        ws.send_text("bye")


def test_errors_return_500_and_are_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="basir.request"):
        assert client.get("/error").status_code == 500
    assert any("/error failed" in r.getMessage() for r in caplog.records)


def test_one_log_line_per_request(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="basir.request"):
        client.get("/ping")
    lines = [r.getMessage() for r in caplog.records if r.name == "basir.request"]
    assert len(lines) == 1 and "GET /ping -> 200" in lines[0]
