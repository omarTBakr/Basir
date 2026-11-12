"""Tests for basir/api/ingest.py and devices.py: two phones over WebSocket, drops and result routing."""

from __future__ import annotations

from collections.abc import Callable
import json
import threading
import time

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from basir.api.protocol import FrameHeader, encode_frame
from tests.conftest import FakeBackend

_, _JPEG = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
JPEG = _JPEG.tobytes()


def message(seq: int) -> bytes:
    return encode_frame(FrameHeader(seq=seq, timestamp=1.0 + seq / 30), JPEG)


def device_stats(client: TestClient, device: str) -> dict:
    return next(d for d in client.get("/api/live").json()["devices"] if d["device"] == device)


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_results_are_routed_to_the_sending_phone(client: TestClient) -> None:
    with (
        client.websocket_connect("/ws/stream/route-a") as a,
        client.websocket_connect("/ws/stream/route-b") as b,
    ):
        for seq in range(3):
            a.send_bytes(message(seq))
            b.send_bytes(message(100 + seq))
            result_a, result_b = json.loads(a.receive_text()), json.loads(b.receive_text())
            assert (result_a["device"], result_a["seq"]) == ("route-a", seq)
            assert (result_b["device"], result_b["seq"]) == ("route-b", 100 + seq)
            assert result_a["latency_ms"] >= result_a["queue_ms"] >= 0
        live = client.get("/api/live").json()
        assert {d["device"] for d in live["devices"]} == {"route-a", "route-b"}
        assert live["worker_device"] == "fake"
    # Both disconnected: the handlers' cleanup runs on the server loop just after the sockets close.
    wait_until(lambda: client.get("/api/live").json()["devices"] == [])


def test_latest_frame_wins_while_the_worker_is_busy(client: TestClient, backend: FakeBackend) -> None:
    backend.gate = threading.Event()
    with client.websocket_connect("/ws/stream/busy-a") as ws:
        ws.send_bytes(message(0))
        assert backend.entered.wait(5)  # frame 0 is in the worker, which is now blocked
        for seq in range(1, 5):
            ws.send_bytes(message(seq))
        wait_until(lambda: device_stats(client, "busy-a")["received"] == 5)
        stats = device_stats(client, "busy-a")
        assert stats["dropped"] == 2 and stats["queue_depth"] == 2  # queue size 2: frames 1 and 2 were replaced
        backend.gate.set()
        results = [json.loads(ws.receive_text()) for _ in range(3)]
    assert [r["seq"] for r in results] == [0, 3, 4]
    assert results[-1]["dropped"] == 2
    metrics = client.get("/metrics").text
    assert 'basir_frames_dropped_total{device="busy-a"} 2.0' in metrics
    assert 'basir_frames_received_total{device="busy-a"} 5.0' in metrics


def test_frames_from_two_phones_share_a_batch(client: TestClient, backend: FakeBackend) -> None:
    backend.gate = threading.Event()
    with (
        client.websocket_connect("/ws/stream/share-a") as a,
        client.websocket_connect("/ws/stream/share-b") as b,
    ):
        a.send_bytes(message(0))
        assert backend.entered.wait(5)
        a.send_bytes(message(1))
        b.send_bytes(message(1))
        wait_until(lambda: device_stats(client, "share-b")["received"] == 1)
        backend.gate.set()
        a.receive_text()
        assert json.loads(a.receive_text())["batch_size"] == 2
        assert json.loads(b.receive_text())["batch_size"] == 2
    assert backend.batches == [1, 2]


def test_bad_messages_get_an_error_and_the_stream_continues(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream/bad-a") as ws:
        ws.send_text("hello")
        assert "binary" in json.loads(ws.receive_text())["error"]
        ws.send_bytes(b"\x00\x00")
        assert "error" in json.loads(ws.receive_text())
        ws.send_bytes(encode_frame(FrameHeader(0, 1.0), b"PNG not JPEG"))
        assert "JPEG" in json.loads(ws.receive_text())["error"]
        ws.send_bytes(message(1))
        assert json.loads(ws.receive_text())["seq"] == 1


def test_duplicate_and_invalid_device_ids_are_refused(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream/dup-a"):
        with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect("/ws/stream/dup-a"):
            pass
        assert refused.value.code == 1008
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/stream/bad id!"):
        pass


def test_connection_gauge_follows_connections(client: TestClient) -> None:
    def gauge() -> float:
        line = next(x for x in client.get("/metrics").text.splitlines() if x.startswith("basir_ws_connections "))
        return float(line.split()[1])

    before = gauge()
    with client.websocket_connect("/ws/stream/gauge-a"):
        assert gauge() == before + 1
    wait_until(lambda: gauge() == before)
