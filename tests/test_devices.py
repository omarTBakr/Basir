"""Tests for basir/api/devices.py (connected phones, rolling stats) and the event-loop safety of its readers."""

from __future__ import annotations

import inspect
import time

from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
import pytest

from basir.api import mapping, ui, viewer
from basir.api.batcher import ProcessedFrame
from basir.api.devices import RESULTS_QUEUE, DeviceRegistry, DeviceSession
from basir.api.framebus import QueuedFrame
from basir.api.protocol import FrameHeader, encode_frame
from tests.test_ingest import JPEG


def processed(device: str, seq: int) -> ProcessedFrame:
    frame = QueuedFrame(device, FrameHeader(seq, 0.0), b"\xff\xd8", time.perf_counter())
    return ProcessedFrame(frame, 1, 0.001, 0.002, 0.003)


def test_the_server_records_its_own_receipt_time() -> None:
    """Phones stamp frames with performance.now(); 'last frame' must use the server's wall clock."""
    session = DeviceSession("p")
    assert session.stats(0)["last_received_at"] is None and session.stats(0)["last_frame_s"] is None
    before = time.time()
    session.record_received(dropped_older=False)
    stats = session.stats(0)
    assert before <= stats["last_received_at"] <= time.time()  # type: ignore[operator]
    assert stats["last_frame_s"] == 0.0
    session.record_received(dropped_older=True, received_at=time.time() - 42)
    assert session.stats(0)["last_frame_s"] == pytest.approx(42, abs=1)
    assert session.received == 2 and session.dropped == 1


def test_queued_frames_carry_a_wall_clock_receipt_time() -> None:
    before = time.time()
    frame = QueuedFrame("p", FrameHeader(1, 0.001), b"", time.perf_counter())  # phone clock: 1 ms after page load
    assert before <= frame.received_wall <= time.time()


def test_a_slow_phone_keeps_only_the_newest_results() -> None:
    registry = DeviceRegistry()
    session = registry.connect("slow")
    assert session is not None
    for seq in range(RESULTS_QUEUE + 10):
        registry.deliver(processed("slow", seq))
    assert session.results.qsize() == RESULTS_QUEUE
    assert session.results.get_nowait().seq == 10  # the oldest 10 were dropped
    registry.deliver(processed("gone", 0))  # a phone that left: discarded


def test_live_stats_reach_the_api(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream/stats-phone") as phone:
        phone.send_bytes(encode_frame(FrameHeader(seq=1, timestamp=0.5), JPEG))
        phone.receive_text()
        devices = client.get("/api/live").json()["devices"]
    assert devices[0]["device"] == "stats-phone" and devices[0]["last_received_at"] > 1e9


def test_handlers_that_read_loop_state_run_on_the_loop() -> None:
    """Sync `def` routes run in a threadpool, racing the loop that mutates sessions, deques and maps."""
    loop_state_routes = {
        ("ui", "/partials/live"),
        ("ui", "/api/live"),
        ("mapping", "/api/map/sessions"),
        ("mapping", "/api/map/{device_id}/reset"),
        ("mapping", "/api/map/{device_id}/save"),
    }
    found = set()
    for name, module in (("ui", ui), ("mapping", mapping), ("viewer", viewer)):
        for route in module.router.routes:
            if isinstance(route, APIRoute) and (name, route.path) in loop_state_routes:
                assert inspect.iscoroutinefunction(route.endpoint), route.path
                found.add((name, route.path))
            elif isinstance(route, APIWebSocketRoute):
                assert inspect.iscoroutinefunction(route.endpoint)
    assert found == loop_state_routes


@pytest.mark.parametrize("device", ["bad id", "<script>", "x" * 65])
def test_the_live_view_refuses_invalid_device_ids(client: TestClient, device: str) -> None:
    with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect(f"/ws/view/{device}") as ws:
        ws.receive_text()
    assert closed.value.code == 1008
    assert device not in client.app.state.ingest.latest._events  # type: ignore[attr-defined]
