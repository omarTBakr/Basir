"""Phone ingest: one WebSocket per phone at `/ws/stream/{device_id}` (wire format in `protocol`).

Each received frame goes onto the phone's latest-frame-wins queue in the `FrameBus`;
a sender task streams that phone's results back on the same socket.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from basir.api.devices import DeviceRegistry, DeviceSession
from basir.api.framebus import FrameBus, QueuedFrame
from basir.api.metrics import FRAMES_DROPPED, FRAMES_RECEIVED, QUEUE_DEPTH, WS_CONNECTIONS
from basir.api.middleware import get_request_id
from basir.api.protocol import ProtocolError, decode_frame

logger = logging.getLogger("basir.ingest")

DEVICE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
POLICY_VIOLATION = 1008  # WebSocket close code
JPEG_MAGIC = b"\xff\xd8"

router = APIRouter()


@dataclass
class IngestState:
    """What the ingest endpoint shares with the batcher and the Live page."""

    bus: FrameBus
    devices: DeviceRegistry
    worker_device: str


@router.websocket("/ws/stream/{device_id}")
async def stream(websocket: WebSocket, device_id: str) -> None:
    """Receive one phone's frames and send back its results until it disconnects."""
    state: IngestState = websocket.app.state.ingest
    session = state.devices.connect(device_id) if DEVICE_ID.match(device_id) else None
    if session is None:
        await websocket.close(POLICY_VIOLATION, "invalid or already connected device id")
        return
    await websocket.accept()
    WS_CONNECTIONS.inc()
    logger.info("Phone %s connected (request %s)", device_id, get_request_id())
    sender = asyncio.create_task(_send_results(websocket, session))
    try:
        await _receive_frames(websocket, state, session)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)  # a send to a closed socket is expected here
        state.bus.remove(device_id)
        state.devices.disconnect(device_id)
        with suppress(KeyError):
            QUEUE_DEPTH.remove(device_id)
        WS_CONNECTIONS.dec()
        logger.info("Phone %s disconnected after %d frames", device_id, session.received)


async def _receive_frames(websocket: WebSocket, state: IngestState, session: DeviceSession) -> None:
    device = session.device
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        data = message.get("bytes")
        try:
            if data is None:
                raise ProtocolError("frames must be binary messages")
            header, jpeg = decode_frame(data)
            if not jpeg.startswith(JPEG_MAGIC):
                raise ProtocolError("payload is not a JPEG")
        except ProtocolError as e:
            await websocket.send_text(json.dumps({"error": str(e)}))
            continue
        dropped = state.bus.put(QueuedFrame(device, header, jpeg, time.perf_counter()))
        session.record_received(dropped)
        FRAMES_RECEIVED.labels(device).inc()
        if dropped:
            FRAMES_DROPPED.labels(device).inc()
        QUEUE_DEPTH.labels(device).set(state.bus.depth(device))


async def _send_results(websocket: WebSocket, session: DeviceSession) -> None:
    while True:
        result = await session.results.get()
        await websocket.send_text(result.to_json())
