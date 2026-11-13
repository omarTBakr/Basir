"""Live view: the newest frame of each phone, streamed to a browser only while it watches.

Browser socket `/ws/view/{device_id}` (Console listener only). Server -> browser:

    text   {"type": "stats", "connected": true, "fps": 29.8, "shown_fps": 15.0, ...}
    binary b"C" + JPEG            colour, as the phone sent it
    binary b"D" + PNG             depth, colourised (turbo, near = red), when the phone sends depth

Browser -> server: text {"depth_max_m": 4.0} sets the far end of the depth colour scale.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from basir.api.framebus import QueuedFrame
from basir.api.metrics import VIEWERS
from basir.api.protocol import valid_device_id
from basir.api.rgbd import decode_depth, depth_png

COLOUR_TAG, DEPTH_TAG = b"C", b"D"
DEFAULT_DEPTH_MAX_M = 3.0
IDLE_STATS_S = 1.0  # stats are sent at least this often, even without new frames
POLICY_VIOLATION = 1008  # WebSocket close code

router = APIRouter()


class LatestFrames:
    """The newest received frame per phone; waiters wake on every new frame."""

    def __init__(self) -> None:
        self.frames: dict[str, QueuedFrame] = {}
        self._events: dict[str, asyncio.Event] = {}

    def publish(self, frame: QueuedFrame) -> None:
        self.frames[frame.device] = frame
        self._events.pop(frame.device, asyncio.Event()).set()

    def waiter(self, device: str) -> asyncio.Event:
        """An event set by the next frame from `device`."""
        return self._events.setdefault(device, asyncio.Event())

    def forget(self, device: str) -> None:
        self.frames.pop(device, None)
        self._events.pop(device, asyncio.Event()).set()


@router.websocket("/ws/view/{device_id}")
async def view(websocket: WebSocket, device_id: str) -> None:
    """Stream one phone's colour and depth to a browser, at most VIEW_MAX_FPS (invalid ids: closed, 1008)."""
    if not valid_device_id(device_id):
        await websocket.close(POLICY_VIOLATION, "invalid device id")
        return
    state = websocket.app.state.ingest
    settings = websocket.app.state.settings
    await websocket.accept()
    VIEWERS.labels("live").inc()
    options = {"depth_max_m": DEFAULT_DEPTH_MAX_M}
    reader = asyncio.create_task(_read_options(websocket, options))
    period = 1 / settings.view_max_fps
    shown: list[float] = []
    last_seq: int | None = None
    try:
        while not reader.done():
            frame = state.latest.frames.get(device_id)
            if frame is None or frame.header.seq == last_seq:
                with suppress(TimeoutError):
                    await asyncio.wait_for(state.latest.waiter(device_id).wait(), IDLE_STATS_S)
                frame = state.latest.frames.get(device_id)
            if frame is not None and frame.header.seq != last_seq:
                last_seq = frame.header.seq
                await websocket.send_bytes(COLOUR_TAG + frame.jpeg)
                if frame.depth is not None:
                    png = await asyncio.to_thread(_depth_png, frame, options["depth_max_m"])
                    await websocket.send_bytes(DEPTH_TAG + png)
                now = time.perf_counter()
                shown = [t for t in shown if now - t < 2.0] + [now]
            await websocket.send_text(json.dumps(_stats(state, device_id, frame, shown)))
            await asyncio.sleep(period)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        VIEWERS.labels("live").dec()


async def _read_options(websocket: WebSocket, options: dict[str, float]) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        with suppress(TypeError, ValueError, KeyError):
            options["depth_max_m"] = max(0.2, min(20.0, float(json.loads(message.get("text") or "")["depth_max_m"])))


def _depth_png(frame: QueuedFrame, max_m: float) -> bytes:
    assert frame.depth is not None
    return depth_png(decode_depth(frame.depth, frame.header), max_m)


def _stats(state: Any, device: str, frame: QueuedFrame | None, shown: list[float]) -> dict[str, Any]:
    session = state.devices.sessions.get(device)
    stats: dict[str, Any] = {"type": "stats", "device": device, "connected": session is not None}
    if session is not None:
        stats |= session.stats(state.bus.depth(device))
    window = shown[-1] - shown[0] if len(shown) > 1 else 0.0
    stats["shown_fps"] = round((len(shown) - 1) / window, 1) if window > 0 else 0.0
    if frame is not None:
        header = frame.header
        stats |= {
            "seq": header.seq,
            "pose": header.pose is not None,
            "colour": f"{int(header.intrinsics['width'])}x{int(header.intrinsics['height'])}"
            if header.intrinsics
            else "",
            "depth": f"{int(header.depth['width'])}x{int(header.depth['height'])}" if header.depth else "",
        }
    return stats
