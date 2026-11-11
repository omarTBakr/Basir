"""Phone ingest: one WebSocket per phone at `/ws/stream/{device_id}` (wire format in `protocol`).

Each received frame goes onto the phone's latest-frame-wins queue in the `FrameBus`, becomes
the phone's newest frame for Live views and is offered to its live map; a sender task streams
that phone's results back on the same socket.

The route is served twice: by the Console (loopback, no token needed) and by the phone-only
listener on the LAN (`create_ingest_app`), where every phone needs its token
(`?token=...` or `Authorization: Bearer ...`).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import time

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from basir import __version__
from basir.api.devices import DeviceRegistry, DeviceSession
from basir.api.framebus import FrameBus, QueuedFrame
from basir.api.mapping import MapManager
from basir.api.metrics import FRAMES_DROPPED, FRAMES_RECEIVED, QUEUE_DEPTH, WS_CONNECTIONS
from basir.api.middleware import get_request_id
from basir.api.pairing import PairingRequests
from basir.api.protocol import ProtocolError, decode_frame
from basir.api.tokens import TokenStore
from basir.api.viewer import LatestFrames

logger = logging.getLogger("basir.ingest")

DEVICE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
POLICY_VIOLATION = 1008  # WebSocket close codes
TRY_AGAIN_LATER = 1013
PHONE_PAGE = Path(__file__).parent / "phone"
JPEG_MAGIC = b"\xff\xd8"

router = APIRouter()


@dataclass
class IngestState:
    """What the ingest endpoint shares with the batcher, the Live view and the live map."""

    bus: FrameBus
    devices: DeviceRegistry
    worker_device: str
    latest: LatestFrames
    maps: MapManager
    tokens: TokenStore
    pairing: PairingRequests


class PairAsk(BaseModel):
    """A phone asking to join: its device ID and a name to show on the Devices tab."""

    device: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    label: str = ""


def create_ingest_app(console: FastAPI) -> FastAPI:
    """The phone-only app for the LAN listener: the phone page (`/`), pairing, `/ws/stream/{id}` (token
    required) and `/healthz`."""
    app = FastAPI(title="Basir ingest", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.console = console
    app.state.require_token = True
    app.include_router(router)
    app.mount("/phone", StaticFiles(directory=PHONE_PAGE), name="phone")

    @app.get("/", include_in_schema=False)
    def phone_page(request: Request) -> FileResponse:
        """The page a phone opens (QR code on the Devices tab): pairs, then streams from the browser."""
        state = getattr(console.state, "ingest", None)
        if state is not None and request.client:
            state.pairing.seen(request.client.host)
        return FileResponse(PHONE_PAGE / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "basir"}

    @app.post("/pair")
    def pair(request: Request, ask: PairAsk) -> dict[str, str]:
        """Ask to join; the phone shows the returned code and polls `/pair/{id}` until approved."""
        state = _ready(request)
        address = request.client.host if request.client else "?"
        try:
            pending = state.pairing.open(ask.device, ask.label or ask.device, address)
        except ValueError as e:
            raise HTTPException(429, str(e)) from e
        logger.info("Phone %s (%s) asks to pair, code %s", ask.device, address, pending.code)
        return {"request_id": pending.id, "code": pending.code, "status": "pending"}

    @app.get("/pair/{request_id}")
    def pair_status(request: Request, request_id: str) -> dict[str, object]:
        """`pending`, `denied`, or once `approved` with the token (handed over a single time)."""
        answer = _ready(request).pairing.collect(request_id)
        if answer is None:
            raise HTTPException(404, "unknown or expired pairing request")
        return answer

    return app


def _ready(request: Request) -> IngestState:
    state = getattr(request.app.state.console.state, "ingest", None)
    if state is None:
        raise HTTPException(503, "server is starting")
    return state


def _state(websocket: WebSocket) -> IngestState | None:
    owner = getattr(websocket.app.state, "console", websocket.app)
    return getattr(owner.state, "ingest", None)


def _token(websocket: WebSocket) -> str | None:
    auth = websocket.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else websocket.query_params.get("token")


@router.websocket("/ws/stream/{device_id}")
async def stream(websocket: WebSocket, device_id: str) -> None:
    """Receive one phone's frames and send back its results until it disconnects."""
    state = _state(websocket)
    if state is None:
        await websocket.close(TRY_AGAIN_LATER, "server is starting")
        return
    if getattr(websocket.app.state, "require_token", False) and not state.tokens.verify(device_id, _token(websocket)):
        logger.warning("Refused phone %s: missing or wrong token", device_id)
        await websocket.close(POLICY_VIOLATION, "missing or wrong device token")
        return
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
        state.latest.forget(device_id)
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
            header, jpeg, depth = decode_frame(data)
            if not jpeg.startswith(JPEG_MAGIC):
                raise ProtocolError("payload is not a JPEG")
        except ProtocolError as e:
            await websocket.send_text(json.dumps({"error": str(e)}))
            continue
        frame = QueuedFrame(device, header, jpeg, time.perf_counter(), depth)
        state.latest.publish(frame)
        state.maps.submit(frame)
        dropped = state.bus.put(frame)
        session.record_received(dropped)
        FRAMES_RECEIVED.labels(device).inc()
        if dropped:
            FRAMES_DROPPED.labels(device).inc()
        QUEUE_DEPTH.labels(device).set(state.bus.depth(device))


async def _send_results(websocket: WebSocket, session: DeviceSession) -> None:
    while True:
        result = await session.results.get()
        await websocket.send_text(result.to_json())
