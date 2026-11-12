"""Phone ingest: one WebSocket per phone at `/ws/stream/{device_id}` (wire format in `protocol`).

Each received frame goes onto the phone's latest-frame-wins queue in the `FrameBus`, becomes
the phone's newest frame for Live views and is offered to its live map; a sender task streams
that phone's results back on the same socket.

The route is served twice: by the Console (loopback, no token needed) and by the phone-only
listener on the LAN (`create_ingest_app`), where every phone needs its token. The token is
sent as the first WebSocket message, `{"type": "auth", "token": "bsr_..."}`, or as an
`Authorization: Bearer ...` header; never in the URL, where uvicorn would log it (a `?token=`
query is refused). On the Console, `OriginGuard` stops browsers on other sites from using
the tokenless route: browsers always send `Origin` on WebSockets, and only the Console's own
pages match it; clients with no `Origin` are local programs such as the phone simulator.
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
from starlette.websockets import WebSocketState

from basir import __version__
from basir.api.devices import DeviceRegistry, DeviceSession
from basir.api.framebus import FrameBus, QueuedFrame
from basir.api.mapping import MapManager
from basir.api.metrics import FRAMES_DROPPED, FRAMES_RECEIVED, QUEUE_DEPTH, WS_CONNECTIONS
from basir.api.middleware import add_standard_middleware, get_request_id, ip_or_local_host
from basir.api.pairing import PairingRequests
from basir.api.protocol import ProtocolError, decode_frame
from basir.api.tokens import TokenStore
from basir.api.viewer import LatestFrames

logger = logging.getLogger("basir.ingest")

DEVICE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
POLICY_VIOLATION = 1008  # WebSocket close codes
TRY_AGAIN_LATER = 1013
AUTH_TIMEOUT_S = 10.0  # a LAN client must send its auth message this soon after connecting
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
    # Any IP literal as Host (phones connect by LAN address), no domain names (DNS rebinding);
    # cross-site POSTs and WebSockets are refused.
    add_standard_middleware(app, allowed_host=ip_or_local_host)
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
        logger.info(
            "Phone %s (%s) asks to pair, code %s (request %s)", ask.device, address, pending.code, get_request_id()
        )
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


def _bearer(websocket: WebSocket) -> str | None:
    auth = websocket.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


async def _auth_message(websocket: WebSocket) -> str | None:
    """The token from the first message, `{"type": "auth", "token": "..."}`, or None (wrong shape, timeout)."""
    try:
        message = await asyncio.wait_for(websocket.receive(), AUTH_TIMEOUT_S)
    except TimeoutError:
        return None
    try:
        body = json.loads(message.get("text") or "")
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("type") != "auth" or not isinstance(body.get("token"), str):
        return None
    return body["token"]


@router.websocket("/ws/stream/{device_id}")
async def stream(websocket: WebSocket, device_id: str) -> None:
    """Receive one phone's frames and send back its results until it disconnects."""
    state = _state(websocket)
    if state is None:
        await websocket.close(TRY_AGAIN_LATER, "server is starting")
        return
    if not DEVICE_ID.match(device_id):
        await websocket.close(POLICY_VIOLATION, "invalid device id")
        return
    if getattr(websocket.app.state, "require_token", False) and not await _authenticate(websocket, state, device_id):
        return
    session = state.devices.connect(device_id)
    if session is None:
        await websocket.close(POLICY_VIOLATION, "device id already connected")
        return
    if websocket.client_state is WebSocketState.CONNECTING:
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


async def _authenticate(websocket: WebSocket, state: IngestState, device_id: str) -> bool:
    """Check the device token (Bearer header before accepting, else the first message); close with 1008 if wrong."""
    if "token" in websocket.query_params:
        logger.warning("Refused phone %s: token in the URL (request %s)", device_id, get_request_id())
        await websocket.close(POLICY_VIOLATION, "send the token as the first message, not in the URL")
        return False
    bearer = _bearer(websocket)
    if bearer is None:
        await websocket.accept()
    token = bearer if bearer is not None else await _auth_message(websocket)
    if state.tokens.verify(device_id, token):
        return True
    logger.warning("Refused phone %s: missing, wrong or expired token (request %s)", device_id, get_request_id())
    with suppress(RuntimeError):  # the client may already have gone
        await websocket.close(POLICY_VIOLATION, "missing, wrong or expired device token")
    return False


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
