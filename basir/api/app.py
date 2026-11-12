"""The Basir Console: phone ingest, live view, live map, Prometheus `/metrics` and the web UI in one app.

    python scripts/serve.py      # Console on 127.0.0.1:8000 + phone listener on the LAN (:8001)

The lifespan starts the GPU worker process and the batcher, and stops them (and the live
map fusion tasks) on shutdown. `basir.api.ingest.create_ingest_app` serves the phone
stream alone, sharing this app's state.

Security: the Console trusts only loopback. `TrustedHostMiddleware` refuses any other Host
(DNS rebinding) and `OriginGuard` refuses cross-origin WebSockets and POSTs, so a website open
in the laptop's browser can neither watch `/ws/view`, feed `/ws/stream` nor post to the
Devices or map APIs (docs/09-server-and-ui.md, "Security").
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from basir import __version__
from basir.api import devices_page, ingest, mapping, metrics, ui, viewer
from basir.api.batcher import Batcher
from basir.api.devices import DeviceRegistry
from basir.api.framebus import InProcessFrameBus
from basir.api.ingest import IngestState
from basir.api.mapping import MapManager
from basir.api.middleware import LOOPBACK_HOSTS, TEST_HOSTS, add_standard_middleware
from basir.api.mlflow_view import MLflowView
from basir.api.pairing import PairingRequests
from basir.api.temporal_view import TemporalView
from basir.api.tokens import TokenStore
from basir.api.viewer import LatestFrames
from basir.api.worker import InferenceBackend, ProcessBackend
from utilities.config import Settings, get_settings


def create_app(settings: Settings | None = None, backend: InferenceBackend | None = None) -> FastAPI:
    """Build the app; `backend` defaults to the spawned GPU worker (tests pass a fake)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = backend or ProcessBackend(settings.device, settings.worker_startup_timeout_s)
        await asyncio.to_thread(engine.start)
        bus = InProcessFrameBus(settings.ingest_queue_size)
        devices = DeviceRegistry()
        maps = MapManager(settings)
        tokens = TokenStore(settings.devices_file, settings.token_ttl_days)  # TOKEN_TTL_DAYS
        app.state.ingest = IngestState(bus, devices, engine.device, LatestFrames(), maps, tokens, PairingRequests())
        batcher = Batcher(bus, engine, devices.deliver, settings.batch_max_size, settings.batch_max_wait_ms / 1000)
        task = asyncio.create_task(batcher.run())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await maps.stop()
            await asyncio.to_thread(engine.stop)

    app = FastAPI(title="Basir", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.mlflow = MLflowView(settings.mlflow_uri, settings.mlflow_ui_url)
    app.state.temporal = TemporalView(settings.temporal_address, settings.temporal_ui_url)
    add_standard_middleware(app, trusted_hosts=[*LOOPBACK_HOSTS, *TEST_HOSTS])
    app.include_router(metrics.router)
    app.include_router(ingest.router)
    app.include_router(viewer.router)
    app.include_router(mapping.router)
    app.include_router(devices_page.router)
    app.include_router(ui.router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
