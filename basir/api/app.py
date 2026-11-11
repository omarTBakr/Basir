"""The Basir server: phone ingest, Prometheus `/metrics` and the web UI in one FastAPI app.

    uvicorn basir.api.app:create_app --factory    (or: python scripts/serve.py)

The lifespan starts the GPU worker process and the batcher, and stops both on shutdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from basir import __version__
from basir.api import ingest, metrics, ui
from basir.api.batcher import Batcher
from basir.api.devices import DeviceRegistry
from basir.api.framebus import InProcessFrameBus
from basir.api.ingest import IngestState
from basir.api.metrics import HTTPMetricsMiddleware
from basir.api.middleware import RequestIDMiddleware
from basir.api.mlflow_view import MLflowView
from basir.api.temporal_view import TemporalView
from basir.api.worker import InferenceBackend, ProcessBackend
from utilities.config import Settings, get_settings


def create_app(settings: Settings | None = None, backend: InferenceBackend | None = None) -> FastAPI:
    """Build the app; `backend` defaults to the spawned GPU worker (tests pass a fake)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = backend or ProcessBackend(settings.device)
        await asyncio.to_thread(engine.start)
        bus = InProcessFrameBus(settings.ingest_queue_size)
        devices = DeviceRegistry()
        app.state.ingest = IngestState(bus, devices, engine.device)
        batcher = Batcher(bus, engine, devices.deliver, settings.batch_max_size, settings.batch_max_wait_ms / 1000)
        task = asyncio.create_task(batcher.run())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await asyncio.to_thread(engine.stop)

    app = FastAPI(title="Basir", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.mlflow = MLflowView(settings.mlflow_uri, settings.mlflow_ui_url)
    app.state.temporal = TemporalView(settings.temporal_address, settings.temporal_ui_url)
    app.add_middleware(HTTPMetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)  # added last, so it runs first and wraps the metrics
    app.include_router(metrics.router)
    app.include_router(ingest.router)
    app.include_router(ui.router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
