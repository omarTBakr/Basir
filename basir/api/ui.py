"""The Console's pages: server-rendered, with live parts as HTMX fragments polled from `/partials/*`.

The sidebar comes from `basir.api.tabs`; the Devices tab lives in `devices_page`, the live
view and map sockets in `viewer` and `mapping`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from basir.api.dag import NODE_H, NODE_W, status_class
from basir.api.grafana import DASHBOARDS, PANELS, dashboard_url, panel_url
from basir.api.ingest import IngestState
from basir.api.mlflow_view import MLflowSnapshot
from basir.api.services import ServiceStatus, gpu_utilization, probe, scrape_status
from basir.api.tabs import grouped, tab

MLFLOW_TIMEOUT_S = 6.0

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
router = APIRouter(include_in_schema=False)


def _ago(value: datetime | None) -> str:
    """'42 s ago', '5 min ago', '3 h ago' or a date."""
    if value is None:
        return "-"
    seconds = (datetime.now(UTC) - value).total_seconds()
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.0f} h ago"
    return value.strftime("%Y-%m-%d")


def _num(value: float | None) -> str:
    """Compact number: 3 significant digits, thousands separated."""
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


templates.env.filters["ago"] = _ago
templates.env.filters["num"] = _num
templates.env.filters["status_class"] = status_class
templates.env.globals.update(
    node_w=NODE_W,
    node_h=NODE_H,
    tab_groups=grouped(),
    asset_version=str(int(time.time())),  # busts browser caches
)


def live_stats(request: Request) -> dict[str, Any]:
    """Per-phone ingest statistics plus the worker and batching settings."""
    state: IngestState = request.app.state.ingest
    settings = request.app.state.settings
    return {
        "worker_device": state.worker_device,
        "queue_size": settings.ingest_queue_size,
        "batch_max_size": settings.batch_max_size,
        "batch_max_wait_ms": settings.batch_max_wait_ms,
        "devices": [s.stats(state.bus.depth(d)) for d, s in sorted(state.devices.sessions.items())],
    }


async def _mlflow(call: Callable[[], MLflowSnapshot]) -> MLflowSnapshot:
    try:
        return await asyncio.wait_for(asyncio.to_thread(call), MLFLOW_TIMEOUT_S)
    except TimeoutError:
        return MLflowSnapshot(reachable=False, error=f"no answer within {MLFLOW_TIMEOUT_S:.0f} s")


def _page(request: Request, page: str, title: str, poll_s: int, **extra: Any) -> HTMLResponse:
    context = {"page": page, "title": title, "poll_s": poll_s, **extra}
    return templates.TemplateResponse(request, "page.html", context)


def _embed(request: Request, port_setting: str, open_url: str | None) -> dict[str, str | None]:
    """The frameable copy of a service UI (see `embed`) and the real URL for "open in new tab"."""
    port = getattr(request.app.state.settings, port_setting)
    return {"embed_url": f"http://127.0.0.1:{port}/", "open_url": open_url}


@router.get("/")
def home() -> RedirectResponse:
    return RedirectResponse("/live")


@router.get("/live")
def live_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "live.html", {"page": "live", "title": "Live"})


@router.get("/map")
def map_page(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    return templates.TemplateResponse(
        request, "map.html", {"page": "map", "title": "3D map", "voxel_m": settings.map_voxel_m}
    )


@router.get("/partials/status")
async def status_partial(request: Request) -> HTMLResponse:
    """The header's service dots."""
    settings = request.app.state.settings
    mlflow_url = settings.mlflow_ui_url
    checks = await asyncio.gather(
        probe("MLflow", mlflow_url, "/health") if mlflow_url else _local_mlflow(),
        probe("Temporal", settings.temporal_ui_url, "/"),
        probe("Grafana", settings.grafana_url, "/api/health"),
        probe("Prometheus", settings.prometheus_url, "/-/ready"),
    )
    gpu = await gpu_utilization(settings.prometheus_url)
    return templates.TemplateResponse(request, "partials/status.html", {"checks": checks, "gpu": gpu})


async def _local_mlflow() -> ServiceStatus:
    return ServiceStatus("MLflow", "sqlite", True, "local SQLite store")


@router.get("/partials/live")
def live_partial(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/live.html", live_stats(request))


@router.get("/api/live")
def live_api(request: Request) -> dict[str, Any]:
    return live_stats(request)


@router.get("/pipelines")
def pipelines_page(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    return _page(
        request, "pipelines", "Pipelines", 5, **_embed(request, "temporal_embed_port", settings.temporal_ui_url)
    )


@router.get("/partials/pipelines")
async def pipelines_partial(request: Request) -> HTMLResponse:
    mlflow, temporal = await asyncio.gather(
        _mlflow(request.app.state.mlflow.pipelines), request.app.state.temporal.snapshot()
    )
    return templates.TemplateResponse(request, "partials/pipelines.html", {"mlflow": mlflow, "temporal": temporal})


@router.get("/experiments")
def experiments_page(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    return _page(
        request, "experiments", "Experiments", 10, **_embed(request, "mlflow_embed_port", settings.mlflow_ui_url)
    )


@router.get("/partials/experiments")
async def experiments_partial(request: Request) -> HTMLResponse:
    mlflow = await _mlflow(request.app.state.mlflow.experiments)
    return templates.TemplateResponse(request, "partials/experiments.html", {"mlflow": mlflow})


@router.get("/dashboards")
async def dashboards_page(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    services = await asyncio.gather(
        probe("Grafana", settings.grafana_url, "/api/health"),
        probe("Prometheus", settings.prometheus_url, "/-/ready"),
        scrape_status(settings.prometheus_url),
    )
    # `{theme}` is filled in by the page script with the current light/dark theme.
    panels = [(p, panel_url(settings.grafana_url, p, "{theme}")) for p in PANELS]
    dashboards = [(d, dashboard_url(settings.grafana_url, d.uid, "{theme}")) for d in DASHBOARDS]
    return templates.TemplateResponse(
        request,
        "dashboards.html",
        {"page": "dashboards", "title": "Dashboards", "services": services, "panels": panels, "dashboards": dashboards},
    )


@router.get("/{key}")
def coming_later(request: Request, key: str) -> HTMLResponse:
    """A planned tab: what it will do and which phase builds it."""
    planned = tab(key)
    if planned is None or planned.enabled:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "coming.html", {"page": key, "title": planned.title, "tab": planned})
