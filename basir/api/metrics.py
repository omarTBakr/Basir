"""Prometheus metrics: the definitions, the HTTP timing middleware and the `/metrics` route.

The names are a contract with the Grafana "Basir server" dashboard; do not rename them.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5)

HTTP_REQUESTS = Counter("basir_http_requests", "HTTP requests served", ["method", "path", "status"])
HTTP_SECONDS = Histogram("basir_http_request_seconds", "HTTP request duration", buckets=_LATENCY_BUCKETS)
WS_CONNECTIONS = Gauge("basir_ws_connections", "Phones connected over WebSocket")
FRAMES_RECEIVED = Counter("basir_frames_received", "Frames received", ["device"])
FRAMES_DROPPED = Counter("basir_frames_dropped", "Frames replaced by a newer one before inference", ["device"])
QUEUE_DEPTH = Gauge("basir_queue_depth", "Frames waiting for inference", ["device"])
BATCH_SIZE = Histogram("basir_batch_size", "Frames per GPU batch", buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
INFERENCE_SECONDS = Histogram("basir_inference_seconds", "GPU worker time per batch", buckets=_LATENCY_BUCKETS)
FRAME_LATENCY = Histogram("basir_frame_latency_seconds", "Frame receipt to result sent", buckets=_LATENCY_BUCKETS)

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Serve every metric in the Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class HTTPMetricsMiddleware:
    """Count and time HTTP requests, labelled by route template (not raw path) to bound cardinality."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status = 500
        start = time.perf_counter()

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            route = scope.get("route")
            path = getattr(route, "path_format", None) or getattr(route, "path", None) or "unmatched"
            HTTP_REQUESTS.labels(scope["method"], path, str(status)).inc()
            HTTP_SECONDS.observe(time.perf_counter() - start)
