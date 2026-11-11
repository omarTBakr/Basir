"""Tests for basir/api/metrics.py: the exact metric names the Grafana dashboard expects."""

from __future__ import annotations

from fastapi.testclient import TestClient

EXPECTED = [
    "basir_http_requests_total",
    "basir_http_request_seconds_bucket",
    "basir_ws_connections",
    "basir_frames_received_total",
    "basir_frames_dropped_total",
    "basir_queue_depth",
    "basir_batch_size_bucket",
    "basir_inference_seconds_bucket",
    "basir_frame_latency_seconds_bucket",
]


def test_metrics_exposed_with_contract_names(client: TestClient) -> None:
    client.get("/live")
    from basir.api.metrics import FRAMES_DROPPED, FRAMES_RECEIVED, QUEUE_DEPTH

    for metric in (FRAMES_RECEIVED, FRAMES_DROPPED, QUEUE_DEPTH):  # labelled series appear once used
        metric.labels("names-test")
    response = client.get("/metrics")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain")
    names = {line.split("{")[0].split(" ")[0] for line in response.text.splitlines() if not line.startswith("#")}
    assert set(EXPECTED) <= names


def test_http_requests_labelled_by_route_template(client: TestClient) -> None:
    client.get("/live")
    client.get("/no/such-page")
    text = client.get("/metrics").text
    assert 'basir_http_requests_total{method="GET",path="/live",status="200"}' in text
    assert 'basir_http_requests_total{method="GET",path="unmatched",status="404"}' in text
