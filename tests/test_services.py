"""Tests for basir/api/services.py (Grafana / Prometheus reachability) against the local file server."""

from __future__ import annotations

import asyncio
import json

from basir.api.services import probe, scrape_status
from tests.conftest import UNREACHABLE, FileServer


def targets(*entries: dict) -> bytes:
    return json.dumps({"status": "success", "data": {"activeTargets": list(entries)}}).encode()


def test_probe_up_and_down(file_server: FileServer) -> None:
    file_server.files["api/health"] = b'{"database": "ok"}'
    up = asyncio.run(probe("Grafana", file_server.base_url, "/api/health"))
    assert up.up and up.detail == "HTTP 200"
    missing = asyncio.run(probe("Grafana", file_server.base_url, "/nope"))
    assert not missing.up and missing.detail == "HTTP 404"
    down = asyncio.run(probe("Grafana", f"http://{UNREACHABLE}", "/api/health"))
    assert not down.up and "not reachable" in down.detail


def test_scrape_status_reads_the_basir_job(file_server: FileServer) -> None:
    key = "api/v1/targets?state=active"
    up = {"labels": {"job": "basir"}, "health": "up", "scrapeUrl": "http://127.0.0.1:8000/metrics", "lastError": ""}
    file_server.files[key] = targets({"labels": {"job": "node"}, "health": "up"}, up)
    status = asyncio.run(scrape_status(file_server.base_url))
    assert status.up and "127.0.0.1:8000/metrics: up" in status.detail

    file_server.files[key] = targets({**up, "health": "down", "lastError": "connection refused"})
    status = asyncio.run(scrape_status(file_server.base_url))
    assert not status.up and "connection refused" in status.detail

    file_server.files[key] = targets({"labels": {"job": "node"}, "health": "up"})
    assert "no 'basir' job" in asyncio.run(scrape_status(file_server.base_url)).detail


def test_scrape_status_when_prometheus_is_down() -> None:
    status = asyncio.run(scrape_status(f"http://{UNREACHABLE}"))
    assert not status.up and "not reachable" in status.detail
