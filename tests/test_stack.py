"""Tests for the local stack: docker-compose.yml, its configs and scripts/stack.sh."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any
import urllib.request

import pytest

from utilities.config import PROJECT_ROOT, get_settings

INFRA = PROJECT_ROOT / "infra"
SERVICES = {
    "postgres",
    "mlflow",
    "temporal",
    "temporal-ui",
    "node-exporter",
    "gpu-exporter",
    "postgres-exporter",
    "prometheus",
    "grafana",
}
DASHBOARDS = {
    "host": "basir-host",
    "gpu": "basir-gpu",
    "postgres": "basir-postgres",
    "temporal": "basir-temporal",
    "server": "basir-server",
}


@pytest.fixture(scope="module")
def compose_config() -> dict[str, Any]:
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(INFRA / "stack.env.example"), "config", "--format", "json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if "docker: 'compose' is not a docker command" in result.stderr:
        pytest.skip("docker compose plugin is not installed")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_compose_defines_every_service(compose_config: dict[str, Any]) -> None:
    assert set(compose_config["services"]) == SERVICES


def test_every_service_is_healthchecked_and_pinned(compose_config: dict[str, Any]) -> None:
    for name, service in compose_config["services"].items():
        assert service.get("healthcheck", {}).get("test"), f"{name} has no healthcheck"
        image = service["image"]
        assert ":" in image and not image.endswith(":latest"), f"{name} image is not pinned: {image}"


def test_ports_bind_to_loopback_only(compose_config: dict[str, Any]) -> None:
    for name, service in compose_config["services"].items():
        for port in service.get("ports", []):
            assert port["host_ip"] == "127.0.0.1", f"{name} publishes {port} beyond loopback"


def test_data_lives_in_named_volumes(compose_config: dict[str, Any]) -> None:
    assert {"postgres-data", "prometheus-data", "grafana-data"} <= set(compose_config["volumes"])


def test_prometheus_scrapes_every_exporter_and_the_server() -> None:
    jobs = set(re.findall(r"job_name:\s*(\S+)", (INFRA / "prometheus" / "prometheus.yml").read_text()))
    assert jobs == {"prometheus", "node", "gpu", "postgres", "temporal", "basir"}


@pytest.mark.parametrize(("name", "uid"), DASHBOARDS.items())
def test_dashboards_have_stable_uids_and_unique_panel_ids(name: str, uid: str) -> None:
    board = json.loads((INFRA / "grafana" / "dashboards" / f"{name}.json").read_text())
    ids = [panel["id"] for panel in board["panels"]]
    assert board["uid"] == uid
    assert len(ids) == len(set(ids))


def test_server_dashboard_uses_basir_metrics() -> None:
    text = (INFRA / "grafana" / "dashboards" / "server.json").read_text()
    for metric in ("basir_ws_connections", "basir_frames_received_total", "basir_frames_dropped_total"):
        assert metric in text


def test_stack_script_parses() -> None:
    subprocess.run(["bash", "-n", str(PROJECT_ROOT / "scripts" / "stack.sh")], check=True)


@pytest.mark.network
def test_running_stack_targets_are_up() -> None:
    """Every Prometheus target except the Basir server (up only while it runs) reports UP."""
    with urllib.request.urlopen(f"{get_settings().prometheus_url}/api/v1/targets", timeout=10) as response:
        targets = json.load(response)["data"]["activeTargets"]
    health = {t["labels"]["job"]: t["health"] for t in targets}
    assert {job for job, state in health.items() if state != "up"} <= {"basir"}
    assert set(health) >= {"node", "gpu", "postgres", "temporal"}
