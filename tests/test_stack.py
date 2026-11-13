"""Tests for the local stack: Docker/docker-compose.yml, the per-service folders and scripts/stack.sh."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any
import urllib.request

import pytest

from utilities.config import PROJECT_ROOT, get_settings

DOCKER = PROJECT_ROOT / "Docker"
STACK_ENV_EXAMPLE = PROJECT_ROOT / "env" / "stack.env.example"
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
    files = ["-f", str(DOCKER / "docker-compose.yml"), "--env-file", str(STACK_ENV_EXAMPLE)]
    result = subprocess.run(
        ["docker", "compose", *files, "config", "--format", "json"],
        cwd=PROJECT_ROOT,
        env={k: v for k, v in os.environ.items() if k != "MLFLOW_ARTIFACTS_DIR"},  # the example's default
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


def test_each_service_has_its_own_folder_and_is_included() -> None:
    """Docker/<service>/compose.yml per service; Docker/docker-compose.yml includes them all."""
    main = (DOCKER / "docker-compose.yml").read_text()
    included = set(re.findall(r"^\s*-\s*(\S+)/compose\.yml\s*$", main, flags=re.MULTILINE))
    assert included == SERVICES
    for name in SERVICES:
        assert (DOCKER / name / "compose.yml").is_file(), f"Docker/{name}/compose.yml is missing"


def test_bind_mounts_stay_in_the_service_folder(compose_config: dict[str, Any]) -> None:
    """Config files mounted into a container come from that service's own Docker/<service>/ folder."""
    for name, service in compose_config["services"].items():
        for volume in service.get("volumes", []):
            source = volume.get("source", "")
            if volume["type"] == "bind" and source.startswith(str(PROJECT_ROOT)) and "mlartifacts" not in source:
                assert source.startswith(str(DOCKER / name)), f"{name} mounts {source}"


def test_mlflow_artifacts_default_to_the_project_folder(compose_config: dict[str, Any]) -> None:
    sources = {v["target"]: v["source"] for v in compose_config["services"]["mlflow"]["volumes"]}
    assert sources["/mlartifacts"] == str(PROJECT_ROOT / "mlartifacts")


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
    jobs = set(re.findall(r"job_name:\s*(\S+)", (DOCKER / "prometheus" / "prometheus.yml").read_text()))
    assert jobs == {"prometheus", "node", "gpu", "postgres", "temporal", "basir"}


@pytest.mark.parametrize(("name", "uid"), DASHBOARDS.items())
def test_dashboards_have_stable_uids_and_unique_panel_ids(name: str, uid: str) -> None:
    board = json.loads((DOCKER / "grafana" / "dashboards" / f"{name}.json").read_text())
    ids = [panel["id"] for panel in board["panels"]]
    assert board["uid"] == uid
    assert len(ids) == len(set(ids))


def test_server_dashboard_uses_basir_metrics() -> None:
    text = (DOCKER / "grafana" / "dashboards" / "server.json").read_text()
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
