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
AIRFLOW_SECRETS = ("AIRFLOW_ADMIN_PASSWORD", "AIRFLOW_FERNET_KEY", "AIRFLOW_JWT_SECRET", "AIRFLOW_API_SECRET_KEY")
SERVICES = {
    "postgres",
    "mlflow",
    "airflow-init",
    "airflow-api-server",
    "airflow-scheduler",
    "airflow-dag-processor",
    "airflow-triggerer",
    "statsd-exporter",
    "node-exporter",
    "gpu-exporter",
    "postgres-exporter",
    "prometheus",
    "grafana",
}
# Compose folders: the five airflow-* services share Docker/airflow/compose.yml.
FOLDERS = SERVICES - {s for s in SERVICES if s.startswith("airflow-")} | {"airflow"}
ONE_SHOT = {"airflow-init"}  # runs `airflow db migrate` and exits; the others wait for it
DASHBOARDS = {
    "host": "basir-host",
    "gpu": "basir-gpu",
    "postgres": "basir-postgres",
    "airflow": "basir-airflow",
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
        env={k: v for k, v in os.environ.items() if k != "MLFLOW_ARTIFACTS_DIR"}  # the example's default
        | {"BASIR_DATA_DIR": str(PROJECT_ROOT)}  # exported by scripts/stack.sh
        | {key: "x" for key in AIRFLOW_SECRETS},  # generated into env/stack.env by scripts/stack.sh
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
    assert included == FOLDERS
    for name in FOLDERS:
        assert (DOCKER / name / "compose.yml").is_file(), f"Docker/{name}/compose.yml is missing"


def test_bind_mounts_stay_in_the_service_folder(compose_config: dict[str, Any]) -> None:
    """Config files mounted into a container come from that service's own Docker/<service>/ folder.

    Airflow also mounts Basir's code read-only and the data folders at their host paths.
    """
    airflow_mounts = {str(PROJECT_ROOT / d) for d in ("dags", "basir", "utilities", "datasets", "outputs")}
    for name, service in compose_config["services"].items():
        for volume in service.get("volumes", []):
            source = volume.get("source", "")
            if volume["type"] != "bind" or not source.startswith(str(PROJECT_ROOT)) or "mlartifacts" in source:
                continue
            if name.startswith("airflow-") and source in airflow_mounts:
                assert volume.get("read_only") or source.endswith(("datasets", "outputs")), f"{name} mounts {source} rw"
                continue
            assert source.startswith(str(DOCKER / name)), f"{name} mounts {source}"


def test_airflow_runs_basir_steps_with_host_paths(compose_config: dict[str, Any]) -> None:
    services = compose_config["services"]
    for name in SERVICES - ONE_SHOT:
        if not name.startswith("airflow-"):
            continue
        service = services[name]
        env = service["environment"]
        assert env["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor"
        assert env["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"].endswith("@postgres:5432/airflow")
        assert env["BASIR_MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
        assert env["BASIR_DATA_DIR"] == str(PROJECT_ROOT)
        targets = {v["target"]: v["source"] for v in service["volumes"]}
        assert targets[str(PROJECT_ROOT / "datasets")] == str(PROJECT_ROOT / "datasets")  # same path inside
        assert targets["/opt/basir/dags"] == str(PROJECT_ROOT / "dags")
        assert service["depends_on"]["airflow-init"]["condition"] == "service_completed_successfully"
    assert services["airflow-api-server"]["ports"][0]["published"] == "8080"
    assert services["airflow-init"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "temporal" not in json.dumps(compose_config).lower()


def test_postgres_creates_the_airflow_database() -> None:
    script = (DOCKER / "postgres" / "init-databases.sh").read_text()
    assert "for db in mlflow airflow; do" in script and "temporal" not in script


def test_mlflow_artifacts_default_to_the_project_folder(compose_config: dict[str, Any]) -> None:
    sources = {v["target"]: v["source"] for v in compose_config["services"]["mlflow"]["volumes"]}
    assert sources["/mlartifacts"] == str(PROJECT_ROOT / "mlartifacts")


def test_every_service_is_healthchecked_and_pinned(compose_config: dict[str, Any]) -> None:
    for name, service in compose_config["services"].items():
        assert name in ONE_SHOT or service.get("healthcheck", {}).get("test"), f"{name} has no healthcheck"
        image = service["image"]
        assert ":" in image and not image.endswith(":latest"), f"{name} image is not pinned: {image}"


def test_ports_bind_to_loopback_only(compose_config: dict[str, Any]) -> None:
    for name, service in compose_config["services"].items():
        for port in service.get("ports", []):
            assert port["host_ip"] == "127.0.0.1", f"{name} publishes {port} beyond loopback"


def test_data_lives_in_named_volumes(compose_config: dict[str, Any]) -> None:
    assert {"postgres-data", "prometheus-data", "grafana-data", "airflow-logs"} <= set(compose_config["volumes"])


def test_prometheus_scrapes_every_exporter_and_the_server() -> None:
    jobs = set(re.findall(r"job_name:\s*(\S+)", (DOCKER / "prometheus" / "prometheus.yml").read_text()))
    assert jobs == {"prometheus", "node", "gpu", "postgres", "airflow", "basir"}


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


def test_airflow_dashboard_uses_statsd_exporter_metrics() -> None:
    text = (DOCKER / "grafana" / "dashboards" / "airflow.json").read_text()
    for metric in ("airflow_scheduler_heartbeat", "airflow_dagrun_duration_seconds", "airflow_executor_queued_tasks"):
        assert metric in text
    assert not (DOCKER / "grafana" / "dashboards" / "temporal.json").exists()


def test_stack_script_parses() -> None:
    subprocess.run(["bash", "-n", str(PROJECT_ROOT / "scripts" / "stack.sh")], check=True)


@pytest.mark.network
def test_running_stack_targets_are_up() -> None:
    """Every Prometheus target except the Basir server (up only while it runs) reports UP."""
    with urllib.request.urlopen(f"{get_settings().prometheus_url}/api/v1/targets", timeout=10) as response:
        targets = json.load(response)["data"]["activeTargets"]
    health = {t["labels"]["job"]: t["health"] for t in targets}
    assert {job for job, state in health.items() if state != "up"} <= {"basir"}
    assert set(health) >= {"node", "gpu", "postgres", "airflow"}
