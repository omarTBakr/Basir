"""Tests for scripts/run_pipeline.py and basir/pipelines/airflow_client.py against a fake Airflow REST API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from basir.pipelines.airflow_client import AirflowClient, AirflowError, Credentials, credentials, log_text, task_url
from basir.pipelines.runner import ERROR_MARKER
from scripts import run_pipeline
from utilities.config import Settings

CLOUDS = [
    {"dataset": "fr1/desk", "median_cm": 1.89, "within_5cm": 0.84, "frames_used": 40, "points": 123456},
    {"dataset": "fr1/room", "median_cm": 1.81, "within_5cm": 0.82, "frames_used": 90, "points": 234567},
]


class FakeAirflow:
    """Accepts one trigger, then reports `states` for the run, one per poll."""

    def __init__(self, states: list[str], gate: Any = CLOUDS, failed_log: str = "") -> None:
        self.states = states
        self.gate = gate
        self.failed_log = failed_log
        self.triggered: dict[str, Any] | None = None
        self.unpaused = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/token":
            return httpx.Response(201, json={"access_token": "t"})
        assert request.headers["authorization"] == "Bearer t"
        if request.method == "PATCH" and path == "/api/v2/dags/phase0":
            self.unpaused = json.loads(request.content) == {"is_paused": False}
            return httpx.Response(200, json={"dag_id": "phase0", "is_paused": False})
        if request.method == "POST" and path == "/api/v2/dags/phase0/dagRuns":
            self.triggered = json.loads(request.content)
            return httpx.Response(200, json={"dag_run_id": self.triggered["dag_run_id"], "state": "queued"})
        run_id = self.triggered["dag_run_id"] if self.triggered else ""
        if path == f"/api/v2/dags/phase0/dagRuns/{run_id}":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return httpx.Response(200, json={"dag_run_id": run_id, "state": state})
        if path.endswith("/taskInstances/open_pipeline_run/xcomEntries/parent_run_id"):
            return httpx.Response(200, json={"key": "parent_run_id", "value": "abc123"})
        if path.endswith("/taskInstances/gate/xcomEntries/return_value"):
            if self.gate is None:
                return httpx.Response(404, json={"detail": "XCom entry not found"})
            return httpx.Response(200, json={"key": "return_value", "value": self.gate})
        if path.endswith(f"/dagRuns/{run_id}/taskInstances"):
            tasks = [
                {"task_id": "check_env", "state": "success", "map_index": -1, "end_date": "2026-11-12T09:01:00Z"},
                {
                    "task_id": "pointcloud",
                    "state": "failed",
                    "map_index": 1,
                    "rendered_map_index": "fr1/room",
                    "try_number": 2,
                    "end_date": "2026-11-12T09:05:00Z",
                },
                {"task_id": "gate", "state": "upstream_failed", "map_index": -1, "end_date": None},
            ]
            return httpx.Response(200, json={"task_instances": tasks, "total_entries": 3})
        if "/taskInstances/pointcloud/logs/2" in path:
            assert request.url.params["map_index"] == "1"
            return httpx.Response(200, json={"content": [{"event": "Traceback ..."}, {"event": self.failed_log}]})
        return httpx.Response(404, json={"detail": f"unexpected {request.method} {path}"})


@pytest.fixture(autouse=True)
def _mlflow_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLflow run links without asking a server for the run's experiment."""
    monkeypatch.setattr(run_pipeline, "mlflow_run_url", lambda run_id: f"http://127.0.0.1:5000/#/runs/{run_id}")


def _run(fake: FakeAirflow, wait: bool, capsys: pytest.CaptureFixture[str], **params: Any) -> tuple[int, str]:
    client = AirflowClient(
        "http://airflow:8080", Credentials("admin", "pw"), transport=httpx.MockTransport(fake.handler)
    )
    options = run_pipeline.Phase0Params(**params)
    code = run_pipeline.start(client, options, wait, sleep=lambda s: None)
    return code, capsys.readouterr().out


def test_trigger_and_wait_prints_the_runs_and_the_gate(
    clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeAirflow(["queued", "running", "running", "success"])
    code, out = _run(fake, True, capsys, datasets=["fr1/desk", "fr1/room"], every=10)

    assert code == 0 and fake.unpaused and fake.triggered is not None
    assert fake.triggered["conf"]["every"] == 10 and fake.triggered["conf"]["datasets"] == ["fr1/desk", "fr1/room"]
    assert "max_frames" not in fake.triggered["conf"] and fake.triggered["logical_date"] is None
    run_id = fake.triggered["dag_run_id"]
    lines = out.splitlines()
    assert lines[0] == f"Run       {run_id}" and run_id.startswith("phase0-")
    assert lines[1] == f"Airflow   http://airflow:8080/dags/phase0/runs/{run_id}"
    assert lines[2].startswith("MLflow    http://127.0.0.1:5000/") and "abc123" in lines[2]
    assert lines[3] == "PASS      fr1/desk: median 1.89 cm, 84% within 5 cm (40 frames, 123,456 points)"
    assert lines[4].startswith("PASS      fr1/room: median 1.81 cm, 82% within 5 cm")


def test_failed_run_names_the_task_and_the_step_error(
    clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = f"{ERROR_MARKER} GateFailed: Phase 0 gate failed: fr1/room: median 4.20 cm"
    code, out = _run(FakeAirflow(["running", "failed"], gate=None, failed_log=log), True, capsys)
    assert code == 1
    assert (
        out.splitlines()[-1]
        == "FAILED    pointcloud[fr1/room]: GateFailed: Phase 0 gate failed: fr1/room: median 4.20 cm"
    )


def test_without_wait_returns_after_the_links(
    clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(FakeAirflow(["running"]), False, capsys)
    assert code == 0 and len(out.splitlines()) == 3


def test_main_reports_an_unreachable_airflow(clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    clean_env.setenv("BASIR_AIRFLOW_URL", "http://127.0.0.1:1")
    clean_env.setenv("BASIR_AIRFLOW_PASSWORD", "pw")
    assert run_pipeline.main(["phase0"]) == 1
    assert "cannot reach Airflow at http://127.0.0.1:1" in capsys.readouterr().err


def test_credentials_fall_back_to_the_stack_password(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    stack_env = tmp_path / "stack.env"
    stack_env.write_text("POSTGRES_PASSWORD=x\nAIRFLOW_ADMIN_PASSWORD=from-stack\n")
    settings = Settings(_env_file=tmp_path / "none.env")
    assert credentials(settings, stack_env) == Credentials("admin", "from-stack")
    assert credentials(Settings(_env_file=tmp_path / "none.env", airflow_password="own"), stack_env).password == "own"


def test_client_errors_and_helpers() -> None:
    def deny(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Forbidden"})

    client = AirflowClient("http://a:8080/", Credentials("u", "p"), transport=httpx.MockTransport(deny))
    with pytest.raises(AirflowError, match="login as 'u' failed: HTTP 403: Forbidden"):
        client.dag_run("phase0", "r")
    assert task_url("http://a:8080/", "d", "r 1", "t", 2) == "http://a:8080/dags/d/runs/r%201/tasks/t/mapped/2"
    assert log_text({"content": [{"event": "a"}, ["host", "b"], "c"]}) == "a\nb\nc"
    assert log_text("plain") == "plain"
