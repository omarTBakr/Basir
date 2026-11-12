"""Tests for basir/api/airflow_view.py against a fake Airflow REST API (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json

import httpx

from basir.api.airflow_view import AirflowSnapshot, AirflowView, DagRunSummary, TaskState, node_states
from basir.api.dag import layout
from basir.pipelines.airflow_client import Credentials
from basir.pipelines.phase0 import phase0_dag
from tests.conftest import UNREACHABLE

DESK, ROOM = "fr1/desk", "fr1/room"
CREDS = Credentials("admin", "secret")


def _ti(task_id: str, state: str | None, map_index: int = -1, label: str | None = None) -> dict[str, object]:
    return {"task_id": task_id, "map_index": map_index, "rendered_map_index": label, "state": state, "try_number": 1}


class FakeAirflow:
    """/auth/token and the two list endpoints; tokens expire after `token_uses` requests."""

    def __init__(self, token_uses: int = 100) -> None:
        self.logins = 0
        self.token_uses = token_uses
        self.uses = 0
        self.runs = [
            {
                "dag_id": "phase0",
                "dag_run_id": "manual__old",
                "run_type": "manual",
                "state": "success",
                "start_date": "2026-11-10T10:00:00Z",
                "end_date": "2026-11-10T10:20:00Z",
            },
            {
                "dag_id": "phase0",
                "dag_run_id": "phase0-20261112-1",
                "run_type": "manual",
                "state": "running",
                "start_date": "2026-11-12T09:00:00Z",
                "end_date": None,
            },
        ]
        self.tasks = {
            "phase0-20261112-1": [
                _ti("open_pipeline_run", "success"),
                _ti("check_env", "success"),
                _ti("prepare_dataset", "success", 0, DESK),
                _ti("prepare_dataset", "up_for_retry", 1, None),  # no label yet: placed by map index
                _ti("replay", "running"),
                _ti("pointcloud", "queued", 0, DESK),
                _ti("gate", None),  # not scheduled yet
            ],
            "manual__old": [_ti("gate", "success")],
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/token":
            assert json.loads(request.content) == {"username": "admin", "password": "secret"}
            self.logins += 1
            self.uses = 0
            return httpx.Response(201, json={"access_token": f"token-{self.logins}"})
        if request.headers.get("authorization") != f"Bearer token-{self.logins}" or self.uses >= self.token_uses:
            return httpx.Response(401, json={"detail": "Token expired"})
        self.uses += 1
        if request.url.path == "/api/v2/dags/~/dagRuns":
            assert request.url.params["order_by"] == "-run_after"
            return httpx.Response(200, json={"dag_runs": self.runs, "total_entries": len(self.runs)})
        run_id = request.url.path.split("/")[-2]
        return httpx.Response(200, json={"task_instances": self.tasks[run_id], "total_entries": 1})


def _view(fake: FakeAirflow) -> AirflowView:
    return AirflowView("http://airflow:8080/", CREDS, transport=httpx.MockTransport(fake.handler))


def test_snapshot_lists_runs_running_first_with_task_states() -> None:
    snapshot = asyncio.run(_view(FakeAirflow()).snapshot())
    assert snapshot.reachable and [r.run_id for r in snapshot.runs] == ["phase0-20261112-1", "manual__old"]
    run = snapshot.runs[0]
    assert run.url == "http://airflow:8080/dags/phase0/runs/phase0-20261112-1"
    assert run.started is not None and run.started.hour == 9 and run.ended is None
    assert [t.state for t in run.tasks][-1] == "none"
    prepare = next(t for t in run.tasks if t.task_id == "prepare_dataset" and t.map_index == 0)
    assert prepare.label == DESK and prepare.url.endswith("/runs/phase0-20261112-1/tasks/prepare_dataset/mapped/0")


def test_expired_token_logs_in_again() -> None:
    fake = FakeAirflow(token_uses=2)
    view = _view(fake)
    assert asyncio.run(view.snapshot()).reachable
    assert asyncio.run(view.snapshot()).reachable
    assert fake.logins >= 2


def test_bad_credentials_and_unreachable_airflow_give_a_snapshot_not_an_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid credentials"})

    snapshot = asyncio.run(AirflowView("http://airflow:8080", CREDS, transport=httpx.MockTransport(refuse)).snapshot())
    assert not snapshot.reachable and "Invalid credentials" in (snapshot.error or "")

    down = asyncio.run(AirflowView(f"http://{UNREACHABLE}", CREDS, timeout=2.0).snapshot())
    assert not down.reachable and down.runs == [] and UNREACHABLE in (down.error or "")


def test_task_instances_map_onto_dag_json_nodes() -> None:
    nodes = [s.name for s in phase0_dag([DESK, ROOM]).steps]
    tasks = asyncio.run(_view(FakeAirflow()).snapshot()).runs[0].tasks
    assert node_states(tasks, nodes) == {
        "check_env": "success",
        f"prepare_dataset:{DESK}": "success",
        f"prepare_dataset:{ROOM}": "up_for_retry",
        f"replay:{DESK}": "running",
        f"pointcloud:{DESK}": "queued",
        "gate": "none",
    }


def test_colour_overrides_mlflow_statuses_with_airflow_states() -> None:
    dag = phase0_dag([DESK])
    laid_out = layout([s.name for s in dag.steps], dag.edges, {"check_env": "succeeded"})
    run = DagRunSummary(
        "phase0", "r1", "manual", "running", None, None, "u", [TaskState("gate", -1, None, "upstream_failed", 0, "u")]
    )
    snapshot = AirflowSnapshot(reachable=True, runs=[run])
    assert snapshot.by_run_id("r1") is run and snapshot.by_run_id(None) is None
    coloured = {n.name: n for n in snapshot.colour(laid_out, run).nodes}
    assert coloured["gate"].status == "upstream_failed" and coloured["gate"].css == "failed"
    assert coloured["check_env"].status == "succeeded"  # Airflow did not list it: MLflow's status stays
    assert snapshot.colour(laid_out, None) is laid_out
