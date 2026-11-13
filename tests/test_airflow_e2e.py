"""End to end on a running stack: trigger Phase 0 on Airflow and check what lands in MLflow.

Needs the stack up (`scripts/stack.sh up`) with the TUM sequences downloaded; takes minutes.

    uv run pytest -m network tests/test_airflow_e2e.py
"""

from __future__ import annotations

import time

import pytest

from basir.pipelines.airflow_client import FINISHED, AirflowClient, credentials
from basir.pipelines.phase0 import DAG_ID, DEFAULT_DATASETS, OPEN_TASK, phase0_dag
from utilities.config import get_settings

pytestmark = pytest.mark.network
TIMEOUT_S = 60 * 60


def test_phase0_on_airflow_logs_a_parent_and_seven_children() -> None:
    from mlflow.tracking import MlflowClient

    settings = get_settings()
    run_id = f"phase0-e2e-{int(time.time())}"
    with AirflowClient(settings.airflow_url, credentials(settings)) as airflow:
        airflow.unpause(DAG_ID)
        airflow.trigger(DAG_ID, {"datasets": list(DEFAULT_DATASETS)}, run_id=run_id)
        deadline = time.monotonic() + TIMEOUT_S
        while (state := airflow.dag_run(DAG_ID, run_id)["state"]) not in FINISHED:
            assert time.monotonic() < deadline, "Phase 0 did not finish in time"
            time.sleep(10)
        parent_id = airflow.xcom(DAG_ID, run_id, OPEN_TASK, key="parent_run_id")
        task_states = {(t["task_id"], t["map_index"]): t["state"] for t in airflow.task_instances(DAG_ID, run_id)}
    assert state == "success", task_states
    assert parent_id

    mlflow = MlflowClient(settings.mlflow_uri)
    parent = mlflow.get_run(parent_id)
    assert parent.data.tags["basir.pipeline"] == "phase0"
    assert parent.data.tags["basir.workflow_id"] == run_id
    assert parent.data.tags["basir.status"] == "succeeded" and parent.info.status == "FINISHED"
    assert {a.path for a in mlflow.list_artifacts(parent_id)} == {"dag.json", "dag.mmd"}

    children = mlflow.search_runs([parent.info.experiment_id], f"tags.`mlflow.parentRunId` = '{parent_id}'")
    by_step = {c.data.tags["basir.step"]: c for c in children}
    dag = phase0_dag(list(DEFAULT_DATASETS))
    assert len(children) == 7 and set(by_step) == {s.name for s in dag.steps}
    for step in dag.steps:
        assert by_step[step.name].data.tags["basir.depends_on"] == ",".join(step.depends_on)
        assert by_step[step.name].data.tags["basir.status"] == "succeeded"
