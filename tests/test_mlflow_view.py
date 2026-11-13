"""Tests for basir/api/mlflow_view.py against a temporary SQLite store seeded like the pipeline seeds it."""

from __future__ import annotations

import json
from pathlib import Path

from mlflow.tracking import MlflowClient
import pytest

from basir.api.mlflow_view import MLflowView

UI = "http://mlflow.test:5000"
START = 1_790_000_000_000  # ms


@pytest.fixture
def store(tmp_path: Path) -> str:
    """A pointcloud run that passed, one still running, and a pipeline with three steps."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    artifacts = (tmp_path / "mlartifacts").as_uri()

    cloud = client.create_experiment("phase0-pointcloud", artifact_location=artifacts)
    client.create_run(cloud, run_name="fr1_room", start_time=START)  # still running
    done = client.create_run(cloud, run_name="fr1_desk", start_time=START + 1000)
    for key, value in {"consistency_median_cm": 1.8, "consistency_within_5cm": 0.91, "passes": 1.0}.items():
        client.log_metric(done.info.run_id, key, value)
    client.set_terminated(done.info.run_id, "FINISHED", end_time=START + 5000)

    pipeline = client.create_experiment("phase0-pipeline", artifact_location=artifacts)
    tags = {"basir.pipeline": "phase0", "basir.workflow_id": "phase0-abc", "basir.status": "running"}
    parent = client.create_run(pipeline, run_name="phase0-abc", tags=tags, start_time=START)
    dag = {
        "nodes": [{"id": "prepare"}, {"id": "replay"}, {"id": "pointcloud"}],
        "edges": [{"source": "prepare", "target": "replay"}, {"source": "prepare", "target": "pointcloud"}],
    }
    dag_file = tmp_path / "dag.json"
    dag_file.write_text(json.dumps(dag))
    client.log_artifact(parent.info.run_id, str(dag_file))
    for step, depends_on, status in [("prepare", "", "succeeded"), ("replay", "prepare", "failed")]:
        child_tags = {
            "mlflow.parentRunId": parent.info.run_id,
            "basir.step": step,
            "basir.depends_on": depends_on,
            "basir.status": status,
        }
        child = client.create_run(pipeline, run_name=step, tags=child_tags, start_time=START + 2000)
        client.set_terminated(child.info.run_id, "FINISHED" if status == "succeeded" else "FAILED")
    return uri


def test_latest_run_per_experiment_with_verdict_and_key_metrics(store: str) -> None:
    snapshot = MLflowView(store, UI).experiments()
    assert snapshot.reachable
    by_experiment = {r.experiment: r for r in snapshot.latest}
    assert set(by_experiment) == {"phase0-pointcloud", "phase0-pipeline"}  # child steps never count as latest
    assert by_experiment["phase0-pipeline"].name == "phase0-abc"

    running = {r.name for r in snapshot.running}
    assert running == {"fr1_room", "phase0-abc"}

    desk = by_experiment["phase0-pointcloud"]
    assert desk.name == "fr1_desk" and desk.verdict == "pass" and desk.duration_s == 4.0
    assert desk.metrics == {"consistency_median_cm": 1.8, "consistency_within_5cm": 0.91}
    assert desk.url == f"{UI}/#/experiments/{desk_experiment(store)}/runs/{desk.run_id}"


def desk_experiment(uri: str) -> str:
    experiment = MlflowClient(tracking_uri=uri).get_experiment_by_name("phase0-pointcloud")
    assert experiment is not None
    return experiment.experiment_id


def test_pipeline_dag_coloured_by_step_status(store: str) -> None:
    snapshot = MLflowView(store, UI).pipelines()
    assert snapshot.reachable
    (pipeline,) = snapshot.pipelines
    assert pipeline.pipeline == "phase0" and pipeline.workflow_id == "phase0-abc" and pipeline.status == "running"
    statuses = {n.name: (n.status, n.css) for n in pipeline.dag.nodes}
    assert statuses == {
        "prepare": ("succeeded", "ok"),
        "replay": ("failed", "failed"),
        "pointcloud": ("pending", "pending"),  # in dag.json, not started yet
    }
    assert set(pipeline.dag.edges) == {("prepare", "replay"), ("prepare", "pointcloud")}


def test_no_links_without_an_mlflow_server(store: str) -> None:
    snapshot = MLflowView(store, None).experiments()
    assert all(r.url is None for r in snapshot.latest)


def test_unreachable_server_is_reported_not_raised() -> None:
    snapshot = MLflowView("http://127.0.0.1:1", None).experiments()
    assert not snapshot.reachable and snapshot.error
    assert not MLflowView("http://127.0.0.1:1", None).pipelines().reachable
