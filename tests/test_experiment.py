"""Tests for utilities/experiment.py (MLflow run tracking)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import utilities.experiment as experiment
from utilities.experiment import (
    close_run,
    log_artifact,
    log_metrics,
    log_params,
    open_run,
    run_url,
    set_tags,
    start_run,
)


@pytest.fixture
def mlflow_store(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> str:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    clean_env.setattr(experiment, "_warned", False)
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_run_is_logged_and_readable(tmp_path: Path, mlflow_store: str) -> None:
    from mlflow.tracking import MlflowClient

    artifact = tmp_path / "cloud.txt"
    artifact.write_text("points")
    with start_run("test-exp", run_name="r1", tags={"dataset": "toy"}) as run_id:
        assert run_id
        log_params({"every": 15, "voxel": 0.02})
        log_metrics({"points": 1234, "skipped": None})
        log_metrics({"loss": 0.5}, step=3)
        log_artifact(artifact)
        log_artifact(tmp_path / "missing.ply")  # silently ignored

    client = MlflowClient(tracking_uri=mlflow_store)
    run = client.get_run(run_id)
    assert run.data.params == {"every": "15", "voxel": "0.02"}
    assert run.data.metrics["points"] == 1234 and "skipped" not in run.data.metrics
    assert run.data.tags["dataset"] == "toy"
    assert {"basir_version", "git_commit", "device"} <= set(run.data.tags)
    assert [a.path for a in client.list_artifacts(run_id)] == ["cloud.txt"]
    assert run.info.status == "FINISHED"
    assert (tmp_path / "mlartifacts").is_dir()


def test_logging_outside_a_run_is_a_noop(mlflow_store: str) -> None:
    log_params({"a": 1})
    log_metrics({"b": 2})


def test_broken_tracking_uri_warns_but_does_not_raise(
    tmp_path: Path, clean_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clean_env.setenv("BASIR_MLFLOW_TRACKING_URI", "unsupported-scheme://nowhere")
    clean_env.setattr(experiment, "_warned", False)
    with caplog.at_level(logging.WARNING, logger="basir.experiment"), start_run("test-exp") as run_id:
        assert run_id is None
        log_params({"a": 1})
        log_metrics({"b": 2})
    assert sum("MLflow" in r.getMessage() for r in caplog.records) == 1


def test_exceptions_inside_the_run_propagate_and_close_the_run(mlflow_store: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    with pytest.raises(RuntimeError), start_run("test-exp") as run_id:
        raise RuntimeError("boom")
    assert mlflow.active_run() is None
    assert MlflowClient(tracking_uri=mlflow_store).get_run(run_id).info.status == "FAILED"


def test_open_run_stays_open_until_closed_and_hosts_nested_runs(tmp_path: Path, mlflow_store: str) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_store)
    artifact = tmp_path / "dag.json"
    artifact.write_text("{}")
    parent = open_run("test-exp", run_name="pipeline", tags={"stage": "x"}, params={"n": 2}, artifacts=[artifact])
    assert parent and client.get_run(parent).info.status == "RUNNING"

    with start_run("test-exp", run_name="child", parent_run_id=parent) as child:
        set_tags({"step": "done"})
    child_run = client.get_run(child)
    assert child_run.data.tags["mlflow.parentRunId"] == parent and child_run.data.tags["step"] == "done"

    close_run(parent, failed=True, tags={"stage": "y"})
    run = client.get_run(parent)
    assert (run.info.status, run.data.tags["stage"], run.data.params) == ("FAILED", "y", {"n": "2"})
    assert [a.path for a in client.list_artifacts(parent)] == ["dag.json"]
    close_run(None)  # no run: nothing to do


def test_open_run_with_broken_tracking_returns_none(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_MLFLOW_TRACKING_URI", "unsupported-scheme://nowhere")
    clean_env.setattr(experiment, "_warned", False)
    assert open_run("test-exp") is None
    set_tags({"a": "b"})  # outside a run: a no-op


def test_run_url_only_for_tracking_servers(mlflow_store: str, clean_env: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from utilities.config import get_settings

    assert run_url("abc") is None  # local SQLite store
    clean_env.setenv("BASIR_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
    get_settings.cache_clear()
    run = SimpleNamespace(info=SimpleNamespace(experiment_id="7"))
    clean_env.setattr(experiment, "MlflowClient", lambda uri: SimpleNamespace(get_run=lambda run_id: run))
    assert run_url("abc") == "http://127.0.0.1:5000/#/experiments/7/runs/abc"


def test_artifact_location_is_local_for_sqlite(tmp_path: Path, mlflow_store: str) -> None:
    assert experiment._artifact_location() == (tmp_path / "mlartifacts").as_uri()


def test_artifact_location_is_left_to_a_tracking_server(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    clean_env.setenv("BASIR_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert experiment._artifact_location() is None
    assert not (tmp_path / "mlartifacts").exists()
