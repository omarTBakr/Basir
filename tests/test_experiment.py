"""Tests for utilities/experiment.py (MLflow run tracking)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import utilities.experiment as experiment
from utilities.experiment import log_artifact, log_metrics, log_params, start_run


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
    with caplog.at_level(logging.WARNING, logger="basir.experiment"):
        with start_run("test-exp") as run_id:
            assert run_id is None
            log_params({"a": 1})
            log_metrics({"b": 2})
    assert sum("MLflow" in r.getMessage() for r in caplog.records) == 1


def test_exceptions_inside_the_run_propagate_and_close_the_run(mlflow_store: str) -> None:
    import mlflow

    with pytest.raises(RuntimeError):
        with start_run("test-exp"):
            raise RuntimeError("boom")
    assert mlflow.active_run() is None
