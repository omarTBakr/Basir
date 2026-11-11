"""Experiment tracking with MLflow: one run per script execution or evaluation.

Scripts use this module instead of importing mlflow directly, so the tracking
location, standard tags and failure handling live in one place. Logging problems
(for example a locked database) produce one warning and never stop the run.

    with start_run("phase0-pointcloud", run_name="fr1_room", tags={"dataset": "fr1/room"}):
        log_params({"every": 15})
        log_metrics({"points": 123456})
        log_artifact("outputs/fr1_room.ply")

Work spread over several steps (a pipeline) opens one parent run with `open_run`,
logs each step as a nested `start_run(..., parent_run_id=...)` and ends with `close_run`.
Runs are tracked per thread, so parallel steps in one process do not interfere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
import subprocess
import threading
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from utilities.config import get_settings

logger = logging.getLogger("basir.experiment")

_state = threading.local()  # .active is True while this thread's start_run() run is open and healthy
_warned = False


@contextmanager
def start_run(
    experiment: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    parent_run_id: str | None = None,
) -> Iterator[str | None]:
    """Open an MLflow run in `experiment`; yields the run ID (None if MLflow is unavailable).

    With `parent_run_id` the run is nested under that run. The run ends FAILED if the block raises.
    """
    run = None
    try:
        run = mlflow.start_run(
            experiment_id=_experiment_id(experiment),
            run_name=run_name,
            parent_run_id=parent_run_id,
            tags={**_standard_tags(), **(tags or {})},
        )
        _state.active = True
    except Exception as e:
        _warn(f"MLflow tracking disabled for this run: {e}")

    status = "FAILED"
    try:
        yield run.info.run_id if run else None
        status = "FINISHED"
    finally:
        if run is not None:
            _state.active = False
            _safe(mlflow.end_run, status)


def open_run(
    experiment: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    artifacts: Iterable[Path] = (),
) -> str | None:
    """Create a run that stays open after this call returns; finish it with `close_run`. Returns its ID or None."""
    try:
        client = MlflowClient(get_settings().mlflow_uri)
        run_id = client.create_run(
            _experiment_id(experiment), run_name=run_name, tags={**_standard_tags(), **(tags or {})}
        ).info.run_id
    except Exception as e:
        _warn(f"MLflow tracking disabled for this run: {e}")
        return None
    for key, value in (params or {}).items():
        _safe(client.log_param, run_id, key, str(value))
    for path in artifacts:
        _safe(client.log_artifact, run_id, str(path))
    return run_id


def close_run(run_id: str | None, failed: bool = False, tags: dict[str, str] | None = None) -> None:
    """Set `tags` on a run made by `open_run` and end it FINISHED (or FAILED)."""
    if run_id is None:
        return
    try:
        client = MlflowClient(get_settings().mlflow_uri)
    except Exception as e:
        _warn(f"MLflow logging failed: {e}")
        return
    for key, value in (tags or {}).items():
        _safe(client.set_tag, run_id, key, value)
    _safe(client.set_terminated, run_id, "FAILED" if failed else "FINISHED")


def run_url(run_id: str) -> str | None:
    """Web link to a run when tracking goes to an MLflow server; None for a local store or an unknown run."""
    uri = get_settings().mlflow_uri.rstrip("/")
    if not uri.startswith(("http://", "https://")):
        return None
    try:
        experiment_id = MlflowClient(uri).get_run(run_id).info.experiment_id
    except Exception:
        return None
    return f"{uri}/#/experiments/{experiment_id}/runs/{run_id}"


def set_tags(tags: dict[str, str]) -> None:
    if _is_active():
        _safe(mlflow.set_tags, tags)


def log_params(params: dict[str, Any]) -> None:
    if _is_active():
        _safe(mlflow.log_params, {k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if _is_active():
        _safe(mlflow.log_metrics, {k: float(v) for k, v in metrics.items() if v is not None}, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if _is_active() and Path(path).exists():
        _safe(mlflow.log_artifact, str(path), artifact_path)


def _is_active() -> bool:
    return getattr(_state, "active", False)


def _experiment_id(name: str) -> str:
    """Point MLflow at the configured store and return the experiment's ID, creating it if needed."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_uri)
    if (found := mlflow.get_experiment_by_name(name)) is not None:
        return found.experiment_id
    settings.mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(name, artifact_location=settings.mlflow_artifacts_dir.as_uri())


def _standard_tags() -> dict[str, str]:
    """git commit and dirty flag, Basir version, GPU name: enough to reproduce a run."""
    import torch

    from basir import __version__

    settings = get_settings()
    tags = {"basir_version": __version__, "device": settings.effective_device}
    try:
        tags["git_commit"] = _git(settings.project_root, "rev-parse", "--short", "HEAD") or "none"
        tags["git_dirty"] = str(bool(_git(settings.project_root, "status", "--porcelain")))
    except (OSError, subprocess.SubprocessError):
        tags["git_commit"] = "unknown"
    if torch.cuda.is_available():
        tags["gpu"] = torch.cuda.get_device_name(0)
    return tags


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 else ""


def _safe(action: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        action(*args, **kwargs)
    except Exception as e:
        _warn(f"MLflow logging failed: {e}")


def _warn(message: str) -> None:
    global _warned
    if not _warned:
        logger.warning(message)
        _warned = True
