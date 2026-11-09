"""Experiment tracking with MLflow: one run per script execution or evaluation.

Scripts use this module instead of importing mlflow directly, so the tracking
location, standard tags and failure handling live in one place. Logging problems
(for example a locked database) produce one warning and never stop the run.

    with start_run("phase0-pointcloud", run_name="fr1_room", tags={"dataset": "fr1/room"}):
        log_params({"every": 15})
        log_metrics({"points": 123456})
        log_artifact("outputs/fr1_room.ply")
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
import subprocess
from typing import Any

from utilities.config import get_settings

logger = logging.getLogger("basir.experiment")

_active = False  # True only while a run started by start_run() is open and healthy
_warned = False


@contextmanager
def start_run(
    experiment: str, run_name: str | None = None, tags: dict[str, str] | None = None
) -> Iterator[str | None]:
    """Open an MLflow run in `experiment`; yields the run ID (None if MLflow is unavailable)."""
    global _active
    run = None
    try:
        import mlflow

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            settings.mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(experiment, artifact_location=settings.mlflow_artifacts_dir.as_uri())
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name, tags={**_standard_tags(), **(tags or {})})
        _active = True
    except Exception as e:  # noqa: BLE001 - tracking must never break the caller
        _warn(f"MLflow tracking disabled for this run: {e}")

    try:
        yield run.info.run_id if run else None
    finally:
        if run is not None:
            _active = False
            _safe(lambda: __import__("mlflow").end_run())


def log_params(params: dict[str, Any]) -> None:
    if _active:
        _safe(lambda: __import__("mlflow").log_params({k: str(v) for k, v in params.items()}))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if _active:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        _safe(lambda: __import__("mlflow").log_metrics(clean, step=step))


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if _active and Path(path).exists():
        _safe(lambda: __import__("mlflow").log_artifact(str(path), artifact_path))


def _standard_tags() -> dict[str, str]:
    """git commit and dirty flag, Basir version, GPU name: enough to reproduce a run."""
    from basir import __version__

    tags = {"basir_version": __version__, "device": get_settings().effective_device}
    root = get_settings().project_root
    try:
        tags["git_commit"] = _git(root, "rev-parse", "--short", "HEAD") or "none"
        tags["git_dirty"] = str(bool(_git(root, "status", "--porcelain")))
    except (OSError, subprocess.SubprocessError):
        tags["git_commit"] = "unknown"
    try:
        import torch

        if torch.cuda.is_available():
            tags["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return tags


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 else ""


def _safe(action: Any) -> None:
    try:
        action()
    except Exception as e:  # noqa: BLE001
        _warn(f"MLflow logging failed: {e}")


def _warn(message: str) -> None:
    global _warned
    if not _warned:
        logger.warning(message)
        _warned = True
