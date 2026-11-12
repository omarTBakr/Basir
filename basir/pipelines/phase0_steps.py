"""Phase 0 pipeline steps, the small payloads they exchange, and `execute`, which runs one step by name.

Each step logs one MLflow child run under the pipeline's parent run, tagged with
`basir.step`, `basir.depends_on` and `basir.status`. Payloads carry paths and numbers
only (never frames or clouds), because Airflow passes them between tasks as JSON XComs.

Airflow runs every step through `basir.pipelines.runner` in Basir's own environment (see
dags/phase0.py); `run_locally` walks the same DAG in this process, for tests and debugging.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import platform
import tempfile
from typing import Any

from basir.ingest.datasets.prepare import dataset_dir, download_dataset
from basir.ingest.playback import Pacer, play
from basir.ingest.sources.tum import TUMDatasetSource
from basir.pipelines.dag import Dag
from basir.pipelines.phase0 import CLOSE_TASK, OPEN_TASK, PIPELINE, Phase0Params, phase0_dag
from basir.spatial.fusion import TARGET_MEDIAN_CM, TARGET_WITHIN_5CM, fuse, meets_targets, save_outputs
from utilities.config import get_settings
from utilities.environment import package_versions
from utilities.experiment import close_run, log_artifact, log_metrics, log_params, open_run, set_tags, start_run

EXPERIMENT = "phase0-pipeline"


class StepFailed(RuntimeError):
    """A step failure that a retry cannot fix (bad input, a failed gate); Airflow fails the task without retrying."""


class GateFailed(StepFailed):
    """A point cloud missed the Phase 0 targets."""


# Bad input or missing data: retrying will not help.
NON_RETRYABLE: tuple[type[BaseException], ...] = (StepFailed, FileNotFoundError, ValueError)


@dataclass
class StepContext:
    """Which DAG step runs and the parent MLflow run its child run goes under."""

    step: str
    depends_on: list[str]
    parent_run_id: str | None


@dataclass
class OpenRunInput:
    pipeline: str
    workflow_id: str
    dag: dict[str, Any]
    mermaid: str
    params: dict[str, Any]


@dataclass
class CloseRunInput:
    run_id: str | None
    succeeded: bool


@dataclass
class DatasetInput:
    ctx: StepContext
    dataset: str


@dataclass
class PreparedDataset:
    dataset: str
    path: str
    frames: int


@dataclass
class ReplayInput:
    ctx: StepContext
    prepared: PreparedDataset
    stride: int = 1
    speed: float = 0.0
    max_frames: int | None = None


@dataclass
class ReplayResult:
    dataset: str
    frames: int
    fps: float
    depth_valid_pct: float


@dataclass
class PointcloudInput:
    ctx: StepContext
    prepared: PreparedDataset
    every: int = 15
    pixel_stride: int = 4
    voxel: float = 0.02
    gap: int = 1
    max_frames: int | None = None


@dataclass
class PointcloudResult:
    dataset: str
    ply: str
    frames_used: int
    points: int
    median_cm: float | None
    within_5cm: float | None
    pairs: int


@dataclass
class GateInput:
    ctx: StepContext
    clouds: list[PointcloudResult] = field(default_factory=list)


def open_pipeline_run(data: OpenRunInput) -> str | None:
    """Open the pipeline's parent MLflow run with the DAG attached; returns its run ID."""
    tags = {"basir.pipeline": data.pipeline, "basir.workflow_id": data.workflow_id, "basir.status": "running"}
    with tempfile.TemporaryDirectory() as tmp:
        dag_json, dag_mmd = Path(tmp) / "dag.json", Path(tmp) / "dag.mmd"
        dag_json.write_text(json.dumps(data.dag, indent=2))
        dag_mmd.write_text(data.mermaid)
        return open_run(
            EXPERIMENT, run_name=data.workflow_id, tags=tags, params=data.params, artifacts=[dag_json, dag_mmd]
        )


def close_pipeline_run(data: CloseRunInput) -> None:
    """Record the pipeline's outcome on its parent run and end it."""
    close_run(data.run_id, failed=not data.succeeded, tags={"basir.status": _status(data.succeeded)})


def check_env(ctx: StepContext) -> dict[str, str]:
    """Check that the worker has every package and a writable data folder; returns what it found."""
    with _step_run(ctx):
        settings = get_settings()
        versions = package_versions()
        found = {"python": platform.python_version(), "device": settings.effective_device}
        found |= {name: v or "MISSING" for name, v in versions.items()}
        log_params(found | {"data_dir": str(settings.base_data_dir), "mlflow_uri": settings.mlflow_uri})
        if missing := [name for name, v in versions.items() if v is None]:
            raise StepFailed(f"missing packages: {', '.join(missing)}")
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        return found


def prepare_dataset(data: DatasetInput) -> PreparedDataset:
    """Find the dataset on disk, downloading and indexing it first if needed."""
    with _step_run(data.ctx, data.dataset):
        try:
            root = dataset_dir(data.dataset)
            downloaded = False
        except FileNotFoundError:
            root = download_dataset(data.dataset, quiet=True)
            downloaded = True
        frames = len(TUMDatasetSource(root))
        log_params({"dataset": data.dataset, "path": str(root), "downloaded": downloaded})
        log_metrics({"frames_available": frames})
        return PreparedDataset(dataset=data.dataset, path=str(root), frames=frames)


def replay(data: ReplayInput) -> ReplayResult:
    """Play the sequence headless and record frame rate and depth coverage."""
    with _step_run(data.ctx, data.prepared.dataset):
        source = TUMDatasetSource(Path(data.prepared.path), stride=data.stride, max_frames=data.max_frames)
        log_params(
            {
                "dataset": Path(data.prepared.path).name,
                "speed": data.speed,
                "stride": data.stride,
                "frames_available": len(source),
                "intrinsics_source": source.intrinsics_source,
            }
        )
        stats = play(source, Pacer(data.speed))
        log_metrics(stats.metrics())
        return ReplayResult(data.prepared.dataset, stats.frames, stats.fps, stats.depth_valid_pct)


def pointcloud(data: PointcloudInput) -> PointcloudResult:
    """Fuse the sequence into one cloud, score its depth consistency and save the cloud files."""
    with _step_run(data.ctx, data.prepared.dataset):
        root = Path(data.prepared.path)
        source = TUMDatasetSource(root, stride=data.every, max_frames=data.max_frames)
        log_params(
            {
                "dataset": root.name,
                "every": data.every,
                "pixel_stride": data.pixel_stride,
                "voxel": data.voxel,
                "gap": data.gap,
                "intrinsics_source": source.intrinsics_source,
            }
        )
        frames = list(source)
        cloud, result = fuse(frames, data.pixel_stride, data.voxel, data.gap)
        outputs = save_outputs(cloud, frames, get_settings().outputs_dir / f"{root.name}.ply")
        log_metrics(result.metrics())
        for path in outputs:
            log_artifact(path)
        return PointcloudResult(
            dataset=data.prepared.dataset,
            ply=str(outputs[0]),
            frames_used=result.frames_used,
            points=result.points_downsampled,
            median_cm=result.consistency_median_cm,
            within_5cm=result.consistency_within_5cm,
            pairs=result.consistency_pairs,
        )


def gate(data: GateInput) -> list[str]:
    """Pass only if every point cloud meets the Phase 0 targets; returns the datasets checked.

    Fails (without retry) naming each cloud that misses a target.
    """
    with _step_run(data.ctx):
        log_params({"target_median_cm": TARGET_MEDIAN_CM, "target_within_5cm": TARGET_WITHIN_5CM})
        failures = [_describe(c) for c in data.clouds if not meets_targets(c.median_cm, c.within_5cm)]
        log_metrics({"clouds": len(data.clouds), "passed": float(bool(data.clouds) and not failures)})
        if not data.clouds:
            raise GateFailed("gate received no point clouds")
        if failures:
            raise GateFailed(f"Phase 0 gate failed: {'; '.join(failures)}")
        return [c.dataset for c in data.clouds]


def _describe(cloud: PointcloudResult) -> str:
    if cloud.median_cm is None:
        return f"{cloud.dataset}: no overlapping frames to score"
    return (
        f"{cloud.dataset}: median {cloud.median_cm:.2f} cm (target <= {TARGET_MEDIAN_CM}), "
        f"{cloud.within_5cm or 0:.0%} within 5 cm (target >= {TARGET_WITHIN_5CM:.0%})"
    )


def _status(succeeded: bool) -> str:
    return "succeeded" if succeeded else "failed"


@contextmanager
def _step_run(ctx: StepContext, dataset: str | None = None) -> Iterator[None]:
    """Log the step as a child run of the pipeline run; its `basir.status` follows the outcome."""
    tags = {"basir.step": ctx.step, "basir.depends_on": ",".join(ctx.depends_on), "basir.status": "running"}
    if dataset:
        tags["basir.dataset"] = dataset
    with start_run(EXPERIMENT, run_name=ctx.step, tags=tags, parent_run_id=ctx.parent_run_id):
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            set_tags({"basir.status": _status(succeeded)})


# ------------------------------------------------------------------------------
# Running a step by name
# ------------------------------------------------------------------------------


def execute(task: str, payload: dict[str, Any]) -> Any:
    """Run the step behind Airflow task `task` and return its result as JSON-ready data.

    `payload` holds the run's `params`, the `parent_run_id`, and what the task received from
    upstream: `workflow_id` (open), `succeeded` (close), `dataset` (prepare), `prepared` (replay:
    the list of every prepared dataset; pointcloud: one), `clouds` (gate).
    """
    params = Phase0Params.from_dict(payload.get("params"))
    dag = phase0_dag(params.datasets)
    parent = payload.get("parent_run_id") or None  # "" = MLflow tracking was off when the run opened
    if task == OPEN_TASK:
        run_id = open_pipeline_run(
            OpenRunInput(PIPELINE, str(payload["workflow_id"]), dag.to_json(), dag.to_mermaid(), params.as_dict())
        )
        return {"parent_run_id": run_id, "datasets": params.datasets}
    if task == CLOSE_TASK:
        close_pipeline_run(CloseRunInput(parent, bool(payload["succeeded"])))
        return None
    if task == "check_env":
        return check_env(_context(dag, "check_env", parent))
    if task == "prepare_dataset":
        dataset = str(payload["dataset"])
        return asdict(prepare_dataset(DatasetInput(_context(dag, f"prepare_dataset:{dataset}", parent), dataset)))
    if task == "replay":
        step = next(s for s in dag.steps if s.activity == "replay")
        prepared = next(PreparedDataset(**p) for p in payload["prepared"] if p["dataset"] == step.dataset)
        ctx = _context(dag, step.name, parent)
        return asdict(replay(ReplayInput(ctx, prepared, params.replay_stride, params.replay_speed, params.max_frames)))
    if task == "pointcloud":
        prepared = PreparedDataset(**payload["prepared"])
        ctx = _context(dag, f"pointcloud:{prepared.dataset}", parent)
        options = (params.every, params.pixel_stride, params.voxel, params.gap, params.max_frames)
        return asdict(pointcloud(PointcloudInput(ctx, prepared, *options)))
    if task == "gate":
        clouds = [PointcloudResult(**c) for c in payload["clouds"]]
        gate(GateInput(_context(dag, "gate", parent), clouds))
        return [asdict(c) for c in clouds]  # what passed, for `run_pipeline.py --wait`
    raise ValueError(f"phase0 has no task '{task}'")


def run_locally(params: Phase0Params, workflow_id: str) -> dict[str, Any]:
    """Run the whole pipeline in this process, step by step in DAG order, as Airflow would; returns each result.

    The parent run is closed either way; a failing step's exception propagates.
    """
    base: dict[str, Any] = {"params": asdict(params)}
    opened = execute(OPEN_TASK, base | {"workflow_id": workflow_id})
    base["parent_run_id"] = opened["parent_run_id"]
    results: dict[str, Any] = {}
    succeeded = False
    try:
        results["check_env"] = execute("check_env", base)
        prepared = [execute("prepare_dataset", base | {"dataset": d}) for d in params.datasets]
        results["prepare_dataset"] = prepared
        results["replay"] = execute("replay", base | {"prepared": prepared})
        clouds = [execute("pointcloud", base | {"prepared": p}) for p in prepared]
        results["pointcloud"] = clouds
        results["gate"] = execute("gate", base | {"clouds": clouds})
        succeeded = True
    finally:
        execute(CLOSE_TASK, base | {"succeeded": succeeded})
    return results


def _context(dag: Dag, name: str, parent_run_id: str | None) -> StepContext:
    return StepContext(name, list(dag.step(name).depends_on), parent_run_id)
