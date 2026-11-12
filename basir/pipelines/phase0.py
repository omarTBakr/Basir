"""The Phase 0 pipeline's shape and parameters: prepare, replay and fuse the datasets, then gate on the targets.

    check_env -> prepare_dataset:<each dataset>        (one mapped Airflow task instance per dataset)
    prepare_dataset:<first> -> replay:<first>
    prepare_dataset:<each>  -> pointcloud:<each>
    replay + every pointcloud -> gate                 (passes only if every cloud meets the targets)

`dags/phase0.py` runs it on Airflow; `phase0_steps` holds the step logic. This module has no
third-party imports, so Airflow's interpreter can import it while parsing the DAG file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from basir.pipelines.dag import Dag, Step

PIPELINE = "phase0"
DAG_ID = "phase0"
DEFAULT_DATASETS = ("fr1/desk", "fr1/room")
# Bookkeeping tasks around the steps: they open and close the MLflow parent run and are not DAG nodes.
OPEN_TASK, CLOSE_TASK = "open_pipeline_run", "close_pipeline_run"


@dataclass
class Phase0Params:
    """What a run can override (Airflow DAG params / run conf)."""

    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    replay_stride: int = 1
    replay_speed: float = 0.0  # 0 = as fast as possible
    every: int = 15
    pixel_stride: int = 4
    voxel: float = 0.02
    gap: int = 1
    max_frames: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Phase0Params:
        """Params from a run's conf; unknown keys (such as Airflow's own) are ignored, None means the default."""
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in (raw or {}).items() if k in known and (v is not None or k == "max_frames")}
        params = cls(**values)
        if not isinstance(params.datasets, list) or not all(isinstance(d, str) and d for d in params.datasets):
            raise ValueError(f"datasets must be a list of dataset names, got {params.datasets!r}")
        return params

    def as_dict(self) -> dict[str, Any]:
        """The params logged on the MLflow parent run (unset values left out)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# JSON Schema of each param, turned into Airflow `Param`s by dags/phase0.py (the trigger form shows them).
PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    "datasets": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "description": "Preset keys (fr1/desk) or dataset folders; the first one is also replayed.",
    },
    "every": {"type": "integer", "minimum": 1, "description": "Point cloud: use every Nth frame."},
    "pixel_stride": {"type": "integer", "minimum": 1, "description": "Point cloud: use every Nth pixel."},
    "voxel": {"type": "number", "minimum": 0, "description": "Point cloud: voxel size in metres; 0 = off."},
    "gap": {"type": "integer", "minimum": 1, "description": "Point cloud: score frame i against frame i+gap."},
    "replay_stride": {"type": "integer", "minimum": 1, "description": "Replay: use every Nth frame."},
    "replay_speed": {"type": "number", "minimum": 0, "description": "Replay: 1 = real time, 0 = as fast as possible."},
    "max_frames": {"type": ["null", "integer"], "minimum": 1, "description": "Stop after this many frames."},
}


def param_defaults() -> dict[str, Any]:
    """Each param's default, keyed like PARAM_SCHEMA."""
    return asdict(Phase0Params())


def phase0_dag(datasets: list[str]) -> Dag:
    """The Phase 0 DAG; the first dataset is also replayed. Node ids are `<task id>[:<dataset>]`."""
    if not datasets:
        raise ValueError("phase0 needs at least one dataset")
    steps = [Step("check_env", "check_env")]
    steps += [Step(f"prepare_dataset:{d}", "prepare_dataset", d, ("check_env",)) for d in datasets]
    steps.append(Step(f"replay:{datasets[0]}", "replay", datasets[0], (f"prepare_dataset:{datasets[0]}",)))
    steps += [Step(f"pointcloud:{d}", "pointcloud", d, (f"prepare_dataset:{d}",)) for d in datasets]
    steps.append(Step("gate", "gate", None, tuple(s.name for s in steps if s.activity in ("replay", "pointcloud"))))
    return Dag(tuple(steps))
