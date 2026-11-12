"""Tests for the Phase 0 pipeline without Airflow: its DAG and params, the step logic, the runner.

`run_locally` walks the DAG in-process exactly as dags/phase0.py wires it on Airflow, so the
real steps run on toy datasets to check the MLflow parent/child runs, their tags and the
logged DAG. tests/test_dags.py covers the Airflow DAG itself (inside the Airflow image).
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from basir.ingest.datasets.toy import create_toy_dataset
from basir.pipelines import airflow_task, phase0_steps, runner
from basir.pipelines.phase0 import (
    CLOSE_TASK,
    OPEN_TASK,
    PARAM_SCHEMA,
    Phase0Params,
    param_defaults,
    phase0_dag,
)
from basir.pipelines.phase0_steps import GateFailed, GateInput, PointcloudResult, StepContext, StepFailed

DESK, ROOM = "fr1/desk", "fr1/room"


def test_dag_shape() -> None:
    dag = phase0_dag([DESK, ROOM])
    assert [s.name for s in dag.steps] == [
        "check_env",
        f"prepare_dataset:{DESK}",
        f"prepare_dataset:{ROOM}",
        f"replay:{DESK}",
        f"pointcloud:{DESK}",
        f"pointcloud:{ROOM}",
        "gate",
    ]
    assert set(dag.edges) == {
        ("check_env", f"prepare_dataset:{DESK}"),
        ("check_env", f"prepare_dataset:{ROOM}"),
        (f"prepare_dataset:{DESK}", f"replay:{DESK}"),
        (f"prepare_dataset:{DESK}", f"pointcloud:{DESK}"),
        (f"prepare_dataset:{ROOM}", f"pointcloud:{ROOM}"),
        (f"replay:{DESK}", "gate"),
        (f"pointcloud:{DESK}", "gate"),
        (f"pointcloud:{ROOM}", "gate"),
    }
    assert dag.task_edges == {
        ("check_env", "prepare_dataset"),
        ("prepare_dataset", "replay"),
        ("prepare_dataset", "pointcloud"),
        ("replay", "gate"),
        ("pointcloud", "gate"),
    }
    with pytest.raises(ValueError):
        phase0_dag([])


def test_params_defaults_overrides_and_validation() -> None:
    assert set(PARAM_SCHEMA) == set(param_defaults())
    params = Phase0Params.from_dict({"datasets": [ROOM], "every": 5, "max_frames": None, "airflow_extra": 1})
    assert params.datasets == [ROOM] and params.every == 5 and params.pixel_stride == 4
    assert params.as_dict() == {k: v for k, v in asdict(params).items() if v is not None}
    assert Phase0Params.from_dict(None).datasets == [DESK, ROOM]
    for bad in ("fr1/desk", [1], [""]):
        with pytest.raises(ValueError):
            Phase0Params.from_dict({"datasets": bad})


def _cloud(dataset: str, median_cm: float) -> PointcloudResult:
    return PointcloudResult(dataset, "/out.ply", 10, 1000, median_cm, 0.85, 9)


def test_gate_names_each_cloud_that_misses_a_target(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    ctx = StepContext("gate", [], None)
    assert phase0_steps.gate(GateInput(ctx, [_cloud(DESK, 1.9), _cloud(ROOM, 1.8)])) == [DESK, ROOM]
    with pytest.raises(GateFailed) as failure:
        phase0_steps.gate(GateInput(ctx, [_cloud(DESK, 4.2), _cloud(ROOM, 1.8)]))
    assert f"{DESK}: median 4.20 cm" in str(failure.value) and ROOM not in str(failure.value)
    with pytest.raises(GateFailed):
        phase0_steps.gate(GateInput(ctx, []))
    assert isinstance(failure.value, StepFailed) and issubclass(GateFailed, phase0_steps.NON_RETRYABLE)


def test_execute_rejects_unknown_tasks_and_bad_params() -> None:
    with pytest.raises(ValueError):
        phase0_steps.execute("nope", {})
    with pytest.raises(ValueError):
        phase0_steps.execute("check_env", {"params": {"datasets": []}})


@pytest.mark.parametrize("passes", [True, False])
def test_real_steps_log_the_dag_to_mlflow(passes: bool, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    clean_env.setattr(phase0_steps, "meets_targets", lambda median, within: passes)
    datasets = [str(create_toy_dataset(tmp_path / name, num_frames=4)) for name in ("toy_a", "toy_b")]
    params = Phase0Params(datasets=datasets, every=1, pixel_stride=1, voxel=0)

    if passes:
        results = phase0_steps.run_locally(params, "manual__phase0-test")
        assert [c["frames_used"] for c in results["gate"]] == [4, 4]
        assert Path(results["gate"][0]["ply"]).is_file()
        assert results["replay"]["frames"] == 4
    else:
        with pytest.raises(GateFailed):
            phase0_steps.run_locally(params, "manual__phase0-test")

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("phase0-pipeline")
    runs = client.search_runs([experiment.experiment_id])
    (parent,) = [r for r in runs if "basir.pipeline" in r.data.tags]
    children = {r.data.tags["basir.step"]: r for r in runs if r is not parent}

    status = "succeeded" if passes else "failed"
    assert parent.data.tags["basir.pipeline"] == "phase0"
    assert parent.data.tags["basir.workflow_id"] == "manual__phase0-test"
    assert parent.data.tags["basir.status"] == status
    assert parent.info.status == ("FINISHED" if passes else "FAILED")
    assert parent.data.params["every"] == "1"

    dag = phase0_dag(datasets)
    logged = json.loads(Path(client.download_artifacts(parent.info.run_id, "dag.json", str(tmp_path))).read_text())
    assert logged == dag.to_json()
    assert {a.path for a in client.list_artifacts(parent.info.run_id)} == {"dag.json", "dag.mmd"}

    assert set(children) == {s.name for s in dag.steps}
    for step in dag.steps:
        tags = children[step.name].data.tags
        assert tags["mlflow.parentRunId"] == parent.info.run_id
        assert tags["basir.depends_on"] == ",".join(step.depends_on)
        expected = status if step.name == "gate" else "succeeded"
        assert tags["basir.status"] == expected, step.name

    cloud_run = children[f"pointcloud:{datasets[0]}"]
    assert cloud_run.data.metrics["frames_used"] == 4
    assert {"toy_a.ply", "toy_a.topdown.png", "toy_a.trajectory.csv"} <= {
        a.path for a in client.list_artifacts(cloud_run.info.run_id)
    }
    assert children[f"replay:{datasets[0]}"].data.metrics["frames"] == 4


def test_runner_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake(task: str, payload: dict[str, object]) -> object:
        if task == "bad":
            raise ValueError("no such dataset")
        if task == "flaky":
            raise ConnectionError("MLflow went away")
        return {"echo": payload}

    monkeypatch.setitem(runner.PIPELINES, "fake", lambda: (fake, (ValueError,)))
    source, target = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps({"x": 1}))
    assert runner.main(["fake", "ok", str(source), str(target)]) == 0
    assert json.loads(target.read_text()) == {"echo": {"x": 1}}
    assert runner.main(["fake", "bad", str(source), str(target)]) == runner.EXIT_NO_RETRY
    assert f"{runner.ERROR_MARKER} ValueError: no such dataset" in capsys.readouterr().err
    assert runner.main(["fake", "flaky", str(source), str(target)]) == 1
    assert runner.main(["nope"]) == 2


def test_run_step_runs_the_step_in_a_subprocess(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Airflow side: Basir's interpreter (here the test one) runs the step; data errors do not retry."""
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    clean_env.setenv("BASIR_PYTHON", sys.executable)
    payload = {"params": {"datasets": [DESK]}, "parent_run_id": None}
    clouds = [asdict(_cloud(DESK, 1.9))]
    assert airflow_task.run_step("phase0", "gate", payload | {"clouds": clouds}) == clouds
    with pytest.raises(airflow_task.StepFailed):
        airflow_task.run_step("phase0", "gate", payload | {"clouds": [asdict(_cloud(DESK, 9.0))]})
    with pytest.raises(airflow_task.StepCrashed):
        airflow_task.run_step("phase0", "gate", payload)  # KeyError 'clouds': a bug, worth a retry


def test_bookkeeping_task_names_match_the_dag_file() -> None:
    text = (Path(__file__).resolve().parent.parent / "dags" / "phase0.py").read_text()
    assert "task_id=OPEN_TASK" in text and "task_id=CLOSE_TASK" in text
    assert (OPEN_TASK, CLOSE_TASK) == ("open_pipeline_run", "close_pipeline_run")
