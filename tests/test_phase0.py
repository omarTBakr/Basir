"""Tests for basir/pipelines/phase0.py: the Phase 0 workflow on a Temporal test server.

The time-skipping test server is downloaded on first use; offline, the dev server of an
installed Temporal CLI (`temporal`) is used instead.

Fake activities check ordering, parallelism and the gate; the real activities run on toy
datasets to check the MLflow parent/child runs, their tags and the logged DAG.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from basir.ingest.datasets.toy import create_toy_dataset
from basir.pipelines import phase0_steps
from basir.pipelines.phase0 import Phase0Input, Phase0Pipeline, Phase0Result, phase0_dag
from basir.pipelines.phase0_steps import (
    CloseRunInput,
    DatasetInput,
    OpenRunInput,
    PointcloudInput,
    PointcloudResult,
    PreparedDataset,
    ReplayInput,
    ReplayResult,
    StepContext,
)
from basir.pipelines.runtime import ACTIVITIES, TASK_QUEUE, workflow_url

DESK, ROOM = "fr1/desk", "fr1/room"
_offline = False  # set once the test-server download has failed, so later tests skip straight to the CLI


pytestmark = pytest.mark.slow  # the Temporal test server (downloaded on first use) and real workflows


async def _environment() -> WorkflowEnvironment:
    global _offline
    if not _offline:
        try:
            return await WorkflowEnvironment.start_time_skipping()
        except RuntimeError:
            _offline = True
    if (cli := shutil.which("temporal")) is None:
        pytest.skip("no Temporal test server: offline and no `temporal` CLI installed")
    return await WorkflowEnvironment.start_local(dev_server_existing_path=cli)


async def _execute(options: Phase0Input, activities: Sequence[Callable[..., Any]]) -> Phase0Result:
    async with (
        await _environment() as env,
        Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[Phase0Pipeline],
            activities=activities,
            activity_executor=ThreadPoolExecutor(max_workers=4),
        ),
    ):
        return await env.client.execute_workflow(Phase0Pipeline.run, options, id="phase0-test", task_queue=TASK_QUEUE)


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
    with pytest.raises(ValueError):
        phase0_dag([])


class FakeSteps:
    """Fast stand-ins for the activities that record when each step starts and ends."""

    def __init__(self, median_cm: float) -> None:
        self.median_cm = median_cm
        self.events: list[tuple[str, str]] = []
        self.closed: list[bool] = []
        self.both_preparing = threading.Barrier(2, timeout=10)  # raises unless the two prepares overlap
        self._lock = threading.Lock()

    def _log(self, kind: str, step: str) -> None:
        with self._lock:
            self.events.append((kind, step))

    def activities(self) -> list[Callable[..., Any]]:
        @activity.defn(name="open_pipeline_run")
        def open_run(data: OpenRunInput) -> str | None:
            return "parent-run"

        @activity.defn(name="close_pipeline_run")
        def close_run(data: CloseRunInput) -> None:
            self.closed.append(data.succeeded)

        @activity.defn(name="check_env")
        def check_env(ctx: StepContext) -> dict[str, str]:
            self._log("start", ctx.step)
            self._log("end", ctx.step)
            return {}

        @activity.defn(name="prepare_dataset")
        def prepare(data: DatasetInput) -> PreparedDataset:
            self._log("start", data.ctx.step)
            assert data.ctx.parent_run_id == "parent-run" and data.ctx.depends_on == ["check_env"]
            self.both_preparing.wait()
            self._log("end", data.ctx.step)
            return PreparedDataset(data.dataset, f"/data/{data.dataset}", 100)

        @activity.defn(name="replay")
        def replay(data: ReplayInput) -> ReplayResult:
            self._log("start", data.ctx.step)
            self._log("end", data.ctx.step)
            return ReplayResult(data.prepared.dataset, 100, 30.0, 95.0)

        @activity.defn(name="pointcloud")
        def pointcloud(data: PointcloudInput) -> PointcloudResult:
            self._log("start", data.ctx.step)
            time.sleep(0.05)
            self._log("end", data.ctx.step)
            return PointcloudResult(data.prepared.dataset, "/out.ply", 10, 1000, self.median_cm, 0.85, 9)

        return [open_run, close_run, check_env, prepare, replay, pointcloud, phase0_steps.gate]

    def index(self, kind: str, step: str) -> int:
        return self.events.index((kind, step))


def test_steps_follow_the_dag_and_prepares_run_in_parallel(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    fake = FakeSteps(median_cm=1.9)
    result = asyncio.run(_execute(Phase0Input(), fake.activities()))

    assert result.passed and result.mlflow_run_id == "parent-run"
    assert [c.dataset for c in result.clouds] == [DESK, ROOM]
    assert result.replay is not None and result.replay.dataset == DESK
    assert fake.closed == [True]

    i = fake.index
    assert fake.events[:2] == [("start", "check_env"), ("end", "check_env")]
    assert max(i("start", f"prepare_dataset:{d}") for d in (DESK, ROOM)) < min(
        i("end", f"prepare_dataset:{d}") for d in (DESK, ROOM)
    )
    assert i("end", f"prepare_dataset:{DESK}") < i("start", f"replay:{DESK}")
    for d in (DESK, ROOM):
        assert i("end", f"prepare_dataset:{d}") < i("start", f"pointcloud:{d}")
    assert len(fake.events) == 12  # gate is the real activity, so it logs no fake events


def test_gate_failure_fails_the_workflow(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    fake = FakeSteps(median_cm=4.2)
    with pytest.raises(WorkflowFailureError) as failure:
        asyncio.run(_execute(Phase0Input(), fake.activities()))

    cause = failure.value.cause.cause
    assert isinstance(cause, ApplicationError) and cause.type == "GateFailed"
    assert f"{DESK}: median 4.20 cm" in str(cause) and ROOM in str(cause)
    assert fake.closed == [False]


@pytest.mark.parametrize("passes", [True, False])
def test_real_steps_log_the_dag_to_mlflow(passes: bool, clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    clean_env.setattr(phase0_steps, "meets_targets", lambda median, within: passes)
    datasets = [str(create_toy_dataset(tmp_path / name, num_frames=4)) for name in ("toy_a", "toy_b")]
    options = Phase0Input(datasets=datasets, every=1, pixel_stride=1, voxel=0)

    if passes:
        result = asyncio.run(_execute(options, ACTIVITIES))
        assert [c.frames_used for c in result.clouds] == [4, 4]
        assert Path(result.clouds[0].ply).is_file()
    else:
        with pytest.raises(WorkflowFailureError):
            asyncio.run(_execute(options, ACTIVITIES))

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("phase0-pipeline")
    runs = client.search_runs([experiment.experiment_id])
    (parent,) = [r for r in runs if "basir.pipeline" in r.data.tags]
    children = {r.data.tags["basir.step"]: r for r in runs if r is not parent}

    status = "succeeded" if passes else "failed"
    assert parent.data.tags["basir.pipeline"] == "phase0"
    assert parent.data.tags["basir.workflow_id"] == "phase0-test"
    assert parent.data.tags["basir.status"] == status
    assert parent.info.status == ("FINISHED" if passes else "FAILED")

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


def test_workflow_url_uses_the_temporal_ui_setting(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("TEMPORAL_UI_URL", "http://gpu-box:8080/")
    assert workflow_url("phase0-x") == "http://gpu-box:8080/namespaces/default/workflows/phase0-x"
