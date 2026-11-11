"""Phase 0 pipeline as a Temporal workflow: prepare, replay and fuse the datasets, then gate on the targets.

    check_env -> prepare_dataset:<each dataset>        (in parallel)
    prepare_dataset:<first> -> replay:<first>
    prepare_dataset:<each>  -> pointcloud:<each>
    replay + every pointcloud -> gate                 (passes only if every cloud meets the targets)

The workflow walks the DAG from `phase0_dag`, starting each step as soon as its
dependencies finish, so the DAG logged to MLflow is exactly what runs. All I/O,
including MLflow, happens in the activities (`phase0_steps`); this code stays deterministic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from basir.pipelines.dag import Dag, Step

with workflow.unsafe.imports_passed_through():
    from basir.pipelines.phase0_steps import (
        CloseRunInput,
        DatasetInput,
        GateInput,
        OpenRunInput,
        PointcloudInput,
        PointcloudResult,
        PreparedDataset,
        ReplayInput,
        ReplayResult,
        StepContext,
        check_env,
        close_pipeline_run,
        gate,
        open_pipeline_run,
        pointcloud,
        prepare_dataset,
        replay,
    )

PIPELINE = "phase0"
DEFAULT_DATASETS = ("fr1/desk", "fr1/room")

_BOOKKEEPING = RetryPolicy(maximum_attempts=3)
_DATA_ERRORS = ["FileNotFoundError", "ValueError"]  # bad input or missing data: retrying will not help
_HEARTBEAT = timedelta(minutes=1)
# Every step opens an MLflow run; a cold start (imports, database migration) can take minutes on a slow disk.
_QUICK = timedelta(minutes=5)


@dataclass
class Phase0Input:
    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    replay_stride: int = 1
    replay_speed: float = 0.0  # 0 = as fast as possible
    every: int = 15
    pixel_stride: int = 4
    voxel: float = 0.02
    gap: int = 1
    max_frames: int | None = None


@dataclass
class Phase0Result:
    mlflow_run_id: str | None
    passed: bool
    replay: ReplayResult | None
    clouds: list[PointcloudResult]


def phase0_dag(datasets: list[str]) -> Dag:
    """The Phase 0 DAG; the first dataset is also replayed."""
    if not datasets:
        raise ValueError("phase0 needs at least one dataset")
    steps = [Step("check_env", "check_env")]
    steps += [Step(f"prepare_dataset:{d}", "prepare_dataset", d, ("check_env",)) for d in datasets]
    steps.append(Step(f"replay:{datasets[0]}", "replay", datasets[0], (f"prepare_dataset:{datasets[0]}",)))
    steps += [Step(f"pointcloud:{d}", "pointcloud", d, (f"prepare_dataset:{d}",)) for d in datasets]
    steps.append(Step("gate", "gate", None, tuple(s.name for s in steps if s.activity in ("replay", "pointcloud"))))
    return Dag(tuple(steps))


@workflow.defn
class Phase0Pipeline:
    def __init__(self) -> None:
        self._run_id: str | None = None

    @workflow.query
    def mlflow_run_id(self) -> str | None:
        """The parent MLflow run, once the first activity has opened it."""
        return self._run_id

    @workflow.run
    async def run(self, options: Phase0Input) -> Phase0Result:
        try:
            dag = phase0_dag(options.datasets)
        except ValueError as e:
            raise ApplicationError(str(e), non_retryable=True) from e

        self._run_id = await workflow.execute_activity(
            open_pipeline_run,
            OpenRunInput(
                pipeline=PIPELINE,
                workflow_id=workflow.info().workflow_id,
                dag=dag.to_json(),
                mermaid=dag.to_mermaid(),
                params={k: v for k, v in vars(options).items() if v is not None},
            ),
            start_to_close_timeout=_QUICK,
            retry_policy=_BOOKKEEPING,
        )
        try:
            results = await self._execute(dag, options)
        except BaseException:
            await self._close(succeeded=False)
            raise
        await self._close(succeeded=True)
        return Phase0Result(
            mlflow_run_id=self._run_id,
            passed=True,
            replay=next((r for r in results.values() if isinstance(r, ReplayResult)), None),
            clouds=[r for r in results.values() if isinstance(r, PointcloudResult)],
        )

    async def _execute(self, dag: Dag, options: Phase0Input) -> dict[str, Any]:
        """Start every step as a task that first awaits its dependencies; return each step's result."""
        tasks: dict[str, asyncio.Task[Any]] = {}
        for step in dag.steps:
            tasks[step.name] = asyncio.create_task(self._step(step, [tasks[d] for d in step.depends_on], options))
        try:
            await asyncio.gather(*tasks.values())
        except BaseException:
            for task in tasks.values():
                task.cancel()
            raise
        return {name: task.result() for name, task in tasks.items()}

    async def _step(self, step: Step, dependencies: list[asyncio.Task[Any]], options: Phase0Input) -> Any:
        inputs = [await task for task in dependencies]
        ctx = StepContext(step.name, list(step.depends_on), self._run_id)
        prepared = next((i for i in inputs if isinstance(i, PreparedDataset)), None)
        if step.activity == "check_env":
            return await workflow.execute_activity(
                check_env,
                ctx,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        if step.activity == "prepare_dataset":
            assert step.dataset is not None
            return await workflow.execute_activity(
                prepare_dataset,
                DatasetInput(ctx, step.dataset),
                start_to_close_timeout=timedelta(hours=2),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=10), maximum_attempts=3, non_retryable_error_types=_DATA_ERRORS
                ),
            )
        if step.activity == "replay":
            assert prepared is not None
            return await workflow.execute_activity(
                replay,
                ReplayInput(ctx, prepared, options.replay_stride, options.replay_speed, options.max_frames),
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=_HEARTBEAT,
                retry_policy=RetryPolicy(maximum_attempts=2, non_retryable_error_types=_DATA_ERRORS),
            )
        if step.activity == "pointcloud":
            assert prepared is not None
            return await workflow.execute_activity(
                pointcloud,
                PointcloudInput(
                    ctx, prepared, options.every, options.pixel_stride, options.voxel, options.gap, options.max_frames
                ),
                start_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=_HEARTBEAT,
                retry_policy=RetryPolicy(maximum_attempts=2, non_retryable_error_types=_DATA_ERRORS),
            )
        if step.activity == "gate":
            return await workflow.execute_activity(
                gate,
                GateInput(ctx, [i for i in inputs if isinstance(i, PointcloudResult)]),
                start_to_close_timeout=_QUICK,
                retry_policy=RetryPolicy(maximum_attempts=1),  # deterministic: a retry gives the same verdict
            )
        raise ApplicationError(f"unknown activity '{step.activity}'", non_retryable=True)

    async def _close(self, succeeded: bool) -> None:
        await workflow.execute_activity(
            close_pipeline_run,
            CloseRunInput(self._run_id, succeeded),
            start_to_close_timeout=_QUICK,
            retry_policy=_BOOKKEEPING,
        )
