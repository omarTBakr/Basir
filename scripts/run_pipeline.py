#!/usr/bin/env python3
"""Start a Basir pipeline on Temporal; a worker (scripts/worker.py) must be running.

Prints the workflow ID, its Temporal UI page and the MLflow parent run.

    python scripts/run_pipeline.py phase0                          # fr1/desk + fr1/room
    python scripts/run_pipeline.py phase0 --datasets fr1/desk --wait
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporalio.client import WorkflowFailureError, WorkflowHandle
from temporalio.exceptions import ActivityError

from basir.pipelines.phase0 import DEFAULT_DATASETS, Phase0Input, Phase0Pipeline, Phase0Result
from basir.pipelines.runtime import TASK_QUEUE, connect, workflow_url
from utilities.config import get_settings
from utilities.experiment import run_url

RUN_ID_WAIT_SECONDS = 30.0


async def start(options: Phase0Input, wait: bool) -> int:
    client = await connect()
    workflow_id = f"phase0-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    handle = await client.start_workflow(Phase0Pipeline.run, options, id=workflow_id, task_queue=TASK_QUEUE)
    print(f"Workflow  {workflow_id}")
    print(f"Temporal  {workflow_url(workflow_id)}")

    run_id = await _parent_run_id(handle)
    if run_id is None:
        print("MLflow    parent run not open yet (is a worker running?)")
    else:
        print(f"MLflow    {run_url(run_id) or f'run {run_id} in {get_settings().mlflow_uri}'}")
    if not wait:
        return 0

    try:
        result: Phase0Result = await handle.result()
    except WorkflowFailureError as e:
        cause = e.cause.cause if isinstance(e.cause, ActivityError) and e.cause.cause else e.cause
        print(f"FAILED    {cause}")
        return 1
    for cloud in result.clouds:
        print(
            f"PASS      {cloud.dataset}: median {cloud.median_cm:.2f} cm, "
            f"{cloud.within_5cm or 0:.0%} within 5 cm ({cloud.frames_used} frames, {cloud.points:,} points)"
        )
    return 0


async def _parent_run_id(handle: WorkflowHandle) -> str | None:
    """Ask the workflow for its MLflow parent run, waiting briefly for the first activity to open it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + RUN_ID_WAIT_SECONDS
    while loop.time() < deadline:
        run_id = await handle.query(Phase0Pipeline.mlflow_run_id)
        if run_id is not None:
            return run_id
        await asyncio.sleep(0.5)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", choices=["phase0"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Preset keys or folders.")
    parser.add_argument("--every", type=int, default=15, help="Point cloud: use every Nth frame.")
    parser.add_argument("--pixel-stride", type=int, default=4, help="Point cloud: use every Nth pixel.")
    parser.add_argument("--voxel", type=float, default=0.02, help="Point cloud: voxel size in metres; 0 = off.")
    parser.add_argument("--replay-stride", type=int, default=1, help="Replay: use every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wait", action="store_true", help="Wait for the result; exit 1 if the pipeline fails.")
    args = parser.parse_args(argv)

    options = Phase0Input(
        datasets=args.datasets,
        replay_stride=args.replay_stride,
        every=args.every,
        pixel_stride=args.pixel_stride,
        voxel=args.voxel,
        max_frames=args.max_frames,
    )
    try:
        return asyncio.run(start(options, args.wait))
    except RuntimeError as e:  # temporalio raises RuntimeError when the server is unreachable
        print(f"Cannot reach Temporal at {get_settings().temporal_address}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
