#!/usr/bin/env python3
"""Trigger a Basir pipeline (an Airflow DAG) through Airflow's REST API; the stack must be up.

Prints the DAG run ID, its Airflow UI page and the MLflow parent run.

    python scripts/run_pipeline.py phase0                          # fr1/desk + fr1/room
    python scripts/run_pipeline.py phase0 --datasets fr1/desk --wait
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
import sys
import time
from typing import Any
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basir.pipelines.airflow_client import FINISHED, AirflowClient, AirflowError, credentials, run_url
from basir.pipelines.phase0 import DAG_ID, DEFAULT_DATASETS, OPEN_TASK, Phase0Params
from basir.pipelines.runner import ERROR_MARKER
from utilities.config import get_settings
from utilities.experiment import run_url as mlflow_run_url

RUN_ID_WAIT_SECONDS = 120.0  # the first task opens the parent run; a cold start takes a while
POLL_SECONDS = 5.0


def start(
    client: AirflowClient,
    options: Phase0Params,
    wait: bool,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    run_id = f"phase0-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    client.unpause(DAG_ID)
    client.trigger(DAG_ID, conf={k: v for k, v in vars(options).items() if v is not None}, run_id=run_id)
    print(f"Run       {run_id}")
    print(f"Airflow   {run_url(client.base_url, DAG_ID, run_id)}")

    parent = _parent_run_id(client, run_id, sleep, clock)
    if parent is None:
        print("MLflow    parent run not open yet (are the Airflow scheduler and MLflow up?)")
    else:
        print(f"MLflow    {mlflow_run_url(parent) or f'run {parent} in {get_settings().mlflow_uri}'}")
    if not wait:
        return 0

    while (state := client.dag_run(DAG_ID, run_id)["state"]) not in FINISHED:
        sleep(POLL_SECONDS)
    if state != "success":
        print(f"FAILED    {_failure(client, run_id)}")
        return 1
    for cloud in client.xcom(DAG_ID, run_id, "gate") or []:
        print(
            f"PASS      {cloud['dataset']}: median {cloud['median_cm']:.2f} cm, "
            f"{cloud['within_5cm'] or 0:.0%} within 5 cm ({cloud['frames_used']} frames, {cloud['points']:,} points)"
        )
    return 0


def _parent_run_id(
    client: AirflowClient, run_id: str, sleep: Callable[[float], None], clock: Callable[[], float]
) -> str | None:
    """The MLflow parent run that the first task pushes as an XCom, waiting for it to run."""
    deadline = clock() + RUN_ID_WAIT_SECONDS
    while clock() < deadline:
        parent = client.xcom(DAG_ID, run_id, OPEN_TASK, key="parent_run_id")
        if parent is not None:
            return str(parent) or None  # "": MLflow tracking was off
        if client.dag_run(DAG_ID, run_id)["state"] in FINISHED:
            return None
        sleep(1.0)
    return None


def _failure(client: AirflowClient, run_id: str) -> str:
    """The first failed task and the error its step reported (from the task log)."""
    tasks: list[dict[str, Any]] = client.task_instances(DAG_ID, run_id)
    failed = [t for t in tasks if t.get("state") == "failed"]
    if not failed:
        return f"DAG run ended without a failed task ({', '.join(sorted({str(t.get('state')) for t in tasks}))})"
    first = min(failed, key=lambda t: t.get("end_date") or "")
    name = first["task_id"] + (f"[{first['rendered_map_index']}]" if first.get("rendered_map_index") else "")
    try:
        log = client.task_log(DAG_ID, run_id, first["task_id"], int(first.get("try_number") or 1), first["map_index"])
    except AirflowError:
        log = ""
    errors = [line.split(ERROR_MARKER, 1)[1].strip() for line in log.splitlines() if ERROR_MARKER in line]
    return f"{name}: {errors[-1] if errors else 'failed; see its log in Airflow'}"


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

    options = Phase0Params(
        datasets=args.datasets,
        replay_stride=args.replay_stride,
        every=args.every,
        pixel_stride=args.pixel_stride,
        voxel=args.voxel,
        max_frames=args.max_frames,
    )
    settings = get_settings()
    try:
        with AirflowClient(settings.airflow_url, credentials(settings)) as client:
            return start(client, options, args.wait)
    except AirflowError as e:
        print(f"Airflow at {settings.airflow_url}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
