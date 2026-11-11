"""Connecting to Temporal: the client, the worker that runs Basir's pipelines, and Temporal UI links."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from temporalio.client import Client
from temporalio.worker import Worker

from basir.pipelines import phase0_steps
from basir.pipelines.phase0 import Phase0Pipeline
from utilities.config import get_settings

TASK_QUEUE = "basir-pipelines"
NAMESPACE = "default"

ACTIVITIES: list[Callable[..., Any]] = [
    phase0_steps.open_pipeline_run,
    phase0_steps.close_pipeline_run,
    phase0_steps.check_env,
    phase0_steps.prepare_dataset,
    phase0_steps.replay,
    phase0_steps.pointcloud,
    phase0_steps.gate,
]
WORKFLOWS = [Phase0Pipeline]


async def connect() -> Client:
    """A client for the Temporal frontend at TEMPORAL_ADDRESS."""
    return await Client.connect(get_settings().temporal_address, namespace=NAMESPACE)


def create_worker(client: Client, max_concurrent_activities: int = 4) -> Worker:
    """A worker on TASK_QUEUE for every pipeline; activities run in threads, so parallel steps overlap."""
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=WORKFLOWS,
        activities=ACTIVITIES,
        activity_executor=ThreadPoolExecutor(max_workers=max_concurrent_activities),
        max_concurrent_activities=max_concurrent_activities,
    )


def workflow_url(workflow_id: str) -> str:
    """The workflow's page in the Temporal UI at TEMPORAL_UI_URL."""
    return f"{get_settings().temporal_ui_url.rstrip('/')}/namespaces/{NAMESPACE}/workflows/{workflow_id}"
