"""The Temporal layer of the UI: running and recent workflows with their status (read-only).

Every call is bounded by a timeout; when Temporal is down the snapshot says so instead of raising.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote

from temporalio.client import Client

NAMESPACE = "default"


@dataclass(frozen=True)
class WorkflowSummary:
    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str
    status: str  # running, completed, failed, ...
    started: datetime | None
    closed: datetime | None
    url: str


@dataclass
class TemporalSnapshot:
    reachable: bool
    error: str | None = None
    workflows: list[WorkflowSummary] = field(default_factory=list)

    def by_id(self, workflow_id: str | None) -> WorkflowSummary | None:
        """The most recent run of a workflow ID, if listed."""
        return next((w for w in self.workflows if w.workflow_id == workflow_id), None)


class TemporalView:
    """List workflows from a Temporal frontend, reconnecting after failures."""

    def __init__(self, address: str, ui_url: str, timeout: float = 2.0) -> None:
        self.address = address
        self.ui_url = ui_url.rstrip("/")
        self.timeout = timeout
        self._client: Client | None = None

    async def snapshot(self, limit: int = 20) -> TemporalSnapshot:
        """The newest `limit` workflows (running first), or an unreachable snapshot."""
        try:
            workflows = await asyncio.wait_for(self._list(limit), self.timeout)
        except Exception as e:
            self._client = None
            reason = "timed out" if isinstance(e, TimeoutError) else (str(e).splitlines() or [type(e).__name__])[0]
            return TemporalSnapshot(reachable=False, error=f"{self.address}: {reason[:200]}")
        workflows.sort(key=lambda w: (w.status != "running", -(w.started.timestamp() if w.started else 0)))
        return TemporalSnapshot(reachable=True, workflows=workflows)

    async def _list(self, limit: int) -> list[WorkflowSummary]:
        if self._client is None:
            self._client = await Client.connect(self.address, namespace=NAMESPACE)
        return [
            WorkflowSummary(
                workflow_id=w.id,
                run_id=w.run_id,
                workflow_type=w.workflow_type,
                task_queue=w.task_queue,
                status=w.status.name.lower() if w.status else "unknown",
                started=w.start_time,
                closed=w.close_time,
                url=f"{self.ui_url}/namespaces/{NAMESPACE}/workflows/{quote(w.id, safe='')}/{w.run_id}/history",
            )
            async for w in self._client.list_workflows(limit=limit)
        ]
