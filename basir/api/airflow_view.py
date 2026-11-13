"""The Airflow layer of the UI: running and recent DAG runs with their task states (read-only).

Reads Airflow 3's REST API (`/api/v2`, JWT from `POST /auth/token`). Every call is bounded by a
timeout; when Airflow is down the snapshot says so instead of raising. Task instances map onto
the nodes of the run's `dag.json` (see `node_states`), so the Pipelines page colours the same
graph MLflow shows, with Airflow's live states (queued, up_for_retry, upstream_failed, ...).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from basir.api.dag import Dag
from basir.pipelines.airflow_client import TOKEN_PATH, Credentials, error_detail, run_url, task_url

API = "/api/v2"


@dataclass(frozen=True)
class TaskState:
    task_id: str
    map_index: int  # -1 for a task that is not mapped
    label: str | None  # the mapped instance's name (rendered map index), e.g. "fr1/desk"
    state: str  # success, failed, running, queued, upstream_failed, ... ("none" = not scheduled yet)
    try_number: int
    url: str


@dataclass(frozen=True)
class DagRunSummary:
    dag_id: str
    run_id: str
    run_type: str  # manual, scheduled, backfill, asset_triggered
    state: str  # queued, running, success, failed
    started: datetime | None
    ended: datetime | None
    url: str
    tasks: list[TaskState] = field(default_factory=list)

    def node_states(self, node_ids: list[str]) -> dict[str, str]:
        """Each `dag.json` node's Airflow state; see the module function."""
        return node_states(self.tasks, node_ids)


@dataclass
class AirflowSnapshot:
    reachable: bool
    error: str | None = None
    runs: list[DagRunSummary] = field(default_factory=list)

    def by_run_id(self, run_id: str | None) -> DagRunSummary | None:
        """The listed DAG run with this run id (= the MLflow parent run's `basir.workflow_id`)."""
        return next((r for r in self.runs if r.run_id == run_id), None)

    def colour(self, dag: Dag, run: DagRunSummary | None) -> Dag:
        """`dag` (laid out from MLflow) with each node's status replaced by Airflow's, when Airflow knows it."""
        if run is None:
            return dag
        states = run.node_states([n.name for n in dag.nodes])
        return replace(dag, nodes=[replace(n, status=states.get(n.name, n.status)) for n in dag.nodes])


def node_states(tasks: list[TaskState], node_ids: list[str]) -> dict[str, str]:
    """Map task instances onto `dag.json` node ids (`<task id>` or `<task id>:<dataset>`).

    A node `<task>:<dataset>` takes the mapped instance labelled `<dataset>`; failing a label
    (a mapped instance that has not rendered its name yet), the instance whose map index is the
    node's position among that task's nodes, which is the dataset order of both. An unmapped
    task takes its own node, or the only node of that task (replay:<first dataset>).
    """
    by_task: dict[str, list[str]] = {}
    for node in node_ids:
        by_task.setdefault(node.split(":", 1)[0], []).append(node)
    states: dict[str, str] = {}
    for ti in tasks:
        nodes = by_task.get(ti.task_id, [])
        if ti.label and f"{ti.task_id}:{ti.label}" in nodes:
            node = f"{ti.task_id}:{ti.label}"
        elif ti.map_index >= 0 and ti.map_index < len(nodes):
            node = nodes[ti.map_index]
        elif ti.map_index < 0 and len(nodes) == 1:
            node = nodes[0]
        else:
            continue
        states[node] = ti.state
    return states


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class AirflowView:
    """List DAG runs and their task instances from an Airflow API server, logging in again when the token expires."""

    def __init__(
        self,
        base_url: str,
        creds: Credentials,
        timeout: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._creds = creds
        self._transport = transport
        self._token: str | None = None

    async def snapshot(self, limit: int = 15) -> AirflowSnapshot:
        """The newest `limit` DAG runs (running first) with their tasks, or an unreachable snapshot."""
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self._transport
            ) as client:
                runs = await asyncio.wait_for(self._runs(client, limit), self.timeout * 3)
        except Exception as e:
            self._token = None
            reason = "timed out" if isinstance(e, TimeoutError) else (str(e).splitlines() or [type(e).__name__])[0]
            return AirflowSnapshot(reachable=False, error=f"{self.base_url}: {reason[:200]}")
        runs.sort(key=lambda r: (r.state not in ("running", "queued"), -(r.started.timestamp() if r.started else 0)))
        return AirflowSnapshot(reachable=True, runs=runs)

    async def _runs(self, client: httpx.AsyncClient, limit: int) -> list[DagRunSummary]:
        body = await self._get(client, "dags/~/dagRuns", {"limit": limit, "order_by": "-run_after"})
        runs = [
            DagRunSummary(
                dag_id=r["dag_id"],
                run_id=r["dag_run_id"],
                run_type=r.get("run_type") or "",
                state=r.get("state") or "queued",
                started=_time(r.get("start_date")),
                ended=_time(r.get("end_date")),
                url=run_url(self.base_url, r["dag_id"], r["dag_run_id"]),
            )
            for r in body.get("dag_runs", [])
        ]
        task_lists = await asyncio.gather(*(self._tasks(client, r) for r in runs))
        return [replace(r, tasks=tasks) for r, tasks in zip(runs, task_lists, strict=True)]

    async def _tasks(self, client: httpx.AsyncClient, run: DagRunSummary) -> list[TaskState]:
        path = f"dags/{quote(run.dag_id, safe='')}/dagRuns/{quote(run.run_id, safe='')}/taskInstances"
        body = await self._get(client, path, {"limit": 500})
        return [
            TaskState(
                task_id=t["task_id"],
                map_index=int(t.get("map_index", -1)),
                label=t.get("rendered_map_index"),
                state=t.get("state") or "none",
                try_number=int(t.get("try_number") or 0),
                url=task_url(self.base_url, run.dag_id, run.run_id, t["task_id"], int(t.get("map_index", -1))),
            )
            for t in body.get("task_instances", [])
        ]

    async def _get(self, client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(2):
            if self._token is None:
                self._token = await self._login(client)
            response = await client.get(
                f"{API}/{path}", params=params, headers={"Authorization": f"Bearer {self._token}"}
            )
            if response.status_code == 401 and attempt == 0:
                self._token = None
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"GET {path}: {error_detail(response)}")
            return dict(response.json())
        raise AssertionError("unreachable")

    async def _login(self, client: httpx.AsyncClient) -> str:
        response = await client.post(
            TOKEN_PATH, json={"username": self._creds.username, "password": self._creds.password}
        )
        if response.status_code >= 400:
            raise RuntimeError(f"login as '{self._creds.username}' failed: {error_detail(response)}")
        return str(response.json()["access_token"])
