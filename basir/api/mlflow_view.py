"""The MLflow layer of the UI: a compact answer to "what is running, and did the last runs pass?".

Reads only. Pipeline runs are recognised by the tags the pipeline writes: the parent has
`basir.pipeline`, `basir.workflow_id`, `basir.status` and a `dag.json` artifact; each step
is a child run (`mlflow.parentRunId`) with `basir.step`, `basir.depends_on` and `basir.status`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient

from basir.api.dag import Dag, layout, parse_dag

# Metrics worth a glance, in display order; a run shows the first few it has.
KEY_METRICS = (
    "consistency_median_cm",
    "consistency_within_5cm",
    "consistency_p90_cm",
    "fps",
    "frames_used",
    "points_downsampled",
    "seconds",
)
MAX_METRICS = 5
PASS_METRICS = ("passes", "passed")  # 1.0 / 0.0 verdicts logged by the evaluation scripts
PARENT_TAG = "mlflow.parentRunId"


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    experiment: str
    name: str
    verdict: str  # running, pass, fail, done or failed
    started: datetime | None
    duration_s: float | None
    metrics: dict[str, float]
    url: str | None


@dataclass(frozen=True)
class PipelineRun:
    run: RunSummary
    pipeline: str
    workflow_id: str | None
    status: str
    dag: Dag


@dataclass
class MLflowSnapshot:
    reachable: bool
    error: str | None = None
    running: list[RunSummary] = field(default_factory=list)
    latest: list[RunSummary] = field(default_factory=list)
    pipelines: list[PipelineRun] = field(default_factory=list)


class MLflowView:
    """Query an MLflow tracking store and summarise it for the Experiments and Pipelines pages."""

    def __init__(self, tracking_uri: str, ui_url: str | None) -> None:
        # Fail fast when an MLflow server is down instead of retrying for minutes.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.ui_url = ui_url
        self._dags: dict[str, Any] = {}  # run_id -> parsed dag.json (it never changes once logged)

    def experiments(self, runs_per_experiment: int = 20) -> MLflowSnapshot:
        """Top-level runs in progress, and the latest top-level run of each experiment."""
        try:
            names = {e.experiment_id: e.name for e in self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)}
            running = self.client.search_runs(list(names), "attributes.status = 'RUNNING'", max_results=50)
            latest = []
            for experiment_id in names:
                runs = self.client.search_runs(
                    [experiment_id], max_results=runs_per_experiment, order_by=["attributes.start_time DESC"]
                )
                top = next((r for r in runs if PARENT_TAG not in r.data.tags), None)
                if top is not None:
                    latest.append(top)
        except Exception as e:
            return MLflowSnapshot(reachable=False, error=_short(e))
        latest.sort(key=lambda r: r.info.start_time or 0, reverse=True)
        return MLflowSnapshot(
            reachable=True,
            running=[self._summary(r, names) for r in running if PARENT_TAG not in r.data.tags],
            latest=[self._summary(r, names) for r in latest],
        )

    def pipelines(self, limit: int = 10) -> MLflowSnapshot:
        """The most recent pipeline runs, each with its DAG coloured by step status."""
        try:
            names = {e.experiment_id: e.name for e in self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)}
            parents = self.client.search_runs(
                list(names),
                "tags.`basir.pipeline` LIKE '%'",
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )
            pipelines = [self._pipeline(run, names) for run in parents]
        except Exception as e:
            return MLflowSnapshot(reachable=False, error=_short(e))
        return MLflowSnapshot(reachable=True, pipelines=pipelines)

    def _pipeline(self, parent: Run, names: dict[str, str]) -> PipelineRun:
        children = self.client.search_runs(
            [parent.info.experiment_id], f"tags.`{PARENT_TAG}` = '{parent.info.run_id}'", max_results=200
        )
        statuses, edges = {}, []
        for child in children:
            tags = child.data.tags
            step = tags.get("basir.step") or child.info.run_name or child.info.run_id
            statuses[step] = tags.get("basir.status") or child.info.status
            edges += [(dep.strip(), step) for dep in tags.get("basir.depends_on", "").split(",") if dep.strip()]
        steps, dag_edges = parse_dag(self._dag_json(parent.info.run_id))
        if not steps:  # no dag.json (yet): fall back to the steps logged so far
            steps, dag_edges = list(dict.fromkeys([*statuses, *(s for e in edges for s in e)])), edges
        tags = parent.data.tags
        return PipelineRun(
            run=self._summary(parent, names),
            pipeline=tags.get("basir.pipeline", ""),
            workflow_id=tags.get("basir.workflow_id"),
            status=tags.get("basir.status") or parent.info.status,
            dag=layout(steps, dag_edges, statuses),
        )

    def _dag_json(self, run_id: str) -> Any:
        if run_id not in self._dags:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    path = self.client.download_artifacts(run_id, "dag.json", tmp)
                    self._dags[run_id] = json.loads(Path(path).read_text())
            except Exception:
                return None  # not logged yet; try again next time
        return self._dags[run_id]

    def _summary(self, run: Run, names: dict[str, str]) -> RunSummary:
        info, metrics = run.info, run.data.metrics
        end = info.end_time or (datetime.now(UTC).timestamp() * 1000 if info.status == "RUNNING" else None)
        url = f"{self.ui_url}/#/experiments/{info.experiment_id}/runs/{info.run_id}" if self.ui_url else None
        return RunSummary(
            run_id=info.run_id,
            experiment=names.get(info.experiment_id, info.experiment_id),
            name=info.run_name or info.run_id[:8],
            verdict=_verdict(run),
            started=datetime.fromtimestamp(info.start_time / 1000, UTC) if info.start_time else None,
            duration_s=(end - info.start_time) / 1000 if end and info.start_time else None,
            metrics={k: metrics[k] for k in _key_metrics(metrics)},
            url=url,
        )


def _verdict(run: Run) -> str:
    """running, pass / fail (from a logged verdict metric), else done / failed from the run status."""
    if run.info.status == "RUNNING":
        return "running"
    verdict = next((run.data.metrics[k] for k in PASS_METRICS if k in run.data.metrics), None)
    if verdict is not None:
        return "pass" if verdict >= 1.0 else "fail"
    return "done" if run.info.status == "FINISHED" else "failed"


def _key_metrics(metrics: dict[str, float]) -> list[str]:
    """Up to MAX_METRICS names: the known key metrics first, else whatever the run logged."""
    known = [k for k in KEY_METRICS if k in metrics]
    return (known or sorted(k for k in metrics if k not in PASS_METRICS))[:MAX_METRICS]


def _short(error: Exception) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0][:200]
