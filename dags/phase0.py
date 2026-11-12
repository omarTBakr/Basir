"""Phase 0 on Airflow: prepare, replay and fuse the TUM sequences, then gate on the targets.

    open_pipeline_run (setup) -> check_env -> prepare_dataset[each dataset]
        prepare_dataset -> replay (first dataset) and pointcloud[each dataset] -> gate
    -> close_pipeline_run (teardown: runs whatever happened, marks the MLflow parent run)

Every step runs in Basir's own Python environment through `basir.pipelines.airflow_task`;
the shape comes from `basir.pipelines.phase0.phase0_dag`, which is also logged to MLflow as
dag.json / dag.mmd. Parsing this file fails if the two ever disagree.

Trigger with overrides from the Airflow UI ("Trigger DAG" form), `scripts/run_pipeline.py`
or the REST API (`conf` = any of the params below).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from basir.pipelines.airflow_task import StepFailed, run_step
from basir.pipelines.phase0 import (
    CLOSE_TASK,
    DAG_ID,
    DEFAULT_DATASETS,
    OPEN_TASK,
    PARAM_SCHEMA,
    PIPELINE,
    param_defaults,
    phase0_dag,
)

# Every step opens an MLflow run, and a cold start (imports, a slow disk) can take minutes.
BOOKKEEPING: dict[str, Any] = {
    "retries": 3,
    "retry_delay": timedelta(seconds=10),
    "execution_timeout": timedelta(minutes=10),
}


def _run(task_id: str, payload: dict[str, Any]) -> Any:
    try:
        return run_step(PIPELINE, task_id, payload)
    except StepFailed as e:  # bad input, missing data or a failed gate: retrying will not help
        raise AirflowFailException(str(e)) from e


def _payload(parent_run_id: str | None, **extra: Any) -> dict[str, Any]:
    params = get_current_context()["params"]
    return {"params": {name: params[name] for name in PARAM_SCHEMA}, "parent_run_id": parent_run_id, **extra}


@dag(
    dag_id=DAG_ID,
    schedule=None,
    catchup=False,
    max_active_runs=1,  # runs write the same outputs/<dataset>.ply files
    params={name: Param(param_defaults()[name], **schema) for name, schema in PARAM_SCHEMA.items()},
    default_args={"owner": "basir", "retry_delay": timedelta(seconds=10)},
    tags=["basir", "phase0"],
    doc_md=__doc__,
)
def phase0() -> None:
    @task(task_id=OPEN_TASK, **BOOKKEEPING)
    def open_pipeline_run() -> list[str]:
        """Open the MLflow parent run (tags basir.pipeline / basir.workflow_id = this run id) with the DAG attached.

        Returns the datasets (what the per-dataset tasks map over: Airflow maps only over a
        task's return value) and pushes the parent run's ID as the XCom `parent_run_id`.
        """
        context = get_current_context()
        opened = _run(OPEN_TASK, _payload(None, workflow_id=context["run_id"]))
        # "" when MLflow tracking is off: Airflow stores no XCom for None, and downstream tasks need one
        context["ti"].xcom_push(key="parent_run_id", value=opened["parent_run_id"] or "")
        return list(opened["datasets"])

    @task(retries=1, execution_timeout=timedelta(minutes=10))
    def check_env(parent_run_id: str | None) -> dict[str, str]:
        return _run("check_env", _payload(parent_run_id))

    @task(
        retries=2,
        retry_delay=timedelta(seconds=10),
        execution_timeout=timedelta(hours=2),
        map_index_template="{{ dataset }}",
    )
    def prepare_dataset(dataset: str, parent_run_id: str | None) -> dict[str, Any]:
        get_current_context()["dataset"] = dataset  # type: ignore[typeddict-unknown-key]  # names the mapped instance
        return _run("prepare_dataset", _payload(parent_run_id, dataset=dataset))

    @task(retries=1, execution_timeout=timedelta(minutes=30))
    def replay(prepared: list[dict[str, Any]], parent_run_id: str | None) -> dict[str, Any]:
        return _run("replay", _payload(parent_run_id, prepared=list(prepared)))

    @task(retries=1, execution_timeout=timedelta(hours=1), map_index_template="{{ dataset }}")
    def pointcloud(prepared: dict[str, Any], parent_run_id: str | None) -> dict[str, Any]:
        get_current_context()["dataset"] = prepared["dataset"]  # type: ignore[typeddict-unknown-key]
        return _run("pointcloud", _payload(parent_run_id, prepared=prepared))

    @task(retries=0, execution_timeout=timedelta(minutes=10))  # deterministic: a retry gives the same verdict
    def gate(clouds: list[dict[str, Any]], parent_run_id: str | None) -> list[dict[str, Any]]:
        return _run("gate", _payload(parent_run_id, clouds=list(clouds)))

    @task(task_id=CLOSE_TASK, **BOOKKEEPING)
    def close_pipeline_run(parent_run_id: str | None) -> None:
        """Mark the parent run succeeded (the gate returned) or failed (anything else), then end it."""
        succeeded = get_current_context()["ti"].xcom_pull(task_ids="gate") is not None
        _run(CLOSE_TASK, _payload(parent_run_id, succeeded=succeeded))

    opened = open_pipeline_run()
    parent = opened["parent_run_id"]  # the XCom pushed under that key
    env = check_env(parent)
    prepared = prepare_dataset.partial(parent_run_id=parent).expand(dataset=opened)
    env >> prepared
    replayed = replay(prepared, parent)
    clouds = pointcloud.partial(parent_run_id=parent).expand(prepared=prepared)
    verdict = gate(clouds, parent)
    replayed >> verdict
    closed = close_pipeline_run(parent)
    [env, prepared, replayed, clouds, verdict] >> closed
    closed.as_teardown(setups=opened)


def step_edges(airflow_dag: Any) -> set[tuple[str, str]]:
    """The DAG's (upstream, downstream) task pairs, without the bookkeeping tasks around the steps."""
    bookkeeping = {OPEN_TASK, CLOSE_TASK}
    return {
        (upstream, t.task_id)
        for t in airflow_dag.tasks
        for upstream in t.upstream_task_ids
        if upstream not in bookkeeping and t.task_id not in bookkeeping
    }


phase0_airflow_dag = phase0()

if step_edges(phase0_airflow_dag) != phase0_dag(list(DEFAULT_DATASETS)).task_edges:
    raise RuntimeError(
        "dags/phase0.py and basir.pipelines.phase0.phase0_dag disagree: "
        f"{sorted(step_edges(phase0_airflow_dag))} != {sorted(phase0_dag(list(DEFAULT_DATASETS)).task_edges)}"
    )
