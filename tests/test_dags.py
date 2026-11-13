"""Tests for dags/: every DAG imports cleanly and Phase 0 has the shape logged to MLflow as dag.json.

Airflow is not installed in the project venv (it pins older versions of Basir's dependencies),
so these tests skip there. Run them in the Airflow image (which has pytest):

    BASIR_DATA_DIR=$PWD docker compose -f Docker/docker-compose.yml --env-file env/stack.env \\
        run --rm --no-deps -v "$PWD/tests:/opt/basir/tests:ro" airflow-scheduler \\
        python -m pytest --noconftest -p no:cacheprovider -o addopts= /opt/basir/tests/test_dags.py

`--noconftest`: tests/conftest.py imports the Console, whose dependencies live only in Basir's
environment. Parsing needs no database, so the stack does not have to be up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="Airflow is only installed in the Airflow image (see the module docstring)")

from basir.pipelines.phase0 import CLOSE_TASK, DAG_ID, DEFAULT_DATASETS, OPEN_TASK, PARAM_SCHEMA, phase0_dag

DAGS = Path(__file__).resolve().parent.parent / "dags"


@pytest.fixture(scope="module")
def dagbag():  # type: ignore[no-untyped-def]
    from airflow.dag_processing.dagbag import DagBag

    return DagBag(dag_folder=str(DAGS))  # examples are off: AIRFLOW__CORE__LOAD_EXAMPLES=false


def test_every_dag_imports(dagbag) -> None:  # type: ignore[no-untyped-def]
    assert dagbag.import_errors == {}
    assert DAG_ID in dagbag.dag_ids


def test_phase0_tasks_and_edges_match_dag_json(dagbag) -> None:  # type: ignore[no-untyped-def]
    dag = dagbag.dags[DAG_ID]  # not get_dag(), which asks the metadata database
    steps = {s.activity for s in phase0_dag(list(DEFAULT_DATASETS)).steps}
    assert set(dag.task_ids) == steps | {OPEN_TASK, CLOSE_TASK}

    edges = {(u, t.task_id) for t in dag.tasks for u in t.upstream_task_ids}
    step_edges = {(a, b) for a, b in edges if OPEN_TASK not in (a, b) and CLOSE_TASK not in (a, b)}
    assert step_edges == phase0_dag(list(DEFAULT_DATASETS)).task_edges

    # the parent run is opened first and closed last, whatever happened in between
    assert all((OPEN_TASK, s) in edges for s in ("check_env", "prepare_dataset", "pointcloud", "gate"))
    assert {u for u, t in edges if t == CLOSE_TASK} == steps | {OPEN_TASK}
    assert dag.get_task(OPEN_TASK).is_setup and dag.get_task(CLOSE_TASK).is_teardown


def test_phase0_params_retries_and_mapping(dagbag) -> None:  # type: ignore[no-untyped-def]
    dag = dagbag.dags[DAG_ID]  # not get_dag(), which asks the metadata database
    assert set(dag.params) == set(PARAM_SCHEMA)
    assert dag.params["datasets"] == list(DEFAULT_DATASETS)
    assert dag.schedule is None or str(dag.timetable.summary) in ("None", "")
    assert dag.max_active_runs == 1

    retries = {t.task_id: t.retries for t in dag.tasks}
    assert retries[OPEN_TASK] == retries[CLOSE_TASK] == 3 and retries["gate"] == 0
    assert all(t.execution_timeout is not None and t.execution_timeout.total_seconds() >= 600 for t in dag.tasks)
    from airflow.sdk.definitions.mappedoperator import MappedOperator

    mapped = {t.task_id for t in dag.tasks if isinstance(t, MappedOperator)}
    assert mapped == {"prepare_dataset", "pointcloud"}
