"""Tests for basir/pipelines/dag.py."""

from __future__ import annotations

import pytest

from basir.pipelines.dag import Dag, Step


def _dag() -> Dag:
    return Dag(
        (
            Step("check_env", "check_env"),
            Step("prepare_dataset:fr1/desk", "prepare_dataset", "fr1/desk", ("check_env",)),
            Step("gate", "gate", None, ("prepare_dataset:fr1/desk",)),
        )
    )


def test_edges_and_json() -> None:
    dag = _dag()
    assert dag.edges == [("check_env", "prepare_dataset:fr1/desk"), ("prepare_dataset:fr1/desk", "gate")]
    assert dag.to_json() == {
        "nodes": [
            {"id": "check_env", "activity": "check_env", "dataset": None},
            {"id": "prepare_dataset:fr1/desk", "activity": "prepare_dataset", "dataset": "fr1/desk"},
            {"id": "gate", "activity": "gate", "dataset": None},
        ],
        "edges": [
            {"source": "check_env", "target": "prepare_dataset:fr1/desk"},
            {"source": "prepare_dataset:fr1/desk", "target": "gate"},
        ],
    }


def test_mermaid() -> None:
    assert _dag().to_mermaid() == (
        "flowchart LR\n"
        '    check_env["check_env"]\n'
        '    prepare_dataset_fr1_desk["prepare_dataset fr1/desk"]\n'
        '    gate["gate"]\n'
        "    check_env --> prepare_dataset_fr1_desk\n"
        "    prepare_dataset_fr1_desk --> gate\n"
    )


@pytest.mark.parametrize(
    "steps",
    [
        (Step("a", "x"), Step("a", "x")),  # duplicate
        (Step("b", "x", None, ("a",)), Step("a", "x")),  # dependency comes later
        (Step("a,b", "x"),),  # comma breaks basir.depends_on
    ],
)
def test_invalid_dags_are_rejected(steps: tuple[Step, ...]) -> None:
    with pytest.raises(ValueError):
        Dag(steps)
