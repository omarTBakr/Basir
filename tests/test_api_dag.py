"""Tests for basir/api/dag.py (dag.json parsing, status buckets, layered layout)."""

from __future__ import annotations

import pytest

from basir.api.dag import GAP_X, NODE_W, layout, parse_dag, status_class
from basir.pipelines.dag import Dag, Step


def test_parse_pipeline_format() -> None:
    raw = {
        "nodes": [{"id": "prepare", "activity": "a"}, {"id": "replay"}, {"id": "cloud"}],
        "edges": [{"source": "prepare", "target": "replay"}, {"source": "prepare", "target": "cloud"}],
    }
    assert parse_dag(raw) == (["prepare", "replay", "cloud"], [("prepare", "replay"), ("prepare", "cloud")])


def test_parse_reads_what_the_pipeline_writes() -> None:
    dag = Dag((Step("a", "x"), Step("b", "y", "fr1/desk", ("a",)), Step("c", "z", None, ("a", "b"))))
    assert parse_dag(dag.to_json()) == (["a", "b", "c"], dag.edges)


def test_parse_skips_malformed_entries() -> None:
    names, edges = parse_dag({"nodes": [{"id": "a"}, "junk", {}], "edges": [["a", "b"], {"source": "a"}, "junk"]})
    assert names == ["a"] and edges == []
    assert parse_dag(None) == ([], [])


@pytest.mark.parametrize(
    ("raw", "bucket"),
    [("succeeded", "ok"), ("FINISHED", "ok"), ("failed", "failed"), ("running", "running"), (None, "pending")],
)
def test_status_class(raw: str | None, bucket: str) -> None:
    assert status_class(raw) == bucket


def test_layout_puts_each_step_after_its_deepest_dependency() -> None:
    dag = layout(["a", "b", "c", "d"], [("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")], {"a": "succeeded"})
    x = {n.name: n.x for n in dag.nodes}
    step = NODE_W + GAP_X
    assert x == {"a": 0, "b": step, "c": step, "d": 2 * step}
    assert {n.name: n.css for n in dag.nodes} == {"a": "ok", "b": "pending", "c": "pending", "d": "pending"}
    assert len(dag.paths) == 4 and dag.width == 3 * step - GAP_X


def test_layout_survives_cycles_and_unknown_edges() -> None:
    dag = layout(["a", "b"], [("a", "b"), ("b", "a"), ("a", "zzz")], {})
    assert len(dag.nodes) == 2 and len(dag.edges) == 2
