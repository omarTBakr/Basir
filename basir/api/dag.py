"""Pipeline DAGs for the UI: parse `dag.json`, normalise step statuses, lay the graph out left to right.

The layout is plain layering (each step one column right of its deepest dependency), drawn
as SVG by the template, so the page needs no JavaScript to show a pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NODE_W, NODE_H = 150, 36
GAP_X, GAP_Y = 48, 14

_STATUS_WORDS = {
    "ok": {"success", "succeeded", "passed", "pass", "finished", "completed", "complete", "ok", "done"},
    "failed": {"failed", "fail", "error", "killed", "cancelled", "canceled", "timed_out", "terminated"},
    "running": {"running", "started", "in_progress"},
    "pending": {"pending", "scheduled", "queued", "waiting"},
    "skipped": {"skipped", "cached"},
}


def status_class(raw: str | None) -> str:
    """Bucket a free-form status (`basir.status`, MLflow, Temporal) into ok/failed/running/pending/skipped/unknown."""
    word = (raw or "pending").strip().lower().replace(" ", "_").replace("-", "_")
    return next((bucket for bucket, words in _STATUS_WORDS.items() if word in words), "unknown")


@dataclass(frozen=True)
class DagNode:
    name: str
    status: str  # the raw status, shown as text
    x: int
    y: int

    @property
    def css(self) -> str:
        return status_class(self.status)


@dataclass(frozen=True)
class Dag:
    nodes: list[DagNode]
    edges: list[tuple[str, str]]
    width: int
    height: int

    @property
    def paths(self) -> list[str]:
        """SVG path data for each edge, from the right edge of a step to the left edge of its dependant."""
        where = {n.name: n for n in self.nodes}
        out = []
        for source, target in self.edges:
            a, b = where[source], where[target]
            x1, y1, x2, y2 = a.x + NODE_W, a.y + NODE_H // 2, b.x, b.y + NODE_H // 2
            mid = (x1 + x2) // 2
            out.append(f"M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}")
        return out


def parse_dag(raw: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """Node names and (source, target) edges from `dag.json`.

    Accepts nodes as strings or objects with `id`/`name`, and edges as pairs or objects
    with `source`/`target` (or `from`/`to`).
    """
    if not isinstance(raw, dict):
        return [], []
    names = [_node_name(n) for n in raw.get("nodes", [])]
    edges = [e for e in (_edge(e) for e in raw.get("edges", [])) if e is not None]
    for source, target in edges:  # an edge may name a step the node list forgot
        names += [n for n in (source, target) if n not in names]
    return [n for n in dict.fromkeys(names) if n], edges


def layout(names: list[str], edges: list[tuple[str, str]], statuses: dict[str, str]) -> Dag:
    """Place each step in the column after its deepest dependency (cycles are cut, not looped)."""
    level = dict.fromkeys(names, 0)
    edges = [(s, t) for s, t in edges if s in level and t in level and s != t]
    for _ in range(len(names)):
        changed = False
        for source, target in edges:
            if level[target] < level[source] + 1 < len(names):
                level[target] = level[source] + 1
                changed = True
        if not changed:
            break
    rows: dict[int, int] = {}
    nodes = []
    for name in names:
        column = level[name]
        row = rows.get(column, 0)
        rows[column] = row + 1
        nodes.append(DagNode(name, statuses.get(name, "pending"), column * (NODE_W + GAP_X), row * (NODE_H + GAP_Y)))
    columns = max(level.values(), default=-1) + 1
    width = max(columns * (NODE_W + GAP_X) - GAP_X, 0)
    height = max(max(rows.values(), default=0) * (NODE_H + GAP_Y) - GAP_Y, 0)
    return Dag(nodes, edges, width, height)


def _node_name(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("id") or node.get("name") or "")
    return str(node)


def _edge(edge: Any) -> tuple[str, str] | None:
    if isinstance(edge, list | tuple) and len(edge) == 2:
        return str(edge[0]), str(edge[1])
    if isinstance(edge, dict):
        source, target = edge.get("source", edge.get("from")), edge.get("target", edge.get("to"))
        if source is not None and target is not None:
            return str(source), str(target)
    return None
