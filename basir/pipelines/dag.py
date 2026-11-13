"""A pipeline's steps and their dependencies, with the JSON and Mermaid forms logged to MLflow.

Pure Python with no I/O, so Temporal workflow code can build and walk it deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class Step:
    """One node: `name` is unique (e.g. "pointcloud:fr1/room"), `activity` is the Temporal activity it runs."""

    name: str
    activity: str
    dataset: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Dag:
    """Steps in a dependency-respecting order: every step comes after the steps it depends on."""

    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            if step.name in seen:
                raise ValueError(f"duplicate step '{step.name}'")
            if missing := [d for d in step.depends_on if d not in seen]:
                raise ValueError(f"step '{step.name}' depends on {missing}, which do not come before it")
            if "," in step.name:
                raise ValueError(f"step name '{step.name}' must not contain a comma")
            seen.add(step.name)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """(dependency, step) pairs."""
        return [(dep, step.name) for step in self.steps for dep in step.depends_on]

    def to_json(self) -> dict[str, Any]:
        """{"nodes": [{"id", "activity", "dataset"}], "edges": [{"source", "target"}]}."""
        return {
            "nodes": [{"id": s.name, "activity": s.activity, "dataset": s.dataset} for s in self.steps],
            "edges": [{"source": a, "target": b} for a, b in self.edges],
        }

    def to_mermaid(self) -> str:
        """A left-to-right Mermaid flowchart."""
        lines = ["flowchart LR"]
        lines += [f'    {_node_id(s.name)}["{s.activity}{f" {s.dataset}" if s.dataset else ""}"]' for s in self.steps]
        lines += [f"    {_node_id(a)} --> {_node_id(b)}" for a, b in self.edges]
        return "\n".join(lines) + "\n"


def _node_id(name: str) -> str:
    return re.sub(r"\W", "_", name)
