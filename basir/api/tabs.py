"""The Console's tabs: one registry that the sidebar, the routes and the "coming later" pages read.

Adding a feature = a router module with its page at `/<key>`, a template, and one `Tab` here.
A tab with a `phase` is not built yet: the sidebar greys it out and `/<key>` explains what it will do.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUPS = ("Operate", "Observe", "Coming later")


@dataclass(frozen=True)
class Tab:
    key: str
    title: str
    icon: str  # one character, shown in the sidebar
    group: str
    summary: str = ""
    phase: int | None = None  # set while the feature is planned but not built

    @property
    def path(self) -> str:
        return f"/{self.key}"

    @property
    def enabled(self) -> bool:
        return self.phase is None


TABS = (
    Tab("live", "Live", "◉", "Operate", "Colour and depth from a phone, side by side"),
    Tab("map", "3D map", "△", "Operate", "The live map fused from keyframes, and saved runs"),
    Tab("devices", "Devices", "▦", "Operate", "Connected phones, pairing and tokens"),
    Tab("dashboards", "Dashboards", "▤", "Observe", "Grafana: server, host, GPU, Postgres, Temporal"),
    Tab("experiments", "Experiments", "⚖", "Observe", "MLflow runs"),
    Tab("pipelines", "Pipelines", "⇄", "Observe", "Temporal workflows and their DAGs"),
    Tab("ask", "Ask", "?", "Coming later", "Where did I last see my keys? Ask by text or photo", phase=1),
    Tab("objects", "Objects", "▣", "Coming later", "Every object seen, with masks and last-seen time", phase=2),
    Tab("guide", "Guide me", "➤", "Coming later", "Bearing and distance back to an object", phase=4),
    Tab("robot", "Robot", "⚙", "Coming later", "ROS 2 bridge and Nav2 paths to remembered objects", phase=7),
)


def tab(key: str) -> Tab | None:
    return next((t for t in TABS if t.key == key), None)


def grouped() -> list[tuple[str, list[Tab]]]:
    """(group, tabs) in sidebar order."""
    return [(group, [t for t in TABS if t.group == group]) for group in GROUPS]
