"""Which Grafana dashboards and panels the Dashboards page embeds, and their embed URLs.

Edit `DASHBOARDS` and `PANELS` here: this is the one place that knows dashboard uids and
panel ids. Every dashboard is also offered whole (`/d/<uid>?kiosk`), which works even when
a panel id below is out of date.
"""

from __future__ import annotations

from dataclasses import dataclass

TIME_RANGE = "from=now-15m&to=now&refresh=10s"


@dataclass(frozen=True)
class Dashboard:
    uid: str
    title: str


@dataclass(frozen=True)
class Panel:
    dashboard_uid: str
    panel_id: int
    title: str


DASHBOARDS = (
    Dashboard("basir-server", "Basir server"),
    Dashboard("basir-host", "Host"),
    Dashboard("basir-gpu", "GPU"),
    Dashboard("basir-postgres", "Postgres"),
    Dashboard("basir-temporal", "Temporal"),
)

PANELS = (
    Panel("basir-server", 7, "Frames"),
    Panel("basir-server", 9, "Inference latency"),
    Panel("basir-server", 10, "Batch size"),
    Panel("basir-server", 8, "Queue depth and connections"),
    Panel("basir-gpu", 5, "GPU utilization"),
    Panel("basir-gpu", 6, "GPU memory"),
    Panel("basir-host", 5, "CPU by mode"),
    Panel("basir-host", 8, "Disk I/O"),
    Panel("basir-postgres", 5, "Postgres connections"),
    Panel("basir-temporal", 5, "Temporal workflow outcomes"),
)


def panel_url(grafana_url: str, panel: Panel, theme: str) -> str:
    """A single panel, chrome-free (`d-solo`), in the given light/dark theme."""
    base = grafana_url.rstrip("/")
    return f"{base}/d-solo/{panel.dashboard_uid}/?orgId=1&panelId={panel.panel_id}&kiosk&theme={theme}&{TIME_RANGE}"


def dashboard_url(grafana_url: str, uid: str, theme: str) -> str:
    """A whole dashboard in kiosk mode."""
    return f"{grafana_url.rstrip('/')}/d/{uid}/?orgId=1&kiosk&theme={theme}&{TIME_RANGE}"
