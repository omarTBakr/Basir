"""Tests for basir/api/ui.py, app.py and grafana.py: every page renders, with or without the services."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
import pytest

from basir.api.app import create_app
from basir.api.grafana import DASHBOARDS, PANELS, dashboard_url, panel_url
from tests.conftest import FakeBackend, server_settings


@pytest.mark.parametrize("path", ["/live", "/pipelines", "/experiments", "/dashboards"])
def test_pages_render(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert "htmx.org@2.0.4" in response.text and 'aria-current="page"' in response.text
    assert response.headers["x-request-id"]


def test_home_redirects_to_live(client: TestClient) -> None:
    assert client.get("/", follow_redirects=False).headers["location"] == "/live"


def test_live_partial_without_phones(client: TestClient) -> None:
    text = client.get("/partials/live").text
    assert "No phones connected" in text and "fake" in text


def test_unreachable_services_are_shown_not_raised(client: TestClient) -> None:
    pipelines = client.get("/partials/pipelines")
    assert pipelines.status_code == 200 and "Airflow not reachable" in pipelines.text
    dashboards = client.get("/dashboards").text
    assert "Grafana not reachable" in dashboards and dashboards.count('badge failed">down') == 3


def test_experiments_partial_summarises_the_store(client: TestClient, tmp_path: Path) -> None:
    mlflow = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.create_experiment("phase0-pointcloud", artifact_location=(tmp_path / "a").as_uri())
    run = mlflow.create_run(experiment, run_name="fr1_desk")
    mlflow.log_metric(run.info.run_id, "consistency_median_cm", 4.2)
    mlflow.log_metric(run.info.run_id, "passes", 0.0)
    mlflow.set_terminated(run.info.run_id, "FINISHED")
    text = client.get("/partials/experiments").text
    assert "phase0-pointcloud" in text and "fr1_desk" in text
    assert 'badge failed">fail' in text and "consistency_median_cm" in text and "4.2" in text
    assert "Nothing running" in text


def test_mlflow_server_down_is_shown(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    settings = server_settings(tmp_path, mlflow_tracking_uri="http://127.0.0.1:1")
    with TestClient(create_app(settings, FakeBackend())) as client:
        assert "MLflow not reachable" in client.get("/partials/experiments").text


def test_lifespan_starts_and_stops_the_worker(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    backend = FakeBackend()
    with TestClient(create_app(server_settings(tmp_path), backend)):
        assert backend.started and not backend.stopped
    assert backend.stopped


def test_grafana_urls() -> None:
    assert {d.uid for d in DASHBOARDS} == {
        "basir-host",
        "basir-gpu",
        "basir-postgres",
        "basir-airflow",
        "basir-server",
    }
    assert {p.dashboard_uid for p in PANELS} <= {d.uid for d in DASHBOARDS}
    panel = PANELS[0]
    url = panel_url("http://g:3000/", panel, "dark")
    assert url.startswith(
        f"http://g:3000/d-solo/{panel.dashboard_uid}/?orgId=1&panelId={panel.panel_id}&kiosk&theme=dark"
    )
    assert dashboard_url("http://g:3000", "basir-gpu", "light").startswith("http://g:3000/d/basir-gpu/?orgId=1&kiosk")
