"""A small client for Airflow 3's REST API (`/api/v2`): trigger DAG runs, read their state, XComs and logs.

Airflow 3 authenticates the API with a JWT: `POST /auth/token` with the user's name and password
returns `access_token`, sent as `Authorization: Bearer`. The token is fetched lazily and again
after a 401. Used by `scripts/run_pipeline.py`; the Console's `basir.api.airflow_view` shares
the URL helpers and credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from utilities.config import ENV_DIR, PROJECT_ROOT, Settings
from utilities.utility import parse_env_file

STACK_ENV = PROJECT_ROOT / ENV_DIR / "stack.env"
TOKEN_PATH = "/auth/token"
API = "/api/v2"
FINISHED = {"success", "failed"}


class AirflowError(RuntimeError):
    """Airflow answered with an error, or did not answer."""


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def credentials(settings: Settings, stack_env: Path = STACK_ENV) -> Credentials:
    """AIRFLOW_USERNAME / AIRFLOW_PASSWORD, the password defaulting to the stack's generated admin password."""
    password = settings.airflow_password or parse_env_file(stack_env).get("AIRFLOW_ADMIN_PASSWORD", "")
    return Credentials(settings.airflow_username, password)


def run_url(airflow_url: str, dag_id: str, run_id: str) -> str:
    """The run's page in the Airflow UI."""
    return f"{airflow_url.rstrip('/')}/dags/{quote(dag_id, safe='')}/runs/{quote(run_id, safe='')}"


def task_url(airflow_url: str, dag_id: str, run_id: str, task_id: str, map_index: int = -1) -> str:
    """A task instance's page (its logs) in the Airflow UI."""
    url = f"{run_url(airflow_url, dag_id, run_id)}/tasks/{quote(task_id, safe='')}"
    return f"{url}/mapped/{map_index}" if map_index >= 0 else url


def error_detail(response: httpx.Response) -> str:
    """Airflow's `detail` message, else the start of the body."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    text = detail if isinstance(detail, str) else json.dumps(detail) if detail else response.text
    return f"HTTP {response.status_code}: {text.strip()[:300]}"


class AirflowClient:
    """Blocking client for one Airflow API server."""

    def __init__(
        self, base_url: str, creds: Credentials, timeout: float = 10.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._creds = creds
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self._token: str | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AirflowClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _login(self) -> str:
        try:
            response = self._http.post(
                TOKEN_PATH, json={"username": self._creds.username, "password": self._creds.password}
            )
        except httpx.HTTPError as e:
            raise AirflowError(f"cannot reach Airflow at {self.base_url}: {type(e).__name__}: {e}") from e
        if response.status_code >= 400:
            raise AirflowError(f"Airflow login as '{self._creds.username}' failed: {error_detail(response)}")
        return str(response.json()["access_token"])

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """`method` `/api/v2/<path>` with a bearer token; the decoded JSON body (None if empty)."""
        for attempt in range(2):
            if self._token is None:
                self._token = self._login()
            try:
                response = self._http.request(
                    method, f"{API}/{path.lstrip('/')}", headers={"Authorization": f"Bearer {self._token}"}, **kwargs
                )
            except httpx.HTTPError as e:
                raise AirflowError(f"cannot reach Airflow at {self.base_url}: {type(e).__name__}: {e}") from e
            if response.status_code == 401 and attempt == 0:
                self._token = None  # expired: log in again once
                continue
            if response.status_code >= 400:
                raise AirflowError(f"{method} {path}: {error_detail(response)}")
            return response.json() if response.content else None
        raise AssertionError("unreachable")

    # -- DAG runs ------------------------------------------------------------

    def unpause(self, dag_id: str) -> None:
        self.request("PATCH", f"dags/{quote(dag_id, safe='')}", json={"is_paused": False})

    def trigger(self, dag_id: str, conf: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
        """Start a run now with `conf` overriding the DAG's params; returns the new DAG run."""
        body: dict[str, Any] = {"logical_date": None, "conf": conf}
        if run_id:
            body["dag_run_id"] = run_id
        return dict(self.request("POST", f"dags/{quote(dag_id, safe='')}/dagRuns", json=body))

    def dag_run(self, dag_id: str, run_id: str) -> dict[str, Any]:
        return dict(self.request("GET", f"dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}"))

    def task_instances(self, dag_id: str, run_id: str) -> list[dict[str, Any]]:
        path = f"dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}/taskInstances"
        return list(self.request("GET", path, params={"limit": 1000})["task_instances"])

    def xcom(self, dag_id: str, run_id: str, task_id: str, key: str = "return_value", map_index: int = -1) -> Any:
        """A task's XCom value (decoded from JSON), or None if it has not pushed one."""
        path = (
            f"dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}/taskInstances/"
            f"{quote(task_id, safe='')}/xcomEntries/{quote(key, safe='')}"
        )
        try:
            entry = self.request("GET", path, params={"map_index": map_index, "deserialize": True})
        except AirflowError as e:
            if "HTTP 404" in str(e):
                return None
            raise
        value = entry.get("value")
        if isinstance(value, str):  # some versions return the value stringified
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    def task_log(self, dag_id: str, run_id: str, task_id: str, try_number: int, map_index: int = -1) -> str:
        """A task attempt's log as plain text."""
        path = (
            f"dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}/taskInstances/"
            f"{quote(task_id, safe='')}/logs/{try_number}"
        )
        return log_text(self.request("GET", path, params={"map_index": map_index, "full_content": True}))


def log_text(body: Any) -> str:
    """Flatten a `/logs/` answer (structured events or plain text) into lines."""
    content = body.get("content") if isinstance(body, dict) else body
    if isinstance(content, str):
        return content
    lines = []
    for item in content or []:
        if isinstance(item, dict):
            lines.append(str(item.get("event", "")))
        elif isinstance(item, list | tuple):  # older shape: [[host, text], ...]
            lines.append(str(item[-1]))
        else:
            lines.append(str(item))
    return "\n".join(lines)
