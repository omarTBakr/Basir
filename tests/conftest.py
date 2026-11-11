"""Shared fixtures: environment isolation, a local HTTP file server and a Basir server with a fake GPU worker."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import threading
import time

from fastapi.testclient import TestClient
import pytest
import uvicorn

from basir.api.app import create_app
from utilities.config import ENV_PREFIX, Settings, get_settings

UNREACHABLE = "127.0.0.1:1"  # nothing listens on port 1


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Remove every env var Settings reads (plain and BASIR_-prefixed) and reset the cache.

    ENV_FILE points at an empty source, so the developer's real env/.env (e.g. an MLflow
    server URI) never leaks into tests. Tests of the default lookup unset it themselves.
    """
    fields = {name.lower() for name in Settings.model_fields}
    for key in list(os.environ):
        k = key.lower()
        if k in fields or k == "env_file" or (k.startswith(ENV_PREFIX) and k[len(ENV_PREFIX) :] in fields):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ENV_FILE", os.devnull)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@dataclass
class FileServer:
    """In-memory files served over HTTP, with Range support and fault injection."""

    files: dict[str, bytes] = field(default_factory=dict)
    truncate: dict[str, int] = field(default_factory=dict)  # path -> bytes sent before dropping
    requests: list[dict[str, str | None]] = field(default_factory=list)
    base_url: str = ""

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"


@pytest.fixture
def file_server() -> Iterator[FileServer]:
    state = FileServer()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _empty(self, status: int, **headers: str) -> None:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.lstrip("/")
            range_header = self.headers.get("Range")
            state.requests.append({"path": path, "range": range_header})
            data = state.files.get(path)
            if data is None:
                return self._empty(404)

            start, status = 0, 200
            if range_header and range_header.startswith("bytes="):
                start = int(range_header[len("bytes=") :].split("-", 1)[0])
                if start >= len(data):
                    return self._empty(416, Content_Range=f"bytes */{len(data)}")
                status = 206

            body = data[start:]
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.end_headers()

            cut = state.truncate.get(path)
            if cut is None:
                self.wfile.write(body)
                return
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Basir server with a fake GPU worker
# ---------------------------------------------------------------------------


class FakeBackend:
    """`InferenceBackend` without a process: records batch sizes; `gate` holds batches until set."""

    device = "fake"

    def __init__(self, seconds: float = 0.002) -> None:
        self.seconds = seconds
        self.batches: list[int] = []
        self.entered = threading.Event()
        self.gate: threading.Event | None = None
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    async def infer(self, jpegs: list[bytes]) -> float:
        self.batches.append(len(jpegs))
        self.entered.set()
        if self.gate is not None:
            await asyncio.to_thread(self.gate.wait, 10)
        return self.seconds

    def stop(self) -> None:
        self.stopped = True


def server_settings(data_dir: Path, **overrides: object) -> Settings:
    """Settings with every external service unreachable and MLflow in `data_dir`."""
    values: dict[str, object] = {
        "data_dir": str(data_dir),
        "temporal_address": UNREACHABLE,
        "grafana_url": f"http://{UNREACHABLE}",
        "prometheus_url": f"http://{UNREACHABLE}",
        "batch_max_wait_ms": 5.0,
        **overrides,
    }
    return Settings(_env_file=data_dir / "no.env", **values)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(tmp_path: Path, clean_env: pytest.MonkeyPatch, backend: FakeBackend) -> Iterator[TestClient]:
    """The app (lifespan running) on a TestClient, with a fake worker and no external services."""
    with TestClient(create_app(server_settings(tmp_path), backend)) as test_client:
        yield test_client


@pytest.fixture
def live_server(tmp_path: Path, clean_env: pytest.MonkeyPatch, backend: FakeBackend) -> Iterator[str]:
    """The app served by uvicorn on a free port in a thread; yields its ws:// base URL."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(create_app(server_settings(tmp_path), backend), port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.05)
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
