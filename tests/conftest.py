"""Shared fixtures: environment isolation and a local HTTP file server."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import pytest

from utilities.config import ENV_PREFIX, Settings, get_settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Remove every env var Settings reads (plain and BASIR_-prefixed) and reset the cache."""
    fields = {name.lower() for name in Settings.model_fields}
    for key in list(os.environ):
        k = key.lower()
        if k in fields or k == "env_file" or (k.startswith(ENV_PREFIX) and k[len(ENV_PREFIX):] in fields):
            monkeypatch.delenv(key, raising=False)
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
                start = int(range_header[len("bytes="):].split("-", 1)[0])
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
