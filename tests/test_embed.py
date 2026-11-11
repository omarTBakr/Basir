"""Tests for basir/api/embed.py: the frameable proxy forwards everything and drops only frame-blocking headers."""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from fastapi.testclient import TestClient
import pytest

from basir.api.embed import create_proxy_app


@pytest.fixture
def upstream() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.startswith("/old"):
                self.send_response(302)
                self.send_header("Location", f"http://{self.headers['Host']}/new")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = f"path={self.path} host={self.headers['Host']}".encode()
            self.send_response(200)
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
            self.send_header("Set-Cookie", "a=1")
            self.send_header("Set-Cookie", "b=2")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_strips_frame_headers_and_keeps_the_rest(upstream: str) -> None:
    with TestClient(create_proxy_app(upstream)) as client:
        response = client.get("/_app/x.js?v=1")
        assert response.status_code == 200
        assert response.text == f"path=/_app/x.js?v=1 host={upstream.removeprefix('http://')}"
        assert "x-frame-options" not in response.headers and "content-security-policy" not in response.headers
        assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]

        redirect = client.get("/old", follow_redirects=False)
        assert redirect.status_code == 302 and redirect.headers["location"] == "http://testserver/new"


def test_unreachable_upstream_is_a_502() -> None:
    with TestClient(create_proxy_app("http://127.0.0.1:9")) as client:
        assert client.get("/").status_code == 502
