"""Security tests for the embed proxies: framing only by the Console, the upstream's own Host, no open
redirect, no path escape, only the methods the UIs need, Host and Origin checks, a body limit."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient
import httpx
import pytest

from basir.api import embed
from basir.api.embed import create_proxy_app, frame_ancestors

UPSTREAM = "http://127.0.0.1:5000"


class Recorder:
    """A fake upstream: records each request and answers with the headers a test asks for."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answer_headers: list[tuple[str, str]] = []
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, headers=self.answer_headers, stream=httpx.ByteStream(b"ok"))


@pytest.fixture
def upstream() -> Recorder:
    return Recorder()


@pytest.fixture
def proxy(upstream: Recorder) -> Iterator[TestClient]:
    app = create_proxy_app(UPSTREAM, console_port=8000, transport=httpx.MockTransport(upstream))
    with TestClient(app, base_url="http://127.0.0.1:8002") as client:
        yield client


def test_frame_headers_allow_only_the_console(proxy: TestClient, upstream: Recorder) -> None:
    upstream.answer_headers = [
        ("X-Frame-Options", "SAMEORIGIN"),
        ("Content-Security-Policy", "default-src 'self'; frame-ancestors 'self'; img-src *"),
    ]
    response = proxy.get("/")
    assert "x-frame-options" not in response.headers
    csp = response.headers["content-security-policy"]
    assert csp == "default-src 'self'; img-src *; " + frame_ancestors(8000)
    assert frame_ancestors(8000) == "frame-ancestors http://127.0.0.1:8000 http://localhost:8000"
    upstream.answer_headers = []
    assert proxy.get("/").headers["content-security-policy"] == frame_ancestors(8000)  # added when missing


def test_upstream_gets_its_own_host_and_nothing_host_like_from_the_client(
    proxy: TestClient, upstream: Recorder
) -> None:
    spoof = {
        "x-forwarded-host": "evil.example",
        "forwarded": "host=evil.example",
        "x-forwarded-for": "6.6.6.6",
        "x-real-ip": "6.6.6.6",
        "origin": "http://127.0.0.1:8002",
        "referer": "http://127.0.0.1:8002/x",
        "cookie": "mlflow=1",
    }
    proxy.post("/ajax-api/2.0/mlflow/runs/delete", json={"run_id": "r"}, headers=spoof)
    sent = upstream.requests[-1]
    assert sent.headers["host"] == "127.0.0.1:5000" and sent.url.netloc == b"127.0.0.1:5000"
    assert sent.headers["origin"] == UPSTREAM and sent.headers["referer"] == UPSTREAM + "/"
    for name in ("x-forwarded-host", "forwarded", "x-forwarded-for", "x-real-ip"):
        assert name not in sent.headers
    assert sent.headers["cookie"] == "mlflow=1" and sent.content == b'{"run_id":"r"}'
    proxy.get("/x")
    assert "origin" not in upstream.requests[-1].headers


@pytest.mark.parametrize(
    "path",
    ["//evil.example/x", "/../../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd", "/a/..//evil.example", "/@evil.example"],
)
def test_paths_never_leave_the_upstream(proxy: TestClient, upstream: Recorder, path: str) -> None:
    proxy.get(path)
    sent = upstream.requests[-1]
    assert sent.url.scheme == "http" and sent.url.host == "127.0.0.1" and sent.url.port == 5000


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (f"{UPSTREAM}/#/experiments/1", "/#/experiments/1"),
        (f"{UPSTREAM}?a=1", "/?a=1"),
        (f"{UPSTREAM}//evil.example/x", "/evil.example/x"),
        ("/new", "/new"),
        ("new", "new"),
        ("https://evil.example/login", "/"),
        ("//evil.example/x", "/"),
        ("///evil.example/x", "/evil.example/x"),
        ("/\\evil.example", "/"),
        ("\\\\evil.example", "/"),
        ("/\t/evil.example", "/"),
        ("http://127.0.0.1:5001/x", "/"),
        ("javascript:alert(1)", "/"),
        ("http:evil.example", "/"),
    ],
)
def test_redirects_stay_on_the_proxy(location: str, expected: str) -> None:
    assert embed._rewrite_location(location, UPSTREAM) == expected


@pytest.mark.parametrize(
    ("location", "expected"),
    [(f"{UPSTREAM}/#/experiments/1", "/#/experiments/1"), ("https://evil.example/login", "/"), ("//evil.example", "/")],
)
def test_redirects_through_the_proxy(proxy: TestClient, upstream: Recorder, location: str, expected: str) -> None:
    upstream.status, upstream.answer_headers = 302, [("Location", location)]
    response = proxy.get("/old", follow_redirects=False)
    assert response.status_code == 302 and response.headers["location"] == expected
    # A Location httpx cannot even parse (`javascript:...`) ends as a 502, never as a redirect.
    upstream.answer_headers = [("Location", "javascript:alert(1)")]
    assert "location" not in proxy.get("/old", follow_redirects=False).headers


def test_only_needed_methods(proxy: TestClient) -> None:
    for method in ("GET", "HEAD", "POST", "PATCH", "DELETE"):
        assert proxy.request(method, "/x").status_code == 200, method
    for method in ("PUT", "OPTIONS", "TRACE"):
        assert proxy.request(method, "/x").status_code == 405, method


def test_host_and_origin_checks(proxy: TestClient, upstream: Recorder) -> None:
    for host in ("evil.example", "evil.example:8002", "192.168.1.20:8002"):
        assert proxy.get("/", headers={"host": host}).status_code == 400, host  # DNS rebinding
    assert proxy.get("/", headers={"host": "localhost:8002"}).status_code == 200
    count = len(upstream.requests)
    assert proxy.post("/x", headers={"origin": "http://evil.example"}).status_code == 403
    assert proxy.post("/x", headers={"origin": "http://127.0.0.1:8000"}).status_code == 403  # the Console != proxy
    assert len(upstream.requests) == count
    assert proxy.post("/x", headers={"origin": "http://127.0.0.1:8002"}).status_code == 200


def test_body_limit(proxy: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "MAX_BODY_BYTES", 10)
    assert proxy.post("/x", content=b"x" * 11).status_code == 413
    assert proxy.post("/x", content=iter([b"x" * 6, b"x" * 6])).status_code == 413  # chunked, no length
    assert proxy.post("/x", content=b"x" * 10).status_code == 200
