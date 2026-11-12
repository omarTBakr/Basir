"""Security tests: the Console refuses foreign Hosts and cross-origin sockets and POSTs; the phone listener
accepts any IP as Host but no domain names and no cross-site requests; both get request IDs and metrics."""

from __future__ import annotations

from collections.abc import Iterator
import json

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from basir.api.ingest import create_ingest_app
from basir.api.metrics import HTTP_REQUESTS
from basir.api.middleware import ip_or_local_host, same_origin

EVIL = "http://evil.example"
PHONE_WS = "ws://192.168.1.20:8001"  # TestClient sockets ignore base_url, so give the host in full
SOCKETS = ("/ws/view/realme-14", "/ws/map/realme-14", "/ws/stream/realme-14")


@pytest.fixture
def phones(client: TestClient) -> Iterator[TestClient]:
    """The phone listener sharing `client`'s state, reached by IP as a phone would."""
    with TestClient(create_ingest_app(client.app), base_url="http://192.168.1.20:8001") as phone:  # type: ignore[arg-type]
        yield phone


def test_console_refuses_foreign_hosts(client: TestClient) -> None:
    assert client.get("/live").status_code == 200
    for host in ("127.0.0.1:8000", "localhost:8000"):
        assert client.get("/live", headers={"host": host}).status_code == 200
    for host in ("evil.example", "evil.example:8000", "192.168.1.20:8000", "127.0.0.1.nip.io:8000"):
        assert client.get("/live", headers={"host": host}).status_code == 400, host
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws/view/realme-14", headers={"host": "evil.example"}),
    ):
        pass


@pytest.mark.parametrize("path", SOCKETS)
def test_console_sockets_refuse_other_origins(client: TestClient, path: str) -> None:
    for origin in (EVIL, "null", "http://testserver.evil.example", "http://testserver:9999"):
        with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(path, headers={"origin": origin}):
            pass
        assert refused.value.code == 1008, origin


def test_console_sockets_accept_their_own_origin(client: TestClient) -> None:
    with client.websocket_connect("/ws/map/realme-14", headers={"origin": "http://testserver"}) as ws:
        assert json.loads(ws.receive_text())["type"] == "meta"


def test_console_posts_refuse_other_origins(client: TestClient) -> None:
    posts = [
        ("/api/devices/pair", {"device": "realme-14"}),
        ("/api/devices/realme-14/revoke", None),
        ("/api/devices/requests/nope/deny", None),
        ("/api/map/realme-14/reset", None),
        ("/api/map/realme-14/save", None),
    ]
    for path, form in posts:
        for headers in ({"origin": EVIL}, {"origin": "null"}, {"sec-fetch-site": "cross-site"}):
            assert client.post(path, data=form, headers=headers).status_code == 403, (path, headers)
    assert "realme-14" not in client.app.state.ingest.tokens.devices()  # type: ignore[attr-defined]
    # HTMX posts carry the page's own origin; non-browser clients send none. Both work.
    same = client.post("/api/devices/pair", data={"device": "a"}, headers={"origin": "http://testserver"})
    assert same.status_code == 200
    assert client.post("/api/devices/pair", data={"device": "b"}).status_code == 200
    fetch = {"sec-fetch-site": "same-origin"}
    assert client.post("/api/devices/b/revoke", headers=fetch).status_code == 200


def test_console_on_a_real_port_matches_origin_with_port(live_server: str) -> None:
    port = live_server.rsplit(":", 1)[1]
    with connect(f"{live_server}/ws/map/realme-14", origin=f"http://127.0.0.1:{port}") as ws:  # type: ignore[arg-type]
        assert json.loads(ws.recv())["type"] == "meta"
    for origin in (EVIL, f"http://localhost:{port}", "http://127.0.0.1:1"):
        with pytest.raises(InvalidStatus):
            connect(f"{live_server}/ws/stream/realme-14", origin=origin).close()  # type: ignore[arg-type]
    with connect(f"{live_server}/ws/stream/sim-1") as ws:  # no Origin: a local program (simulator), no token
        ws.send(b"\x00")
        assert "error" in json.loads(ws.recv())


def test_phone_listener_hosts(phones: TestClient) -> None:
    assert phones.get("/healthz").status_code == 200
    for host in ("192.168.1.20:8001", "10.42.0.1:8001", "[fe80::1]:8001", "localhost:8001", "laptop.local:8001"):
        assert phones.get("/healthz", headers={"host": host}).status_code == 200, host
    for host in ("evil.example", "evil.example:8001", "192.168.1.20.nip.io:8001", ""):
        assert phones.get("/healthz", headers={"host": host}).status_code == 400, host


def test_phone_listener_refuses_cross_site_pairing_and_sockets(phones: TestClient) -> None:
    ask = {"device": "realme-14", "label": "realme 14"}
    assert phones.post("/pair", json=ask, headers={"origin": EVIL}).status_code == 403
    assert phones.post("/pair", json=ask, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    own = phones.post("/pair", json=ask, headers={"origin": "https://192.168.1.20:8001"})
    assert own.status_code == 200 and own.json()["status"] == "pending"
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        phones.websocket_connect(f"{PHONE_WS}/ws/stream/realme-14", headers={"origin": EVIL}),
    ):
        pass
    assert refused.value.code == 1008 and refused.value.reason == "cross-origin request refused"


def test_phone_listener_has_request_ids_and_metrics(phones: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    before = HTTP_REQUESTS.labels("POST", "/pair", "200")._value.get()
    with caplog.at_level("INFO", logger="basir.ingest"):
        answer = phones.post("/pair", json={"device": "realme-15"}, headers={"x-request-id": "phone-req-1"})
    assert answer.status_code == 200 and answer.headers["x-request-id"] == "phone-req-1"
    assert HTTP_REQUESTS.labels("POST", "/pair", "200")._value.get() == before + 1
    assert any("realme-15" in r.getMessage() and "request phone-req-1" in r.getMessage() for r in caplog.records)
    assert phones.get("/healthz").headers.get("x-request-id")


def test_same_origin_rules() -> None:
    assert same_origin({}, "127.0.0.1:8000")  # no Origin, no Sec-Fetch-Site: not a browser
    assert same_origin({b"origin": b"http://127.0.0.1:8000"}, "127.0.0.1:8000")
    assert same_origin({b"origin": b"https://192.168.1.2:8001"}, "192.168.1.2:8001")
    assert not same_origin({b"origin": b"http://127.0.0.1:8000"}, "127.0.0.1:8001")
    assert not same_origin({b"origin": b"null"}, "127.0.0.1:8000")
    assert not same_origin({b"origin": b"file://"}, "127.0.0.1:8000")
    assert not same_origin({b"sec-fetch-site": b"same-site"}, "127.0.0.1:8000")
    assert ip_or_local_host("192.168.1.2:8001") and ip_or_local_host("[::1]:8001")
    assert not ip_or_local_host("evil.example") and not ip_or_local_host("1.2.3.4.evil.example")
