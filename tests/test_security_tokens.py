"""Security tests for device tokens: never in the URL (first message or Bearer header), expiry, an
atomic token file under concurrent pairing, and no token in any log line."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from basir.api import ingest
from basir.api.ingest import create_ingest_app
from basir.api.middleware import RedactSecretsFilter
from basir.api.protocol import FrameHeader, encode_frame
from basir.api.tokens import DAY_S, TokenStore
from tests.conftest import LanServer
from tests.test_ingest import JPEG

PHONE_WS = "ws://192.168.1.20:8001"  # TestClient sockets ignore base_url, so give the host in full
FRAME = encode_frame(FrameHeader(seq=0, timestamp=1.0), JPEG)


@pytest.fixture
def phones(client: TestClient) -> Iterator[TestClient]:
    with TestClient(create_ingest_app(client.app), base_url="http://192.168.1.20:8001") as phone:  # type: ignore[arg-type]
        yield phone


def _tokens(client: TestClient) -> TokenStore:
    return client.app.state.ingest.tokens  # type: ignore[attr-defined,no-any-return]


# Streaming (a result coming back) needs the Console and the phone listener in ONE event loop, as
# `scripts/serve.py` runs them: the frame bus wakes the batcher with an asyncio.Event, which a second
# TestClient (its own loop) can never trigger. Hence `lan_server` here; the TestClient `phones`
# fixture serves the refusal tests, where the listener answers on its own.


def test_token_as_first_message_streams(lan_server: LanServer) -> None:
    token = lan_server.console.state.ingest.tokens.create("realme-14")
    with connect(f"{_ws(lan_server)}/ws/stream/realme-14") as ws:
        ws.send(json.dumps({"type": "auth", "token": token}))
        ws.send(FRAME)
        assert json.loads(ws.recv(timeout=10))["device"] == "realme-14"


def test_bearer_header_still_works(lan_server: LanServer) -> None:
    token = lan_server.console.state.ingest.tokens.create("realme-14")
    with connect(
        f"{_ws(lan_server)}/ws/stream/realme-14", additional_headers={"Authorization": f"Bearer {token}"}
    ) as ws:
        ws.send(FRAME)
        assert json.loads(ws.recv(timeout=10))["device"] == "realme-14"
    with pytest.raises(InvalidStatus) as refused:  # a wrong Bearer token is refused before the handshake
        connect(f"{_ws(lan_server)}/ws/stream/realme-14", additional_headers={"Authorization": "Bearer wrong"}).close()
    assert refused.value.response.status_code == 403


def _ws(lan_server: LanServer) -> str:
    return lan_server.phone_url.replace("http://", "ws://")


def test_token_in_query_string_is_refused(client: TestClient, phones: TestClient) -> None:
    token = _tokens(client).create("realme-14")
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        phones.websocket_connect(f"{PHONE_WS}/ws/stream/realme-14?token={token}"),
    ):
        pass
    assert refused.value.code == 1008


@pytest.mark.parametrize(
    "first", [json.dumps({"type": "auth", "token": "bsr_wrong"}), "not json", json.dumps({"token": 5}), None]
)
def test_wrong_or_missing_auth_message_closes_1008(
    client: TestClient, phones: TestClient, monkeypatch: pytest.MonkeyPatch, first: str | None
) -> None:
    _tokens(client).create("realme-14")
    monkeypatch.setattr(ingest, "AUTH_TIMEOUT_S", 0.2)
    with phones.websocket_connect(f"{PHONE_WS}/ws/stream/realme-14") as ws:
        if first is None:
            pass  # says nothing: closed after AUTH_TIMEOUT_S
        else:
            ws.send_text(first)
        with pytest.raises(WebSocketDisconnect) as refused:
            ws.receive_text()
    assert refused.value.code == 1008
    assert client.get("/api/live").json()["devices"] == []


def test_expired_token_is_refused_and_shown(client: TestClient, phones: TestClient) -> None:
    store = _tokens(client)
    token = store.create("realme-14")
    data = json.loads(store.path.read_text())
    data["realme-14"]["expires_at"] = time.time() - 1
    store.path.write_text(json.dumps(data))
    assert not store.verify("realme-14", token)
    with phones.websocket_connect(f"{PHONE_WS}/ws/stream/realme-14") as ws:
        ws.send_text(json.dumps({"type": "auth", "token": token}))
        with pytest.raises(WebSocketDisconnect) as refused:
            ws.receive_text()
    assert refused.value.code == 1008
    assert "expired" in client.get("/partials/devices").text


def test_devices_page_shows_expiry(client: TestClient) -> None:
    _tokens(client).create("realme-14")
    page = client.get("/partials/devices").text
    assert "Token expires" in page and "in 30 days" in page


def test_ttl_and_old_file_format(tmp_path: Path) -> None:
    path = tmp_path / "devices.json"
    store = TokenStore(path, ttl_days=2)
    token = store.create("new")
    entry = json.loads(path.read_text())["new"]
    assert set(entry) == {"sha256", "created_at", "expires_at"}
    assert entry["expires_at"] - entry["created_at"] == pytest.approx(2 * DAY_S)

    old = json.loads(path.read_text())
    old["legacy"] = {"sha256": old["new"]["sha256"], "created": time.time() - 3 * DAY_S}  # pre-expiry format
    old["recent"] = {"sha256": old["new"]["sha256"], "created": time.time()}
    path.write_text(json.dumps(old))
    assert store.verify("new", token) and store.verify("recent", token)
    assert not store.verify("legacy", token)  # created 3 days ago, TTL 2 days
    info = store.entries()
    assert info["legacy"].expired() and not info["recent"].expired()
    assert store.devices()["legacy"] == pytest.approx(time.time() - 3 * DAY_S, abs=5)
    assert store.expires_at("nobody") is None


def test_concurrent_pair_and_revoke_lose_nothing(tmp_path: Path) -> None:
    path = tmp_path / "devices.json"
    keep = [f"keep-{i}" for i in range(40)]
    gone = [f"gone-{i}" for i in range(20)]
    for device in gone:
        TokenStore(path).create(device)

    def work(job: tuple[str, str]) -> None:
        action, device = job
        store = TokenStore(path)  # separate instances share the per-file lock
        store.create(device) if action == "create" else store.revoke(device)

    jobs = [("create", d) for d in keep] + [("revoke", d) for d in gone]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(work, jobs))
    assert sorted(TokenStore(path).devices()) == sorted(keep)
    assert [p.name for p in tmp_path.iterdir()] == ["devices.json"]  # no temp files left behind


def test_token_never_reaches_a_log_line() -> None:
    token = "bsr_" + "A" * 32
    lines = [
        ('%s - "WebSocket %s" %s', ("1.2.3.4:5", f"/ws/stream/p?token={token}", "403")),
        ("approved %s", (token,)),
        ("Phone %s connected", ("p",)),
    ]
    out = []
    for msg, args in lines:
        record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, msg, args, None)
        assert RedactSecretsFilter().filter(record)
        out.append(record.getMessage())
    assert all(token not in line for line in out)
    assert out[0].endswith('"WebSocket /ws/stream/p?token=[redacted]" 403') and out[2] == "Phone p connected"
