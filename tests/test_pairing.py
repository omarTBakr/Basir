"""Tests for basir/api/pairing.py and the phone listener: a phone asks, the laptop approves, the phone streams."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from basir.api.pairing import MAX_PENDING, PairingRequests
from basir.api.protocol import FrameHeader, encode_frame
from basir.api.tokens import TokenStore
from tests.conftest import LanServer
from tests.test_ingest import JPEG


def test_request_approve_collect_once(tmp_path: Path) -> None:
    pairing, tokens = PairingRequests(), TokenStore(tmp_path / "devices.json")
    ask = pairing.open("realme-14", "realme 14", "192.168.1.23")
    assert len(ask.code) == 4 and pairing.pending() == [ask]
    assert pairing.collect(ask.id) == {"status": "pending"}
    assert pairing.approve(ask.id, tokens) is ask and pairing.pending() == []
    assert "192.168.1.23" in pairing.approved_addresses
    answer = pairing.collect(ask.id)
    assert answer is not None and answer["status"] == "approved" and tokens.verify("realme-14", answer["token"])
    assert pairing.collect(ask.id) is None  # the token is handed over once


def test_deny_replace_and_limits(tmp_path: Path) -> None:
    pairing = PairingRequests()
    first = pairing.open("a", "a", "x")
    again = pairing.open("a", "a", "x")  # the same phone asking again replaces its request
    assert pairing.get(first.id) is None and pairing.pending() == [again]
    assert pairing.deny(again.id) and pairing.collect(again.id) == {"status": "denied"}
    assert not pairing.deny("nope") and pairing.approve("nope", TokenStore(tmp_path / "d.json")) is None
    for i in range(MAX_PENDING):
        pairing.open(f"p{i}", "", "x")
    with pytest.raises(ValueError):
        pairing.open("one-too-many", "", "x")


def test_visits_are_remembered() -> None:
    pairing = PairingRequests()
    assert pairing.last_visit() is None
    pairing.seen("192.168.1.23")
    visit = pairing.last_visit()
    assert visit is not None and visit[0] == "192.168.1.23" and visit[1] < 5


def test_phone_pairs_over_the_lan_listener_then_streams(lan_server: LanServer) -> None:
    phone, console = httpx.Client(base_url=lan_server.phone_url), httpx.Client(base_url=lan_server.console_url)
    page = phone.get("/")
    assert page.status_code == 200 and "Basir phone" in page.text and phone.get("/phone/phone.js").status_code == 200
    assert phone.get("/live").status_code == 404  # the Console is not on the phone listener
    assert "Approve the phone" in console.get("/partials/devices").text  # step 2 after the visit

    ask = phone.post("/pair", json={"device": "realme-14", "label": "realme 14"}).json()
    devices = console.get("/partials/devices").text
    assert ask["code"] in devices and "realme 14" in devices
    request_id = ask["request_id"]
    assert phone.get(f"/pair/{request_id}").json() == {"status": "pending"}
    assert console.post(f"/api/devices/requests/{request_id}/approve").status_code == 200
    answer = phone.get(f"/pair/{request_id}").json()
    assert answer["status"] == "approved" and answer["stream_path"] == "/ws/stream/realme-14"
    assert "Start streaming on the phone" in console.get("/partials/devices").text

    ws_url = lan_server.phone_url.replace("http://", "ws://")
    with pytest.raises(InvalidStatus):  # a token in the URL is refused before the handshake
        connect(f"{ws_url}/ws/stream/realme-14?token={answer['token']}").close()
    for device, token in (("realme-14", "wrong"), ("other", answer["token"])):
        with connect(f"{ws_url}/ws/stream/{device}") as ws:
            ws.send(json.dumps({"type": "auth", "token": token}))
            with pytest.raises(ConnectionClosed) as closed:
                ws.recv(timeout=5)
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    with connect(f"{ws_url}/ws/stream/realme-14") as ws:
        ws.send(json.dumps({"type": "auth", "token": answer["token"]}))  # the token is the first message
        ws.send(encode_frame(FrameHeader(seq=0, timestamp=1.0), JPEG))
        assert '"device": "realme-14"' in str(ws.recv())
        assert "is streaming" in console.get("/partials/devices").text  # step 4: done
    assert phone.post("/pair", json={"device": "bad id!"}).status_code == 422
