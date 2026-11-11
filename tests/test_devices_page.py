"""Tests for basir/api/devices_page.py: pairing shows a token once, revoking removes it."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from basir.api.lan import lan_address


def test_page_shows_the_qr_link_and_step_one(client: TestClient) -> None:
    page = client.get("/devices").text
    assert f"https://{lan_address()}:8001/" in page and "qrcode-generator@1.4.4" in page
    steps = client.get("/partials/devices").text
    assert "Open Basir on the phone" in steps and "Waiting for the phone to open the page" in steps


def test_pair_then_revoke(client: TestClient) -> None:
    assert "No phone paired yet" in client.get("/partials/devices").text
    paired = client.post("/api/devices/pair", data={"device": "realme-14"})
    assert paired.status_code == 200
    token = re.search(r"bsr_[A-Za-z0-9_\-]+", paired.text)
    assert token and client.app.state.ingest.tokens.verify("realme-14", token.group())  # type: ignore[attr-defined]
    listing = client.get("/partials/devices").text
    assert "realme-14" in listing and token.group() not in listing  # shown once only

    revoked = client.post("/api/devices/realme-14/revoke").text
    assert "No phone paired yet" in revoked


def test_pair_rejects_bad_device_ids(client: TestClient) -> None:
    assert client.post("/api/devices/pair", data={"device": "bad id!"}).status_code == 400


def test_lan_address_is_an_ipv4_address() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+", lan_address())
