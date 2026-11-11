"""The Devices tab: a step-by-step guide to connect a phone, pairing requests, connected and paired phones."""

from __future__ import annotations

from datetime import UTC, datetime
import socket
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from basir.api.ingest import DEVICE_ID, IngestState
from basir.api.lan import lan_address, lan_addresses
from basir.api.ui import live_stats, templates

router = APIRouter(include_in_schema=False)


HOTSPOT_PREFIX = "10.42."  # NetworkManager's address range for a hotspot the laptop shares


def phone_url(request: Request, address: str | None = None) -> str:
    """The address a phone opens (a QR code): the phone listener on one of this laptop's addresses."""
    settings = request.app.state.settings
    return f"{'https' if settings.ingest_tls else 'http'}://{address or lan_address()}:{settings.ingest_port}/"


def networks(request: Request) -> list[dict[str, str]]:
    """One entry per network a phone can join: the router's network and any hotspot the laptop shares."""
    return [
        {
            "interface": interface,
            "kind": "hotspot" if address.startswith(HOTSPOT_PREFIX) else "network",
            "url": phone_url(request, address),
        }
        for interface, address in lan_addresses()
    ]


def current_step(state: IngestState, streaming: bool) -> int:
    """Where the guide is: 1 open the page, 2 approve the phone, 3 start streaming, 4 done."""
    if streaming:
        return 4
    if state.pairing.pending():
        return 2
    visit = state.pairing.last_visit()
    if visit is None:
        return 1
    # A phone approved now, or paired before (it reopened the page with its stored token), only has to start.
    return 3 if visit[0] in state.pairing.approved_addresses or state.tokens.devices() else 2


def _context(request: Request) -> dict[str, Any]:
    state: IngestState = request.app.state.ingest
    live = live_stats(request)
    paired = {d: datetime.fromtimestamp(t, UTC) for d, t in sorted(state.tokens.devices().items())}
    return {
        "live": live,
        "paired": paired,
        "requests": [r.public() for r in state.pairing.pending()],
        "step": current_step(state, bool(live["devices"])),
        "phone_url": phone_url(request),
    }


@router.get("/devices")
def devices_page(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    context = {
        "page": "devices",
        "title": "Devices",
        "networks": networks(request),
        "phone_url": phone_url(request),
        "laptop": socket.gethostname(),
        "lan_enabled": settings.ingest_host not in ("127.0.0.1", "localhost", ""),
    }
    return templates.TemplateResponse(request, "devices.html", context)


@router.get("/partials/devices")
def devices_partial(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "partials/devices.html", _context(request))


@router.post("/api/devices/requests/{request_id}/approve")
def approve(request: Request, request_id: str) -> HTMLResponse:
    """Approve a phone's pairing request; the phone collects its token on its next poll."""
    state: IngestState = request.app.state.ingest
    if state.pairing.approve(request_id, state.tokens) is None:
        raise HTTPException(404, "unknown or expired pairing request")
    return templates.TemplateResponse(request, "partials/devices.html", _context(request))


@router.post("/api/devices/requests/{request_id}/deny")
def deny(request: Request, request_id: str) -> HTMLResponse:
    request.app.state.ingest.pairing.deny(request_id)
    return templates.TemplateResponse(request, "partials/devices.html", _context(request))


@router.post("/api/devices/pair")
def pair(request: Request, device: str = Form(...)) -> HTMLResponse:
    """Manual pairing: create (or replace) a phone's token, shown once in the returned fragment."""
    device = device.strip()
    if not DEVICE_ID.match(device):
        raise HTTPException(400, "device id: 1-64 letters, digits, '.', '_' or '-'")
    state: IngestState = request.app.state.ingest
    token = state.tokens.create(device)
    context = _context(request) | {"new_device": device, "new_token": token}
    return templates.TemplateResponse(request, "partials/devices.html", context)


@router.post("/api/devices/{device_id}/revoke")
def revoke(request: Request, device_id: str) -> HTMLResponse:
    request.app.state.ingest.tokens.revoke(device_id)
    return templates.TemplateResponse(request, "partials/devices.html", _context(request))
