"""Plug-and-play pairing: a phone asks to join over Wi-Fi, you approve it on the Devices tab, it gets its token.

    phone                                  laptop (phone listener, :8001)          Devices tab
    ─────                                  ──────────────────────────────          ───────────
    POST /pair {"device", "label"}   ──▶   pending request, 4-digit code    ──▶   "realme-14 wants in, code 4821"
    GET /pair/{id} (poll every 1 s)  ◀──   {"status": "pending"}                   [Approve] [Deny]
    GET /pair/{id}                   ◀──   {"status": "approved", "token", "stream_path"}   (once)
    WS /ws/stream/{device}?token=...

The code is shown on both screens so you can tell which phone you are approving. Requests
expire after `PAIR_TTL_S`; at most `MAX_PENDING` wait at once.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import time
from typing import Any

from basir.api.tokens import TokenStore

PAIR_TTL_S = 300.0
MAX_PENDING = 10


@dataclass
class PairRequest:
    id: str
    device: str
    label: str
    code: str
    address: str
    created: float
    status: str = "pending"  # pending | approved | denied | paired
    token: str | None = None

    def public(self) -> dict[str, Any]:
        """What the Devices tab shows."""
        return {
            "id": self.id,
            "device": self.device,
            "label": self.label,
            "code": self.code,
            "address": self.address,
            "age_s": round(time.time() - self.created),
        }


class PairingRequests:
    """Pending pairing requests and recent phone-page visits, in memory (a restart forgets them;
    the phone simply asks again)."""

    def __init__(self) -> None:
        self._requests: dict[str, PairRequest] = {}
        self._visits: dict[str, float] = {}  # phone address -> last time it opened the phone page
        self.approved_addresses: set[str] = set()  # phones approved since start (drives the Devices steps)

    def seen(self, address: str) -> None:
        """A phone opened the phone page (so it is on the Wi-Fi and accepted the certificate)."""
        self._visits[address] = time.time()

    def last_visit(self, within_s: float = PAIR_TTL_S) -> tuple[str, float] | None:
        """(address, seconds ago) of the most recent phone-page visit, if recent."""
        if not self._visits:
            return None
        address, at = max(self._visits.items(), key=lambda item: item[1])
        age = time.time() - at
        return (address, age) if age <= within_s else None

    def _expire(self) -> None:
        now = time.time()
        for rid in [r.id for r in self._requests.values() if now - r.created > PAIR_TTL_S]:
            del self._requests[rid]

    def open(self, device: str, label: str, address: str) -> PairRequest:
        self._expire()
        for old in [r for r in self._requests.values() if r.device == device and r.status == "pending"]:
            del self._requests[old.id]  # the same phone asking again replaces its old request
        if len(self.pending()) >= MAX_PENDING:
            raise ValueError("too many pairing requests waiting; approve or deny some first")
        request = PairRequest(
            id=secrets.token_urlsafe(12),
            device=device,
            label=label[:64],
            code=f"{secrets.randbelow(10_000):04d}",
            address=address,
            created=time.time(),
        )
        self._requests[request.id] = request
        return request

    def get(self, request_id: str) -> PairRequest | None:
        self._expire()
        return self._requests.get(request_id)

    def pending(self) -> list[PairRequest]:
        self._expire()
        return [r for r in self._requests.values() if r.status == "pending"]

    def approve(self, request_id: str, tokens: TokenStore) -> PairRequest | None:
        request = self.get(request_id)
        if request is None or request.status != "pending":
            return None
        request.token = tokens.create(request.device)
        request.status = "approved"
        self.approved_addresses.add(request.address)
        return request

    def deny(self, request_id: str) -> bool:
        request = self.get(request_id)
        if request is None or request.status != "pending":
            return False
        request.status = "denied"
        return True

    def collect(self, request_id: str) -> dict[str, Any] | None:
        """The phone's view of its request; an approved token is handed over once, then forgotten."""
        request = self.get(request_id)
        if request is None:
            return None
        if request.status != "approved":
            return {"status": request.status}
        answer = {
            "status": "approved",
            "device": request.device,
            "token": request.token,
            "stream_path": f"/ws/stream/{request.device}",
        }
        del self._requests[request_id]
        return answer
