"""Per-phone tokens for the LAN listener: created on the Devices page, stored only as SHA-256 hashes.

store = TokenStore(settings.devices_file)
token = store.create("realme-14")      # shown once, typed or scanned into the phone app
store.verify("realme-14", token)       # True
store.revoke("realme-14")
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import secrets
import time

PREFIX = "bsr_"


class TokenStore:
    """A JSON file of device -> {sha256, created}; the plain token is never written."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, dict[str, str | float]]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, str | float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.chmod(0o600)
        tmp.replace(self.path)

    def create(self, device: str) -> str:
        """A new token for `device`, replacing any earlier one."""
        token = PREFIX + secrets.token_urlsafe(24)
        data = self._load()
        data[device] = {"sha256": _hash(token), "created": time.time()}
        self._save(data)
        return token

    def verify(self, device: str, token: str | None) -> bool:
        entry = self._load().get(device)
        if not token or not entry:
            return False
        return hmac.compare_digest(str(entry.get("sha256", "")), _hash(token))

    def revoke(self, device: str) -> bool:
        data = self._load()
        if data.pop(device, None) is None:
            return False
        self._save(data)
        return True

    def devices(self) -> dict[str, float]:
        """Paired device -> creation time (seconds since the epoch)."""
        return {device: float(entry.get("created", 0)) for device, entry in self._load().items()}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
