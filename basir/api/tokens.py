"""Per-phone tokens for the LAN listener: created by pairing approval or on the Devices page, stored only
as SHA-256 hashes, valid for `TOKEN_TTL_DAYS` (default 30).

store = TokenStore(settings.devices_file)
token = store.create("realme-14")      # shown once, typed or scanned into the phone app
store.verify("realme-14", token)       # True until it expires or is revoked
store.revoke("realme-14")

File format (`devices.json`): `{device: {"sha256", "created_at", "expires_at"}}`, times in seconds
since the epoch. Files written before expiry existed hold `{"sha256", "created"}`; such an entry
expires `ttl_days` after `created`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time

PREFIX = "bsr_"
DEFAULT_TTL_DAYS = 30.0
DAY_S = 86_400.0

_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """One lock per token file, shared by every `TokenStore` on it in this process."""
    with _locks_guard:
        return _locks.setdefault(path.resolve(), threading.Lock())


@dataclass(frozen=True)
class TokenInfo:
    """When a paired device's token was made and when it stops working (seconds since the epoch)."""

    created_at: float
    expires_at: float

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    def expires_label(self, now: float | None = None) -> str:
        """'in 29 days (2026-12-12)', 'in 5 h (2026-11-12)' for the Devices page."""
        left = self.expires_at - (time.time() if now is None else now)
        date = datetime.fromtimestamp(self.expires_at, UTC).strftime("%Y-%m-%d")
        span = f"{left / DAY_S:.0f} days" if left >= 2 * DAY_S else f"{max(left, 0) / 3600:.0f} h"
        return f"in {span} ({date})"


class TokenStore:
    """A JSON file of device -> {sha256, created_at, expires_at}; the plain token is never written.

    Every read-modify-write holds a per-file lock and writes a unique temp file that replaces
    the real one atomically, so concurrent pairing and revoking cannot lose an entry.
    """

    def __init__(self, path: Path, ttl_days: float = DEFAULT_TTL_DAYS) -> None:
        self.path = path
        self.ttl_s = ttl_days * DAY_S
        self._lock = _lock_for(path)

    def _load(self) -> dict[str, dict[str, str | float]]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, str | float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)  # before any secret is written (a no-op on ntfs3/vfat mounts)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise

    def _info(self, entry: dict[str, str | float]) -> TokenInfo:
        created = float(entry.get("created_at", entry.get("created", 0)) or 0)
        expires = entry.get("expires_at")
        return TokenInfo(created, float(expires) if expires is not None else created + self.ttl_s)

    def create(self, device: str) -> str:
        """A new token for `device`, replacing any earlier one; valid for `ttl_days`."""
        token = PREFIX + secrets.token_urlsafe(24)
        now = time.time()
        with self._lock:
            data = self._load()
            data[device] = {"sha256": _hash(token), "created_at": now, "expires_at": now + self.ttl_s}
            self._save(data)
        return token

    def verify(self, device: str, token: str | None) -> bool:
        """True when `token` is `device`'s current token and has not expired."""
        entry = self._load().get(device)
        if not token or not isinstance(entry, dict):
            return False
        if self._info(entry).expired():
            return False
        return hmac.compare_digest(str(entry.get("sha256", "")), _hash(token))

    def revoke(self, device: str) -> bool:
        with self._lock:
            data = self._load()
            if data.pop(device, None) is None:
                return False
            self._save(data)
        return True

    def entries(self) -> dict[str, TokenInfo]:
        """Paired device -> its token's creation and expiry times (expired ones included)."""
        return {d: self._info(e) for d, e in self._load().items() if isinstance(e, dict)}

    def devices(self) -> dict[str, float]:
        """Paired device -> creation time (seconds since the epoch)."""
        return {device: info.created_at for device, info in self.entries().items()}

    def expires_at(self, device: str) -> float | None:
        """When `device`'s token expires, or None when it has none (for the Devices page)."""
        info = self.entries().get(device)
        return info.expires_at if info else None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
