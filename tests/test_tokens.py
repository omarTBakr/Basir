"""Tests for basir/api/tokens.py: per-phone tokens, stored hashed."""

from __future__ import annotations

from pathlib import Path

from basir.api.tokens import PREFIX, TokenStore


def test_create_verify_revoke(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "devices.json")
    token = store.create("realme-14")
    assert token.startswith(PREFIX) and token not in (tmp_path / "devices.json").read_text()
    assert store.verify("realme-14", token)
    assert not store.verify("realme-14", token + "x") and not store.verify("other", token)
    assert not store.verify("realme-14", None)
    assert set(store.devices()) == {"realme-14"}

    replaced = store.create("realme-14")
    assert not store.verify("realme-14", token) and store.verify("realme-14", replaced)
    assert store.revoke("realme-14") and not store.revoke("realme-14")
    assert not store.verify("realme-14", replaced) and store.devices() == {}


def test_missing_or_corrupt_file_means_no_devices(tmp_path: Path) -> None:
    path = tmp_path / "devices.json"
    assert TokenStore(path).devices() == {}
    path.write_text("not json")
    assert TokenStore(path).devices() == {} and not TokenStore(path).verify("a", "b")
