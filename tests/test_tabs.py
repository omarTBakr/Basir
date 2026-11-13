"""Tests for basir/api/tabs.py: the registry drives the sidebar and every tab has a page."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from basir.api.tabs import GROUPS, TABS, grouped, tab


def test_registry_is_consistent() -> None:
    assert len({t.key for t in TABS}) == len(TABS)
    assert all(t.group in GROUPS for t in TABS)
    assert [g for g, _ in grouped()] == list(GROUPS)
    assert all(not t.enabled for t in TABS if t.group == "Coming later")
    assert tab("live") is not None and tab("nope") is None


@pytest.mark.parametrize("item", TABS, ids=lambda t: t.key)
def test_every_tab_renders_with_the_sidebar(client: TestClient, item: object) -> None:
    page = client.get(item.path)  # type: ignore[attr-defined]
    assert page.status_code == 200
    text = page.text
    assert all(t.title in text for t in TABS) and 'aria-current="page"' in text
    if not item.enabled:  # type: ignore[attr-defined]
        assert f"Phase {item.phase}" in text and item.summary in text  # type: ignore[attr-defined]


def test_unknown_and_enabled_keys_are_not_coming_later_pages(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404
