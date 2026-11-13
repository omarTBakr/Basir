"""Tests for basir/api/temporal_view.py: graceful degradation when Temporal is down."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from basir.api.temporal_view import TemporalSnapshot, TemporalView, WorkflowSummary
from tests.conftest import UNREACHABLE


def test_unreachable_temporal_gives_a_snapshot_not_an_error() -> None:
    view = TemporalView(UNREACHABLE, "http://127.0.0.1:8233", timeout=3.0)
    snapshot = asyncio.run(view.snapshot())
    assert not snapshot.reachable and snapshot.workflows == []
    assert UNREACHABLE in (snapshot.error or "")


def test_by_id_finds_the_listed_workflow() -> None:
    workflow = WorkflowSummary("phase0-abc", "run-1", "Phase0Pipeline", "basir-pipelines", "running", None, None, "u")
    snapshot = TemporalSnapshot(reachable=True, workflows=[workflow])
    assert snapshot.by_id("phase0-abc") is workflow
    assert snapshot.by_id("other") is None and snapshot.by_id(None) is None


def test_workflow_links_point_at_the_temporal_ui() -> None:
    class FakeExecution:
        id, run_id, workflow_type, task_queue = "a/b", "r1", "Phase0Pipeline", "basir-pipelines"
        status = None
        start_time = datetime(2026, 11, 10, tzinfo=UTC)
        close_time = None

    class FakeClient:
        async def _items(self) -> object:
            yield FakeExecution()

        def list_workflows(self, limit: int) -> object:
            return self._items()

    view = TemporalView("unused:7233", "http://127.0.0.1:8233/")
    view._client = FakeClient()  # type: ignore[assignment]
    (workflow,) = asyncio.run(view.snapshot()).workflows
    assert workflow.url == "http://127.0.0.1:8233/namespaces/default/workflows/a%2Fb/r1/history"
    assert workflow.status == "unknown"
