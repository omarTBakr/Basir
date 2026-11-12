"""Tests for basir/api/worker.py: the stub model in-process, a real spawned worker on the CPU, and the
worker lifecycle (failed start, death, restart) with tiny fake child processes from tests/fake_workers.py."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from multiprocessing.connection import Connection

import cv2
import numpy as np
from prometheus_client import REGISTRY
import pytest

from basir.api.batcher import Batcher
from basir.api.framebus import InProcessFrameBus, QueuedFrame
from basir.api.protocol import FrameHeader
from basir.api.worker import ProcessBackend, StubModel, WorkerDown
from tests import fake_workers


def jpeg(value: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((480, 640, 3), value, np.uint8))
    assert ok
    return buffer.tobytes()


def test_stub_model_scores_every_frame() -> None:
    model = StubModel("cpu")
    scores = model([jpeg(0), jpeg(200), b"not a jpeg"])
    assert len(scores) == 3
    assert scores[0] == pytest.approx(scores[2])  # undecodable frames become black, like frame 0


@pytest.mark.slow
def test_spawned_worker_runs_batches_and_stops() -> None:
    backend = ProcessBackend("cpu")
    backend.start()
    try:
        assert backend.device == "cpu"
        seconds = asyncio.run(backend.infer([jpeg(10), jpeg(20)]))
        assert 0 < seconds < 30
    finally:
        backend.stop()
    with pytest.raises(RuntimeError, match="not running"):
        asyncio.run(backend.infer([jpeg(10)]))


def alive() -> float | None:
    return REGISTRY.get_sample_value("basir_worker_alive")


@pytest.mark.parametrize(
    ("target", "timeout", "message"),
    [
        (fake_workers.crash_on_start, 30.0, "exited during startup"),
        (fake_workers.refuse_on_start, 30.0, "CUDA out of memory"),
        (fake_workers.never_ready, 0.5, "did not start within"),
    ],
)
def test_a_failed_start_raises_clearly_and_cleans_up(
    target: Callable[[Connection, str], None], timeout: float, message: str
) -> None:
    backend = ProcessBackend("cpu", startup_timeout=timeout, target=target)
    with pytest.raises(RuntimeError, match=message):
        backend.start()
    assert backend._conn is None and backend._process is None  # pipe closed, process reaped
    assert not backend.alive and alive() == 0


def test_stop_takes_the_lock_and_is_idempotent() -> None:
    backend = ProcessBackend("cpu", startup_timeout=30, target=fake_workers.serve)
    backend.start()
    assert backend.alive and alive() == 1 and backend.device == "fake"
    assert asyncio.run(backend.infer([b"a", b"b"])) == pytest.approx(0.002)
    backend._lock.acquire()  # a batch "in flight": stop must wait for it
    stopped = asyncio.run(_stop_while_locked(backend))
    assert stopped and backend._conn is None and alive() == 0
    backend.stop()  # again: nothing to do
    with pytest.raises(WorkerDown, match="not running"):
        asyncio.run(backend.infer([b"a"]))


async def _stop_while_locked(backend: ProcessBackend) -> bool:
    stopper = asyncio.create_task(asyncio.to_thread(backend.stop))
    await asyncio.sleep(0.2)
    assert not stopper.done()  # blocked on the lock
    backend._lock.release()
    await stopper
    return True


def test_a_dead_worker_is_logged_once_and_restarted_once(caplog: pytest.LogCaptureFixture) -> None:
    backend = ProcessBackend("cpu", startup_timeout=30, restarts=1, target=fake_workers.die_after_one_batch)
    backend.start()
    try:
        assert asyncio.run(backend.infer([b"x"])) == pytest.approx(0.001)
        with caplog.at_level(logging.ERROR, "basir.worker"), pytest.raises(WorkerDown):
            asyncio.run(backend.infer([b"x"]))  # the child dies; it is restarted once
        assert backend.alive and alive() == 1 and backend.restarts_left == 0
        assert [r.message for r in caplog.records].count("GPU worker died (exit code 1)") == 1
        assert asyncio.run(backend.infer([b"x", b"y"])) == pytest.approx(0.002)  # the new child works
        with pytest.raises(WorkerDown):
            asyncio.run(backend.infer([b"x"]))  # dies again: no restarts left
        assert not backend.alive and alive() == 0
        with pytest.raises(WorkerDown):
            asyncio.run(backend.infer([b"x"]))  # fails fast while down
    finally:
        backend.stop()


def test_batcher_logs_a_down_worker_once_without_tracebacks(caplog: pytest.LogCaptureFixture) -> None:
    class DownBackend:
        device = "down"

        def start(self) -> None: ...

        def stop(self) -> None: ...

        async def infer(self, jpegs: list[bytes]) -> float:
            raise WorkerDown("GPU worker is not running")

    async def scenario() -> Batcher:
        batcher = Batcher(InProcessFrameBus(2), DownBackend(), lambda p: None, 8, 0.0)
        for seq in range(5):
            await batcher.process([QueuedFrame("a", FrameHeader(seq, 0.0), b"\xff\xd8", 0.0)])
        return batcher

    with caplog.at_level(logging.WARNING, "basir.batcher"):
        batcher = asyncio.run(scenario())
    assert batcher.skipped == 5
    assert len(caplog.records) == 1 and caplog.records[0].exc_info is None
