"""Tests for basir/api/worker.py: the stub model in-process, and one real spawned worker on the CPU."""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
import pytest

from basir.api.worker import ProcessBackend, StubModel


def jpeg(value: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((480, 640, 3), value, np.uint8))
    assert ok
    return buffer.tobytes()


def test_stub_model_scores_every_frame() -> None:
    model = StubModel("cpu")
    scores = model([jpeg(0), jpeg(200), b"not a jpeg"])
    assert len(scores) == 3
    assert scores[0] == pytest.approx(scores[2])  # undecodable frames become black, like frame 0


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
