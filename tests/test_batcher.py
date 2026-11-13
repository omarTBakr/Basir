"""Tests for basir/api/batcher.py (micro-batching across phones, with a fake worker)."""

from __future__ import annotations

import asyncio
import time

from basir.api.batcher import Batcher, ProcessedFrame
from basir.api.framebus import InProcessFrameBus, QueuedFrame
from basir.api.protocol import FrameHeader
from tests.conftest import FakeBackend


def frame(device: str, seq: int) -> QueuedFrame:
    return QueuedFrame(device, FrameHeader(seq, float(seq)), b"\xff\xd8", time.perf_counter())


class FailingBackend(FakeBackend):
    async def infer(self, jpegs: list[bytes]) -> float:
        raise RuntimeError("GPU worker exited")


def test_one_batch_serves_every_phone() -> None:
    delivered: list[ProcessedFrame] = []
    backend = FakeBackend(seconds=0.004)

    async def scenario() -> None:
        bus = InProcessFrameBus(queue_size=2)
        batcher = Batcher(bus, backend, delivered.append, max_size=8, max_wait=0.01)
        for device in ("a", "b"):
            bus.put(frame(device, 0))
        task = asyncio.create_task(batcher.run())
        while len(delivered) < 2:
            await asyncio.sleep(0.005)
        task.cancel()

    asyncio.run(scenario())
    assert backend.batches == [2]
    assert {p.frame.device for p in delivered} == {"a", "b"}
    for p in delivered:
        assert p.batch_size == 2 and p.inference_s == 0.004
        assert p.latency_s >= p.queue_s >= 0


def test_batches_never_exceed_max_size() -> None:
    delivered: list[ProcessedFrame] = []
    backend = FakeBackend()

    async def scenario() -> None:
        bus = InProcessFrameBus(queue_size=3)
        batcher = Batcher(bus, backend, delivered.append, max_size=4, max_wait=0.0)
        for device in "abc":
            for i in range(3):
                bus.put(frame(device, i))
        await batcher.process(await bus.next_batch(batcher.max_size, batcher.max_wait))
        await batcher.process(await bus.next_batch(batcher.max_size, batcher.max_wait))
        await batcher.process(await bus.next_batch(batcher.max_size, batcher.max_wait))

    asyncio.run(scenario())
    assert backend.batches == [4, 4, 1]
    assert len(delivered) == 9


def test_failed_batch_delivers_nothing_and_batching_continues() -> None:
    delivered: list[ProcessedFrame] = []

    async def scenario() -> None:
        bus = InProcessFrameBus(queue_size=2)
        await Batcher(bus, FailingBackend(), delivered.append, 8, 0.0).process([frame("a", 0)])

    asyncio.run(scenario())
    assert delivered == []
