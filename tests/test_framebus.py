"""Tests for basir/api/framebus.py (per-phone latest-frame-wins queues)."""

from __future__ import annotations

import asyncio
import time

import pytest

from basir.api.framebus import InProcessFrameBus, QueuedFrame
from basir.api.protocol import FrameHeader


def frame(device: str, seq: int) -> QueuedFrame:
    return QueuedFrame(device, FrameHeader(seq, float(seq)), b"\xff\xd8", time.perf_counter())


def test_full_queue_drops_the_oldest_frame() -> None:
    async def scenario() -> list[tuple[str, int]]:
        bus = InProcessFrameBus(queue_size=2)
        dropped = [bus.put(frame("a", i)) for i in range(5)]
        assert dropped == [False, False, True, True, True]
        assert bus.depth("a") == 2
        batch = await bus.next_batch(8, 0.0)
        return [(f.device, f.header.seq) for f in batch]

    assert asyncio.run(scenario()) == [("a", 3), ("a", 4)]


def test_batch_mixes_devices_and_respects_max_size() -> None:
    async def scenario() -> tuple[list[str], int]:
        bus = InProcessFrameBus(queue_size=4)
        for device in ("a", "b", "c"):
            for i in range(3):
                bus.put(frame(device, i))
        batch = await bus.next_batch(4, 0.0)
        return [f.device for f in batch], sum(bus.depth(d) for d in "abc")

    devices, left = asyncio.run(scenario())
    assert len(devices) == 4 and set(devices) == {"a", "b", "c"}  # one per device before seconds
    assert left == 5


def test_batch_waits_for_late_frames_up_to_max_wait() -> None:
    async def scenario() -> tuple[int, float]:
        bus = InProcessFrameBus(queue_size=2)
        bus.put(frame("a", 0))
        asyncio.get_running_loop().call_later(0.01, bus.put, frame("b", 0))
        start = time.perf_counter()
        batch = await bus.next_batch(8, 0.2)
        return len(batch), time.perf_counter() - start

    size, waited = asyncio.run(scenario())
    assert size == 2
    assert 0.15 < waited < 10.0  # the batch was not full, so it stayed open for max_wait (upper bound: slack)


def test_full_batch_closes_without_waiting() -> None:
    async def scenario() -> float:
        bus = InProcessFrameBus(queue_size=2)
        bus.put(frame("a", 0))
        bus.put(frame("b", 0))
        start = time.perf_counter()
        await bus.next_batch(2, 30.0)
        return time.perf_counter() - start

    assert asyncio.run(scenario()) < 15.0  # far below max_wait (30 s), with room for a loaded machine


def test_next_batch_blocks_until_a_frame_arrives() -> None:
    async def scenario() -> bool:
        bus = InProcessFrameBus(queue_size=2)
        waiting = asyncio.create_task(bus.next_batch(8, 0.0))
        await asyncio.sleep(0.02)
        blocked = not waiting.done()
        bus.put(frame("a", 0))
        assert len(await waiting) == 1
        return blocked

    assert asyncio.run(scenario())


def test_remove_discards_waiting_frames() -> None:
    async def scenario() -> int:
        bus = InProcessFrameBus(queue_size=2)
        bus.put(frame("a", 0))
        bus.remove("a")
        return bus.depth("a")

    assert asyncio.run(scenario()) == 0


def test_queue_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        InProcessFrameBus(queue_size=0)
