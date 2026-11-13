"""Where received frames wait for the GPU worker: one small latest-frame-wins queue per phone.

`FrameBus` is the seam: the in-process asyncio version below serves one server; a Redis
Streams version can replace it later. `put`, `depth` and `remove` are synchronous and called
inline by the ingest socket, so such a version would buffer locally or make them async.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Protocol

from basir.api.protocol import FrameHeader


@dataclass(frozen=True)
class QueuedFrame:
    """A received frame waiting for inference."""

    device: str
    header: FrameHeader
    jpeg: bytes
    received_at: float  # time.perf_counter() on receipt (for latency; not a wall-clock time)
    depth: bytes | None = None  # uint16 depth described by header.depth (protocol v2)
    received_wall: float = field(default_factory=time.time)  # server wall clock on receipt (header.timestamp
    # is the phone's own clock, e.g. performance.now(), so it cannot say when a frame arrived)


class FrameBus(Protocol):
    """Per-device frame queues that a batcher drains across devices."""

    def put(self, frame: QueuedFrame) -> bool:
        """Queue a frame; return True if an older frame of the same device was dropped for it."""
        ...

    async def next_batch(self, max_size: int, max_wait: float) -> list[QueuedFrame]:
        """Wait for a frame, then gather up to `max_size` for at most `max_wait` seconds."""
        ...

    def depth(self, device: str) -> int:
        """Frames waiting for `device`."""
        ...

    def remove(self, device: str) -> None:
        """Forget a device and discard its waiting frames."""
        ...


class InProcessFrameBus:
    """`FrameBus` on asyncio: bounded deques, oldest frame dropped when a device's queue is full."""

    def __init__(self, queue_size: int) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self.queue_size = queue_size
        self._queues: dict[str, deque[QueuedFrame]] = {}
        self._ready = asyncio.Event()
        self._turn = 0  # rotates which device goes first, so no phone is always last

    def put(self, frame: QueuedFrame) -> bool:
        queue = self._queues.setdefault(frame.device, deque())
        dropped = len(queue) >= self.queue_size
        if dropped:
            queue.popleft()
        queue.append(frame)
        self._ready.set()
        return dropped

    async def next_batch(self, max_size: int, max_wait: float) -> list[QueuedFrame]:
        await self._ready.wait()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait
        batch = self._take(max_size)
        while len(batch) < max_size and (remaining := deadline - loop.time()) > 0:
            try:
                await asyncio.wait_for(self._ready.wait(), remaining)
            except TimeoutError:
                break
            batch += self._take(max_size - len(batch))
        return batch

    def depth(self, device: str) -> int:
        queue = self._queues.get(device)
        return len(queue) if queue else 0

    def remove(self, device: str) -> None:
        self._queues.pop(device, None)
        if not any(self._queues.values()):
            self._ready.clear()

    def _take(self, limit: int) -> list[QueuedFrame]:
        """Up to `limit` frames, one per device in turn (oldest first within a device)."""
        devices = list(self._queues)
        if devices:
            self._turn = (self._turn + 1) % len(devices)
            devices = devices[self._turn :] + devices[: self._turn]
        taken: list[QueuedFrame] = []
        while len(taken) < limit and any(self._queues[d] for d in devices):
            for device in devices:
                if self._queues[device] and len(taken) < limit:
                    taken.append(self._queues[device].popleft())
        if not any(self._queues.values()):
            self._ready.clear()
        return taken
