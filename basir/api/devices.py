"""Connected phones: one session each, with its result queue and rolling statistics for the Live page."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
import statistics
import time

from basir.api.batcher import ProcessedFrame
from basir.api.protocol import FrameResult

FPS_WINDOW_S = 5.0  # frames/s is measured over this trailing window
RECENT = 200  # results kept for latency and batch statistics


@dataclass
class DeviceSession:
    """One connected phone."""

    device: str
    connected_at: float = field(default_factory=time.time)
    results: asyncio.Queue[FrameResult] = field(default_factory=asyncio.Queue)
    received: int = 0
    dropped: int = 0
    processed: int = 0
    _arrivals: deque[float] = field(default_factory=deque)
    _recent: deque[ProcessedFrame] = field(default_factory=lambda: deque(maxlen=RECENT))

    def record_received(self, dropped_older: bool) -> None:
        now = time.perf_counter()
        self.received += 1
        self.dropped += dropped_older
        self._arrivals.append(now)
        while self._arrivals and now - self._arrivals[0] > FPS_WINDOW_S:
            self._arrivals.popleft()

    def record_processed(self, processed: ProcessedFrame) -> FrameResult:
        self.processed += 1
        self._recent.append(processed)
        return FrameResult(
            device=self.device,
            seq=processed.frame.header.seq,
            batch_size=processed.batch_size,
            queue_ms=round(processed.queue_s * 1000, 2),
            inference_ms=round(processed.inference_s * 1000, 2),
            latency_ms=round(processed.latency_s * 1000, 2),
            dropped=self.dropped,
        )

    def stats(self, queue_depth: int) -> dict[str, float | int | str]:
        """Numbers for the Live page and `/api/live`."""
        now = time.perf_counter()
        arrivals = [t for t in self._arrivals if now - t <= FPS_WINDOW_S]
        window = min(FPS_WINDOW_S, time.time() - self.connected_at)
        latencies = sorted(p.latency_s * 1000 for p in self._recent)
        return {
            "device": self.device,
            "connected_s": round(time.time() - self.connected_at, 1),
            "fps": round(len(arrivals) / window, 1) if window > 0 else 0.0,
            "received": self.received,
            "processed": self.processed,
            "dropped": self.dropped,
            "queue_depth": queue_depth,
            "batch_size": _mean(p.batch_size for p in self._recent),
            "inference_ms": _mean(p.inference_s * 1000 for p in self._recent),
            "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
            "latency_p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else 0.0,
        }


class DeviceRegistry:
    """The phones connected right now, keyed by device ID."""

    def __init__(self) -> None:
        self.sessions: dict[str, DeviceSession] = {}

    def connect(self, device: str) -> DeviceSession | None:
        """Open a session, or None if that device ID is already connected."""
        if device in self.sessions:
            return None
        self.sessions[device] = session = DeviceSession(device)
        return session

    def disconnect(self, device: str) -> None:
        self.sessions.pop(device, None)

    def deliver(self, processed: ProcessedFrame) -> None:
        """Queue a result for its phone's socket; results for phones that left are discarded."""
        session = self.sessions.get(processed.frame.device)
        if session is not None:
            session.results.put_nowait(session.record_processed(processed))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 1) if items else 0.0
