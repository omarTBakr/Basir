"""Real-time playback of recorded frames: pacing by timestamp, and playback statistics.

No display code here, so the same pacing serves a viewer, a test, or a live-pipeline
simulation that replays a recording as if it came from the camera.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import time

import numpy as np

from basir.core.types import Frame


class Pacer:
    """Sleeps so frames follow their timestamps, scaled by `speed`. Paused time is excluded.

    speed=1 is real time, 2 is twice as fast, 0 (or less) means no waiting.
    """

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed
        self._start = time.perf_counter()
        self._first_t: float | None = None
        self._paused_total = 0.0
        self._paused_at: float | None = None

    def wait(self, timestamp: float) -> None:
        """Block until the frame with this sensor timestamp is due."""
        if self._first_t is None:
            self._first_t = timestamp
        if self.speed <= 0:
            return
        delay = (timestamp - self._first_t) / self.speed - self.elapsed
        if delay > 0:
            time.sleep(delay)

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = time.perf_counter()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += time.perf_counter() - self._paused_at
            self._paused_at = None

    @property
    def elapsed(self) -> float:
        """Playing time in seconds, excluding pauses."""
        now = self._paused_at if self._paused_at is not None else time.perf_counter()
        return now - self._start - self._paused_total


@dataclass
class PlaybackStats:
    frames: int = 0
    valid_depth_sum: float = 0.0
    wall_seconds: float = 0.0

    def add(self, frame: Frame) -> None:
        self.frames += 1
        if frame.depth is not None:
            self.valid_depth_sum += float(np.mean(frame.depth > 0))

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def depth_valid_pct(self) -> float:
        return 100.0 * self.valid_depth_sum / self.frames if self.frames else 0.0

    def metrics(self) -> dict[str, float]:
        """The headline numbers, for MLflow."""
        return {
            "frames": self.frames,
            "fps": self.fps,
            "depth_valid_pct": self.depth_valid_pct,
            "wall_seconds": self.wall_seconds,
        }


def play(frames: Iterable[Frame], pacer: Pacer, on_frame: Callable[[Frame], bool] | None = None) -> PlaybackStats:
    """Pace `frames` by their timestamps and count them; `on_frame` returning False stops early."""
    stats = PlaybackStats()
    for frame in frames:
        pacer.wait(frame.timestamp)
        stats.add(frame)
        if on_frame is not None and not on_frame(frame):
            break
    stats.wall_seconds = pacer.elapsed
    return stats
