"""Tests for basir/ingest/playback.py (pacing and statistics)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from basir.core.types import CameraIntrinsics, Frame
from basir.ingest import playback
from basir.ingest.playback import Pacer, PlaybackStats, play

INTR = CameraIntrinsics(1.0, 1.0, 0.5, 0.5, 2, 2)


def _frame(depth: np.ndarray | None) -> Frame:
    return Frame(id=0, timestamp=0.0, rgb=np.zeros((2, 2, 3), np.uint8), intrinsics=INTR, depth=depth)


class FakeClock:
    """Stands in for the `time` module inside playback: sleeping only advances the clock."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(playback, "time", fake)
    return fake


def test_pacer_follows_timestamps(clock: FakeClock) -> None:
    pacer = Pacer(speed=1.0)
    for t in (100.0, 100.05, 100.10):
        pacer.wait(t)
    assert clock.now - 1000.0 == pytest.approx(0.10)
    assert clock.slept == pytest.approx([0.05, 0.05])


def test_pacer_does_not_wait_when_behind(clock: FakeClock) -> None:
    pacer = Pacer(speed=1.0)
    pacer.wait(0.0)
    clock.now += 0.5  # the caller took longer than the frame interval
    pacer.wait(0.1)
    assert clock.slept == []


def test_pacer_speed_and_no_wait(clock: FakeClock) -> None:
    fast = Pacer(speed=4.0)
    fast.wait(0.0)
    fast.wait(0.2)
    assert clock.slept == pytest.approx([0.05])

    instant = Pacer(speed=0)
    instant.wait(0.0)
    instant.wait(10.0)
    assert clock.slept == pytest.approx([0.05])  # no new sleep


def test_paused_time_is_excluded(clock: FakeClock) -> None:
    pacer = Pacer(speed=1.0)
    pacer.wait(0.0)
    pacer.pause()
    clock.now += 0.08
    paused_elapsed = pacer.elapsed
    clock.now += 0.02
    assert pacer.elapsed == pytest.approx(paused_elapsed)  # frozen while paused
    pacer.resume()
    assert pacer.elapsed == pytest.approx(0.0)
    clock.now += 0.03
    assert pacer.elapsed == pytest.approx(0.03)


def test_pacer_really_sleeps() -> None:
    """One real-clock check with a generous bound (the fake-clock tests cover the arithmetic)."""
    pacer = Pacer(speed=1.0)
    start = time.perf_counter()
    pacer.wait(0.0)
    pacer.wait(0.05)
    assert time.perf_counter() - start >= 0.045


def test_stats() -> None:
    stats = PlaybackStats()
    depth = np.array([[1.0, 0.0], [2.0, 3.0]], np.float32)
    stats.add(_frame(depth))
    stats.add(_frame(None))
    stats.wall_seconds = 0.5
    assert stats.frames == 2
    assert stats.fps == pytest.approx(4.0)
    assert stats.depth_valid_pct == pytest.approx(37.5)  # 75 % valid in one frame, none counted in the other
    assert PlaybackStats().fps == 0.0 and PlaybackStats().depth_valid_pct == 0.0
    assert stats.metrics() == {"frames": 2, "fps": 4.0, "depth_valid_pct": 37.5, "wall_seconds": 0.5}


def test_play_counts_frames_and_stops_when_asked() -> None:
    depth = np.ones((2, 2), np.float32)
    frames = [_frame(depth) for _ in range(5)]
    assert play(frames, Pacer(speed=0)).frames == 5
    seen: list[Frame] = []

    def until_third(frame: Frame) -> bool:
        seen.append(frame)
        return len(seen) < 3

    stats = play(frames, Pacer(speed=0), on_frame=until_third)
    assert stats.frames == 3 and len(seen) == 3
    assert stats.depth_valid_pct == 100.0
