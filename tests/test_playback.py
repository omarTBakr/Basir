"""Tests for basir/ingest/playback.py (pacing and statistics)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from basir.core.types import CameraIntrinsics, Frame
from basir.ingest.playback import Pacer, PlaybackStats, play

INTR = CameraIntrinsics(1.0, 1.0, 0.5, 0.5, 2, 2)


def _frame(depth: np.ndarray | None) -> Frame:
    return Frame(id=0, timestamp=0.0, rgb=np.zeros((2, 2, 3), np.uint8), intrinsics=INTR, depth=depth)


def test_pacer_follows_timestamps() -> None:
    pacer = Pacer(speed=1.0)
    start = time.perf_counter()
    for t in (100.0, 100.05, 100.10):
        pacer.wait(t)
    assert time.perf_counter() - start == pytest.approx(0.10, abs=0.04)


def test_pacer_speed_and_no_wait() -> None:
    fast = Pacer(speed=4.0)
    start = time.perf_counter()
    fast.wait(0.0)
    fast.wait(0.2)
    assert time.perf_counter() - start == pytest.approx(0.05, abs=0.03)

    instant = Pacer(speed=0)
    start = time.perf_counter()
    instant.wait(0.0)
    instant.wait(10.0)
    assert time.perf_counter() - start < 0.01


def test_paused_time_is_excluded() -> None:
    pacer = Pacer(speed=1.0)
    pacer.wait(0.0)
    pacer.pause()
    time.sleep(0.08)
    paused_elapsed = pacer.elapsed
    time.sleep(0.02)
    assert pacer.elapsed == pytest.approx(paused_elapsed)  # frozen while paused
    pacer.resume()
    assert pacer.elapsed < 0.05


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
