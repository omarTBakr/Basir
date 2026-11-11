"""Tests for basir/spatial/live_map.py: the keyframe gate and the incremental voxel map."""

from __future__ import annotations

import math

import numpy as np

from basir.core.geometry import make_transform
from basir.spatial.live_map import KeyframeGate, VoxelMap


def pose(x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    a = math.radians(yaw_deg)
    rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return make_transform(rotation, [x, 0.0, 0.0])


def test_gate_accepts_the_first_pose_then_only_real_motion() -> None:
    gate = KeyframeGate(min_move_m=0.05, min_turn_deg=5.0)
    assert gate.accept(pose())
    assert not gate.accept(pose(x=0.03))  # moved 3 cm
    assert not gate.accept(pose(yaw_deg=3))  # turned 3 degrees
    assert gate.accept(pose(x=0.06))  # moved 6 cm
    assert gate.accept(pose(x=0.06, yaw_deg=6))  # turned 6 degrees in place
    gate.reset()
    assert gate.accept(pose(x=0.06, yaw_deg=6))


def test_voxel_map_averages_per_voxel_and_reports_only_new_voxels() -> None:
    vmap = VoxelMap(voxel=0.1, max_points=100)
    first = vmap.insert(
        np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.51, 0, 0]]),
        np.array([[0, 0, 0], [200, 100, 50], [9, 9, 9]]),
    )
    assert list(first) == [0, 1] and len(vmap) == 2
    xyz, rgb = vmap.points(first)
    assert np.allclose(xyz[0], [0.02, 0.02, 0.02]) and list(rgb[0]) == [100, 50, 25]

    again = vmap.insert(np.array([[0.05, 0.05, 0.05], [-0.01, 0, 0]]), np.array([[100, 100, 100], [1, 2, 3]]))
    assert list(again) == [2]  # the first point fell into a known voxel; the second is new (negative cell)
    xyz, _ = vmap.points()
    assert len(xyz) == 3 and np.allclose(xyz[0], [0.03, 0.03, 0.03])


def test_voxel_map_grows_past_its_initial_buffer_and_respects_the_budget() -> None:
    grid = np.stack(np.meshgrid(np.arange(40), np.arange(40), np.arange(2), indexing="ij"), -1).reshape(-1, 3) * 0.1
    vmap = VoxelMap(voxel=0.1, max_points=3000)
    new = vmap.insert(grid + 0.05, np.full_like(grid, 128, dtype=np.uint8))
    assert len(new) == len(vmap) == 3000 < len(grid)
    assert len(vmap.insert(grid + 0.05, np.zeros_like(grid, dtype=np.uint8))) == 0  # budget reached
    assert len(vmap.insert(np.zeros((0, 3)), np.zeros((0, 3), np.uint8))) == 0
