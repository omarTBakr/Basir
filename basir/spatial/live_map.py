"""Incremental map for a moving phone: a keyframe gate and a voxel map that reports what is new.

    gate, vmap = KeyframeGate(0.05, 5.0), VoxelMap(voxel=0.02, max_points=2_000_000)
    if gate.accept(pose):
        new = vmap.insert(points_world, colours)   # indices of voxels created by this keyframe
        xyz, rgb = vmap.points(new)                # send these to the viewer

Pure numpy; the server wraps it in `basir.api.mapping`.
"""

from __future__ import annotations

import math

import numpy as np

_BITS = 21  # per axis: +-2^20 voxels, about +-21 km at 2 cm
_OFFSET = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


class KeyframeGate:
    """Accept a pose when the camera moved at least `min_move_m` or turned `min_turn_deg` since the last one."""

    def __init__(self, min_move_m: float, min_turn_deg: float) -> None:
        self.min_move_m = min_move_m
        self.min_turn_rad = math.radians(min_turn_deg)
        self.last: np.ndarray | None = None

    def accept(self, pose: np.ndarray) -> bool:
        """`pose` is T_world_cam (4x4); the first pose is always accepted."""
        if self.last is not None:
            moved = float(np.linalg.norm(pose[:3, 3] - self.last[:3, 3]))
            relative = self.last[:3, :3].T @ pose[:3, :3]
            turned = math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1) / 2)))
            if moved < self.min_move_m and turned < self.min_turn_rad:
                return False
        self.last = pose.copy()
        return True

    def reset(self) -> None:
        self.last = None


class VoxelMap:
    """Points averaged per voxel; `insert` returns the indices of voxels it created."""

    def __init__(self, voxel: float, max_points: int) -> None:
        self.voxel = voxel
        self.max_points = max_points
        self._index: dict[int, int] = {}
        self._size = 0
        self._xyz = np.zeros((1024, 3), np.float64)  # running sums, grown by doubling
        self._rgb = np.zeros((1024, 3), np.float64)
        self._count = np.zeros(1024, np.int64)

    def __len__(self) -> int:
        return self._size

    def _reserve(self, size: int) -> None:
        if size <= len(self._count):
            return
        capacity = max(size, 2 * len(self._count))
        for name in ("_xyz", "_rgb", "_count"):
            old = getattr(self, name)
            new = np.zeros((capacity, *old.shape[1:]), old.dtype)
            new[: self._size] = old[: self._size]
            setattr(self, name, new)

    def insert(self, points: np.ndarray, colours: np.ndarray) -> np.ndarray:
        """Add world points (N x 3) with uint8 colours (N x 3); return the new voxels' indices."""
        if len(points) == 0:
            return np.zeros(0, np.int64)
        cells = np.floor(points / self.voxel).astype(np.int64) + _OFFSET
        keys = (cells[:, 0] & _MASK) << (2 * _BITS) | (cells[:, 1] & _MASK) << _BITS | (cells[:, 2] & _MASK)
        unique, inverse = np.unique(keys, return_inverse=True)
        sums = np.zeros((len(unique), 3))
        np.add.at(sums, inverse, points)
        rgb = np.zeros((len(unique), 3))
        np.add.at(rgb, inverse, colours.astype(np.float64))
        counts = np.bincount(inverse, minlength=len(unique))

        slots = np.fromiter((self._index.get(int(k), -1) for k in unique), np.int64, len(unique))
        known = slots >= 0
        self._xyz[slots[known]] += sums[known]
        self._rgb[slots[known]] += rgb[known]
        self._count[slots[known]] += counts[known]

        fresh = np.flatnonzero(~known)[: max(0, self.max_points - self._size)]
        start, end = self._size, self._size + len(fresh)
        self._reserve(end)
        for offset, i in enumerate(fresh):
            self._index[int(unique[i])] = start + offset
        self._xyz[start:end] = sums[fresh]
        self._rgb[start:end] = rgb[fresh]
        self._count[start:end] = counts[fresh]
        self._size = end
        return np.arange(start, end)

    def points(self, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Mean position (float32) and colour (uint8) of the given voxels (all when None)."""
        sel = slice(0, self._size) if indices is None else indices
        n = self._count[sel][:, None]
        return (self._xyz[sel] / n).astype(np.float32), np.clip(self._rgb[sel] / n, 0, 255).astype(np.uint8)
