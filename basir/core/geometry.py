"""3D geometry for Basir: rotations, rigid transforms, projection and depth.

Pure numpy, vectorised, no I/O. Conventions (see basir.core.types):
    * Camera frame: x right, y down, z forward.
    * T_world_cam takes camera-frame points to the world frame.
    * Points are N x 3 arrays; a single point may be a length-3 vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from basir.core.types import Frame

# ---------------------------------------------------------------------------
# Rotations and rigid transforms
# ---------------------------------------------------------------------------


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """3x3 rotation from a quaternion in (x, y, z, w) order (TUM, scipy). Normalised first."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero-length quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray | list[float]) -> np.ndarray:
    """4x4 rigid transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform: [R | t]^-1 = [R^T | -R^T t]."""
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to N x 3 points (or one 3-vector)."""
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def backproject(u: np.ndarray | float, v: np.ndarray | float, z: np.ndarray | float, K: np.ndarray) -> np.ndarray:
    """Pixel coordinates + depth (metres) -> camera-frame points (N x 3, or a 3-vector)."""
    u, v, z = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64), np.asarray(z, dtype=np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def project(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame points -> (u, v, z). Points with z <= 0 get u = v = NaN (not visible)."""
    P = np.asarray(points_cam, dtype=np.float64)
    z = P[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(z > 0, K[0, 0] * P[..., 0] / z + K[0, 2], np.nan)
        v = np.where(z > 0, K[1, 1] * P[..., 1] / z + K[1, 2], np.nan)
    return u, v, z


# ---------------------------------------------------------------------------
# Depth images
# ---------------------------------------------------------------------------


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth image (metres) -> camera-frame points of valid pixels.

    Returns (points N x 3, rows N, cols N); rows / cols index the image so callers can
    fetch colours. Zero and out-of-range depths are skipped; `stride` subsamples pixels.
    """
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[rows, cols]
    valid = (z > min_depth) & (z < max_depth)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    return backproject(cols, rows, z, K), rows, cols


def frame_to_world_points(
    frame: Frame, stride: int = 1, min_depth: float = 0.1, max_depth: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame points and their RGB colours (N x 3 uint8) for one posed RGB-D frame."""
    if frame.depth is None or frame.pose is None:
        raise ValueError(f"Frame {frame.id} needs depth and a pose")
    points_cam, rows, cols = depth_to_points(frame.depth, frame.intrinsics.K, stride, min_depth, max_depth)
    return transform_points(frame.pose.matrix, points_cam), frame.rgb[rows, cols]


# ---------------------------------------------------------------------------
# Multi-view consistency (Phase 0 quality check)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthConsistency:
    """How well frame i's depth agrees with frame j's depth after moving it with the poses."""

    median_cm: float
    p90_cm: float
    within_5cm: float  # fraction of compared points with |residual| <= 5 cm
    n_points: int


def depth_consistency(
    frame_i: Frame, frame_j: Frame, stride: int = 4, max_depth: float = 4.0
) -> DepthConsistency | None:
    """Project frame i's points into frame j and compare against frame j's depth.

    With correct poses, intrinsics and conventions the residual is sensor noise
    (a few cm); a pose or axis bug shows up as tens of cm. Returns None if the two
    views share no valid pixels.
    """
    points_world, _ = frame_to_world_points(frame_i, stride=stride, max_depth=max_depth)
    if frame_j.depth is None or frame_j.pose is None or len(points_world) == 0:
        return None

    u, v, z = project(transform_points(frame_j.pose.inverse(), points_world), frame_j.intrinsics.K)
    h, w = frame_j.depth.shape
    inside = np.isfinite(u) & np.isfinite(v)
    inside &= (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    cols, rows = np.round(u[inside]).astype(int), np.round(v[inside]).astype(int)
    observed = frame_j.depth[rows, cols]
    seen = (observed > 0.1) & (observed < max_depth)
    if not seen.any():
        return None

    residual_cm = np.abs(z[inside][seen] - observed[seen]) * 100.0
    return DepthConsistency(
        median_cm=float(np.median(residual_cm)),
        p90_cm=float(np.percentile(residual_cm, 90)),
        within_5cm=float(np.mean(residual_cm <= 5.0)),
        n_points=int(residual_cm.size),
    )
