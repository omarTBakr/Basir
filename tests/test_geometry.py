"""Tests for basir/core/geometry.py and basir/core/types.py (PLAN.md: geometry tests are mandatory)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from basir.core.geometry import (
    backproject,
    depth_consistency,
    depth_to_points,
    frame_to_world_points,
    invert_transform,
    make_transform,
    project,
    quaternion_to_matrix,
    transform_points,
)
from basir.core.types import CameraIntrinsics, Frame, Pose

RNG = np.random.default_rng(42)
INTR = CameraIntrinsics(fx=517.3, fy=516.5, cx=318.6, cy=255.3, width=640, height=480)
K = INTR.K


def random_pose(rng: np.random.Generator = RNG) -> np.ndarray:
    return make_transform(Rotation.random(random_state=rng.integers(1 << 31)).as_matrix(), rng.normal(size=3))


# ---------------------------------------------------------------------------
# Rotations and transforms
# ---------------------------------------------------------------------------


def test_identity_quaternion() -> None:
    assert np.allclose(quaternion_to_matrix(0, 0, 0, 1), np.eye(3))


@pytest.mark.parametrize(
    "axis, expected",
    [
        ("x", [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        ("y", [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        ("z", [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ],
)
def test_90_degrees_about_each_axis(axis: str, expected: list[list[int]]) -> None:
    s = math.sin(math.pi / 4)
    q = {"x": (s, 0, 0, s), "y": (0, s, 0, s), "z": (0, 0, s, s)}[axis]
    assert np.allclose(quaternion_to_matrix(*q), expected)


def test_quaternion_matches_scipy_and_sign_invariance() -> None:
    for _ in range(50):
        q = RNG.normal(size=4)
        R = quaternion_to_matrix(*q)
        assert np.allclose(R, Rotation.from_quat(q).as_matrix())  # scipy uses x, y, z, w too
        assert np.allclose(R, quaternion_to_matrix(*(-q)))
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_unnormalised_quaternion_is_normalised() -> None:
    assert np.allclose(quaternion_to_matrix(0, 0, 0, 5), np.eye(3))


def test_zero_quaternion_raises() -> None:
    with pytest.raises(ValueError):
        quaternion_to_matrix(0, 0, 0, 0)


def test_invert_transform_matches_numpy_and_roundtrips() -> None:
    for _ in range(20):
        T = random_pose()
        assert np.allclose(invert_transform(T), np.linalg.inv(T))
        assert np.allclose(T @ invert_transform(T), np.eye(4))
        points = RNG.normal(size=(10, 3))
        assert np.allclose(transform_points(invert_transform(T), transform_points(T, points)), points)


def test_transform_single_point() -> None:
    T = make_transform(np.eye(3), [1, 2, 3])
    assert np.allclose(transform_points(T, [0, 0, 0]), [1, 2, 3])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_project_backproject_roundtrip() -> None:
    u = RNG.uniform(0, 640, 500)
    v = RNG.uniform(0, 480, 500)
    z = RNG.uniform(0.3, 6.0, 500)
    pu, pv, pz = project(backproject(u, v, z, K), K)
    assert np.allclose(pu, u) and np.allclose(pv, v) and np.allclose(pz, z)


def test_backproject_project_roundtrip() -> None:
    P = np.column_stack([RNG.uniform(-2, 2, 200), RNG.uniform(-2, 2, 200), RNG.uniform(0.5, 5, 200)])
    assert np.allclose(backproject(*project(P, K), K), P)


def test_principal_point_is_on_optical_axis() -> None:
    assert np.allclose(backproject(INTR.cx, INTR.cy, 2.0, K), [0, 0, 2.0])


def test_axis_directions() -> None:
    """x right, y down: a pixel right of / below the centre has positive x / y."""
    right = backproject(INTR.cx + 100, INTR.cy, 1.0, K)
    below = backproject(INTR.cx, INTR.cy + 100, 1.0, K)
    assert right[0] > 0 and abs(right[1]) < 1e-12
    assert below[1] > 0 and abs(below[0]) < 1e-12


def test_points_behind_camera_are_not_visible() -> None:
    u, v, _ = project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), K)
    assert np.isnan(u[:2]).all() and np.isnan(v[:2]).all()
    assert np.isclose(u[2], INTR.cx)


# ---------------------------------------------------------------------------
# Depth images
# ---------------------------------------------------------------------------


def test_depth_to_points_flat_plane_and_invalid_pixels() -> None:
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    depth[0, :] = 0.0  # no reading
    depth[1, :] = 20.0  # beyond max_depth
    points, rows, _ = depth_to_points(depth, K)
    assert len(points) == 46 * 64
    assert np.allclose(points[:, 2], 2.0)
    assert rows.min() == 2


def test_depth_to_points_stride() -> None:
    depth = np.full((48, 64), 1.0, dtype=np.float32)
    points, rows, _ = depth_to_points(depth, K, stride=4)
    assert len(points) == 12 * 16
    assert set(np.unique(rows)) == set(range(0, 48, 4))


def _frame(depth: np.ndarray, T: np.ndarray, intr: CameraIntrinsics = INTR, idx: int = 0) -> Frame:
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    return Frame(id=idx, timestamp=float(idx), rgb=rgb, intrinsics=intr, depth=depth, pose=Pose(T))


def test_frame_to_world_points_uses_pose_camera_to_world() -> None:
    depth = np.full((480, 640), 1.0, dtype=np.float32)
    T = make_transform(np.eye(3), [10.0, 0.0, 0.0])  # camera 10 m along world x
    points, colors = frame_to_world_points(_frame(depth, T), stride=40)
    assert np.allclose(points[:, 2], 1.0)
    assert np.isclose(points[:, 0].mean(), 10.0, atol=0.05)
    assert (colors[:, 0] == 200).all()


def test_frame_without_pose_raises() -> None:
    frame = Frame(id=0, timestamp=0, rgb=np.zeros((2, 2, 3), np.uint8), intrinsics=INTR, depth=np.ones((2, 2)))
    with pytest.raises(ValueError):
        frame_to_world_points(frame)


def test_two_view_consistency_of_a_world_point() -> None:
    """A world point seen from two poses lands where each camera expects it."""
    world_point = np.array([0.5, -0.2, 3.0])
    for _ in range(10):
        T = random_pose()
        p_cam = transform_points(invert_transform(T), world_point)
        if p_cam[2] <= 0:
            continue
        u, v, z = project(p_cam, K)
        back = transform_points(T, backproject(u, v, z, K))
        assert np.allclose(back, world_point)


# ---------------------------------------------------------------------------
# Multi-view depth consistency
# ---------------------------------------------------------------------------


def _wall_depth(T_world_cam: np.ndarray, wall_z: float = 3.0) -> np.ndarray:
    """Exact depth image of a world plane z = wall_z seen from a camera looking along +z."""
    v, u = np.mgrid[0:480, 0:640]
    rays = backproject(u, v, np.ones_like(u, dtype=float), K)  # z = 1 rays, camera frame
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    rays_world_z = rays @ R[2, :]
    return ((wall_z - t[2]) / rays_world_z).astype(np.float32)


def test_depth_consistency_is_zero_for_correct_poses() -> None:
    T_a = make_transform(np.eye(3), [0, 0, 0])
    T_b = make_transform(Rotation.from_euler("y", 5, degrees=True).as_matrix(), [0.2, 0.0, 0.1])
    result = depth_consistency(_frame(_wall_depth(T_a), T_a), _frame(_wall_depth(T_b), T_b, idx=1))
    assert result is not None
    assert result.median_cm < 0.5 and result.within_5cm > 0.99


def test_depth_consistency_detects_a_wrong_pose() -> None:
    T_a = make_transform(np.eye(3), [0, 0, 0])
    T_b = make_transform(np.eye(3), [0.0, 0.0, 0.3])
    wrong = make_transform(np.eye(3), [0.0, 0.0, -0.3])  # sign error, the classic bug
    result = depth_consistency(_frame(_wall_depth(T_a), T_a), _frame(_wall_depth(T_b), wrong, idx=1))
    assert result is not None and result.median_cm > 40


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_pose_from_tum() -> None:
    pose = Pose.from_tum(1.5, 1, 2, 3, 0, 0, 0, 1)
    assert pose.timestamp == 1.5
    assert np.allclose(pose.position, [1, 2, 3]) and np.allclose(pose.matrix[:3, :3], np.eye(3))
    assert np.allclose(pose.inverse() @ pose.matrix, np.eye(4))


def test_pose_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        Pose(np.eye(3))


def test_intrinsics_scaled_keeps_centre() -> None:
    full = CameraIntrinsics(525.0, 525.0, 319.5, 239.5, 640, 480)
    small = full.scaled(64, 48)
    assert (small.fx, small.cx, small.cy) == pytest.approx((52.5, 31.5, 23.5))
    assert np.allclose(full.K, [[525, 0, 319.5], [0, 525, 239.5], [0, 0, 1]])


def test_toy_dataset_points_are_at_toy_depth(tmp_path: Path) -> None:
    from basir.ingest.datasets.toy import create_toy_dataset
    from basir.ingest.sources.tum import TUMDatasetSource

    frame = next(iter(TUMDatasetSource(create_toy_dataset(tmp_path / "toy", num_frames=2))))
    points_cam, _, _ = depth_to_points(frame.depth, frame.intrinsics.K)
    assert len(points_cam) == 64 * 48
    assert np.allclose(points_cam[:, 2], 1.5)
