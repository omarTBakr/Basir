"""Fused point clouds: build one from posed RGB-D frames, score how consistent it is, and draw it from above."""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

from basir.core.geometry import DepthConsistency, depth_consistency, frame_to_world_points
from basir.core.types import Frame


def build_cloud(frames: list[Frame], pixel_stride: int = 4, voxel: float = 0.02) -> tuple[o3d.geometry.PointCloud, int]:
    """Merge the frames' coloured world points into one cloud, voxel-downsampled (voxel <= 0: off).

    Returns (cloud, number of points before downsampling).
    """
    points, colors = [], []
    for frame in frames:
        p, c = frame_to_world_points(frame, stride=pixel_stride)
        points.append(p)
        colors.append(c)
    all_points = np.concatenate(points)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors).astype(np.float64) / 255.0)
    return (cloud.voxel_down_sample(voxel) if voxel > 0 else cloud), len(all_points)


def average_consistency(frames: list[Frame], gap: int = 1) -> tuple[DepthConsistency | None, int]:
    """Depth consistency averaged over frame pairs `gap` apart, weighted by compared points.

    Returns (average, number of pairs that overlapped); (None, 0) if no pair did.
    """
    results = [r for i in range(len(frames) - gap) if (r := depth_consistency(frames[i], frames[i + gap]))]
    if not results:
        return None, 0
    weights = np.array([r.n_points for r in results], dtype=float)
    return (
        DepthConsistency(
            median_cm=float(np.average([r.median_cm for r in results], weights=weights)),
            p90_cm=float(np.average([r.p90_cm for r in results], weights=weights)),
            within_5cm=float(np.average([r.within_5cm for r in results], weights=weights)),
            n_points=int(weights.sum()),
        ),
        len(results),
    )


def top_down_image(cloud: o3d.geometry.PointCloud, frames: list[Frame], size: int = 800) -> np.ndarray:
    """Bird's-eye view (world x-y plane, TUM's floor plane) of the cloud plus the camera path, as BGR."""
    points = np.asarray(cloud.points)
    colors = (np.asarray(cloud.colors)[:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
    traj = np.array([f.pose.position for f in frames if f.pose is not None])
    lo = np.minimum(points[:, :2].min(0), traj[:, :2].min(0))
    hi = np.maximum(points[:, :2].max(0), traj[:, :2].max(0))
    scale = (size - 20) / max(float((hi - lo).max()), 1e-6)

    def to_px(xy: np.ndarray) -> np.ndarray:
        px = ((xy - lo) * scale + 10).astype(int)
        px[:, 1] = size - px[:, 1]  # image y grows downwards
        return np.clip(px, 0, size - 1)

    image = np.zeros((size, size, 3), np.uint8)
    order = np.argsort(points[:, 2])  # draw low points first so higher surfaces stay visible
    px = to_px(points[order, :2])
    image[px[:, 1], px[:, 0]] = colors[order]
    cv2.polylines(image, [to_px(traj[:, :2]).reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    return image
