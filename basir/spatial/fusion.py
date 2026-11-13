"""Phase 0 fusion check: fuse a sequence into one cloud, score it against the targets, save the outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d

from basir.core.types import Frame
from basir.spatial.pointcloud import average_consistency, build_cloud, top_down_image

# Phase 0 targets (plan): median residual <= 3 cm and >= 80 % of points within 5 cm.
TARGET_MEDIAN_CM = 3.0
TARGET_WITHIN_5CM = 0.80


def meets_targets(median_cm: float | None, within_5cm: float | None) -> bool:
    """True if a depth-consistency score meets both Phase 0 targets."""
    return median_cm is not None and median_cm <= TARGET_MEDIAN_CM and (within_5cm or 0) >= TARGET_WITHIN_5CM


@dataclass
class CloudResult:
    frames_used: int
    points_raw: int
    points_downsampled: int
    seconds: float
    consistency_median_cm: float | None
    consistency_p90_cm: float | None
    consistency_within_5cm: float | None
    consistency_pairs: int

    @property
    def passes(self) -> bool:
        return meets_targets(self.consistency_median_cm, self.consistency_within_5cm)

    def metrics(self) -> dict[str, float]:
        """The numeric fields plus `passes` (1.0 or 0.0), for MLflow."""
        return {k: v for k, v in asdict(self).items() if v is not None} | {"passes": float(self.passes)}


def fuse(
    frames: list[Frame], pixel_stride: int = 4, voxel: float = 0.02, gap: int = 1
) -> tuple[o3d.geometry.PointCloud, CloudResult]:
    """Build the fused cloud and score its depth consistency over frame pairs `gap` apart."""
    started = time.perf_counter()
    cloud, raw = build_cloud(frames, pixel_stride, voxel)
    score, pairs = average_consistency(frames, gap)
    return cloud, CloudResult(
        frames_used=len(frames),
        points_raw=raw,
        points_downsampled=len(cloud.points),
        seconds=time.perf_counter() - started,
        consistency_median_cm=score.median_cm if score else None,
        consistency_p90_cm=score.p90_cm if score else None,
        consistency_within_5cm=score.within_5cm if score else None,
        consistency_pairs=pairs,
    )


def save_outputs(cloud: o3d.geometry.PointCloud, frames: list[Frame], out_ply: Path) -> list[Path]:
    """Write the cloud (.ply), a top-down image (.topdown.png) and the camera path (.trajectory.csv)."""
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    out_png, out_csv = out_ply.with_suffix(".topdown.png"), out_ply.with_suffix(".trajectory.csv")
    o3d.io.write_point_cloud(out_ply, cloud)
    cv2.imwrite(str(out_png), top_down_image(cloud, frames))
    np.savetxt(
        out_csv,
        np.array([[f.timestamp, *f.pose.position] for f in frames if f.pose is not None]),
        delimiter=",",
        header="timestamp,x,y,z",
        comments="",
        fmt="%.6f",
    )
    return [out_ply, out_png, out_csv]
