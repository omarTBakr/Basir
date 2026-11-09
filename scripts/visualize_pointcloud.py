#!/usr/bin/env python3
"""Phase 0 check: fuse a sequence's depth + poses into one coloured point cloud.

If poses, intrinsics and conventions are right, the result is one clean room. If
they're wrong, walls appear doubled or smeared. The depth-consistency score (how far
apart overlapping frames' surfaces are, in cm) puts a number on it. Everything is
logged to MLflow (experiment `phase0-pointcloud`): params, metrics, the .ply file,
a top-down image and the camera trajectory.

    python scripts/visualize_pointcloud.py fr1/room
    python scripts/visualize_pointcloud.py fr1/desk --every 10 --voxel 0.01
    python scripts/visualize_pointcloud.py fr1/room --no-window       # save files only
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from basir.core.geometry import DepthConsistency, depth_consistency, frame_to_world_points  # noqa: E402
from basir.core.types import Frame  # noqa: E402
from basir.ingest.sources.tum import TUMDatasetSource  # noqa: E402
from scripts.download_dataset import dataset_dir  # noqa: E402
from utilities.config import get_settings  # noqa: E402
from utilities.experiment import log_artifact, log_metrics, log_params, start_run  # noqa: E402

logger = logging.getLogger("visualize_pointcloud")

# Phase 0 targets (plan): median residual <= 3 cm and >= 80 % of points within 5 cm.
TARGET_MEDIAN_CM = 3.0
TARGET_WITHIN_5CM = 0.80


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
        return (
            self.consistency_median_cm is not None
            and self.consistency_median_cm <= TARGET_MEDIAN_CM
            and (self.consistency_within_5cm or 0) >= TARGET_WITHIN_5CM
        )


def build_cloud(frames: list[Frame], pixel_stride: int, voxel: float) -> tuple[o3d.geometry.PointCloud, int]:
    """Merge the frames' world points into one voxel-downsampled cloud; returns (cloud, raw count)."""
    points, colors = [], []
    for frame in frames:
        p, c = frame_to_world_points(frame, stride=pixel_stride)
        points.append(p)
        colors.append(c)
    all_points = np.concatenate(points)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors).astype(np.float64) / 255.0)
    return (cloud.voxel_down_sample(voxel) if voxel > 0 else cloud), len(all_points)


def consistency(frames: list[Frame], gap: int) -> tuple[DepthConsistency | None, int]:
    """Average depth consistency over frame pairs `gap` apart (in the selected frames)."""
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


def trajectory_lines(frames: list[Frame]) -> o3d.geometry.LineSet:
    positions = np.array([f.pose.position for f in frames])
    lines = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(positions),
        o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(positions) - 1)]),
    )
    lines.paint_uniform_color([1.0, 0.1, 0.1])
    return lines


def top_down_image(cloud: o3d.geometry.PointCloud, frames: list[Frame], size: int = 800) -> np.ndarray:
    """Bird's-eye view (world x-y plane, TUM's floor plane) of the cloud plus the trajectory."""
    points = np.asarray(cloud.points)
    colors = (np.asarray(cloud.colors)[:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
    traj = np.array([f.pose.position for f in frames])
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Preset key (e.g. fr1/room) or dataset folder.")
    parser.add_argument("--every", type=int, default=15, help="Use every Nth frame (default 15, ~2 per second).")
    parser.add_argument("--pixel-stride", type=int, default=4, help="Use every Nth pixel in each frame.")
    parser.add_argument("--voxel", type=float, default=0.02, help="Voxel size in metres for downsampling; 0 = off.")
    parser.add_argument("--gap", type=int, default=1, help="Consistency: compare selected frames this many apart.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save", type=Path, default=None, help="Output .ply (default: outputs/<dataset>.ply).")
    parser.add_argument("--no-window", action="store_true", help="Save files only; do not open a viewer.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    try:
        root = dataset_dir(args.dataset)
        source = TUMDatasetSource(root, stride=args.every, max_frames=args.max_frames)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    out_ply = args.save or get_settings().outputs_dir / f"{root.name}.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    out_png, out_csv = out_ply.with_suffix(".topdown.png"), out_ply.with_suffix(".trajectory.csv")

    with start_run("phase0-pointcloud", run_name=root.name, tags={"dataset": root.name}):
        log_params({"dataset": root.name, "every": args.every, "pixel_stride": args.pixel_stride,
                    "voxel": args.voxel, "gap": args.gap, "intrinsics_source": source.intrinsics_source})
        started = time.perf_counter()
        frames = list(source)
        cloud, raw = build_cloud(frames, args.pixel_stride, args.voxel)
        score, pairs = consistency(frames, args.gap)
        result = CloudResult(
            frames_used=len(frames),
            points_raw=raw,
            points_downsampled=len(cloud.points),
            seconds=time.perf_counter() - started,
            consistency_median_cm=score.median_cm if score else None,
            consistency_p90_cm=score.p90_cm if score else None,
            consistency_within_5cm=score.within_5cm if score else None,
            consistency_pairs=pairs,
        )

        o3d.io.write_point_cloud(str(out_ply), cloud)
        cv2.imwrite(str(out_png), top_down_image(cloud, frames))
        np.savetxt(out_csv, np.array([[f.timestamp, *f.pose.position] for f in frames]),
                   delimiter=",", header="timestamp,x,y,z", comments="", fmt="%.6f")
        log_metrics({k: v for k, v in asdict(result).items() if v is not None} | {"passes": float(result.passes)})
        for path in (out_ply, out_png, out_csv):
            log_artifact(path)

    print(f"{root.name}: {result.frames_used} frames, {result.points_raw:,} points -> "
          f"{result.points_downsampled:,} after {args.voxel} m voxels ({result.seconds:.1f} s)")
    if score:
        verdict = "PASS" if result.passes else "CHECK"
        print(f"Depth consistency over {pairs} pairs: median {score.median_cm:.2f} cm, "
              f"p90 {score.p90_cm:.2f} cm, {score.within_5cm:.0%} within 5 cm -> {verdict} "
              f"(target: median <= {TARGET_MEDIAN_CM} cm, >= {TARGET_WITHIN_5CM:.0%} within 5 cm)")
    print(f"Saved {out_ply}, {out_png.name}, {out_csv.name}")

    if not args.no_window:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([cloud, trajectory_lines(frames), axes], window_name=f"Basir: {root.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
