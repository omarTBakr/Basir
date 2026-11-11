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
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import open3d as o3d

from basir.core.types import Frame
from basir.ingest.datasets.prepare import dataset_dir
from basir.ingest.sources.tum import TUMDatasetSource
from basir.spatial.fusion import TARGET_MEDIAN_CM, TARGET_WITHIN_5CM, fuse, save_outputs
from utilities.config import get_settings
from utilities.experiment import log_artifact, log_metrics, log_params, start_run

logger = logging.getLogger("visualize_pointcloud")


def trajectory_lines(frames: list[Frame]) -> o3d.geometry.LineSet:
    positions = np.array([f.pose.position for f in frames if f.pose is not None])
    lines = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(positions),
        o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(positions) - 1)]),
    )
    lines.paint_uniform_color([1.0, 0.1, 0.1])
    return lines


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

    with start_run("phase0-pointcloud", run_name=root.name, tags={"dataset": root.name}):
        log_params(
            {
                "dataset": root.name,
                "every": args.every,
                "pixel_stride": args.pixel_stride,
                "voxel": args.voxel,
                "gap": args.gap,
                "intrinsics_source": source.intrinsics_source,
            }
        )
        frames = list(source)
        cloud, result = fuse(frames, args.pixel_stride, args.voxel, args.gap)
        outputs = save_outputs(cloud, frames, out_ply)
        log_metrics(result.metrics())
        for path in outputs:
            log_artifact(path)

    print(
        f"{root.name}: {result.frames_used} frames, {result.points_raw:,} points -> "
        f"{result.points_downsampled:,} after {args.voxel} m voxels ({result.seconds:.1f} s)"
    )
    if result.consistency_median_cm is not None:
        verdict = "PASS" if result.passes else "CHECK"
        print(
            f"Depth consistency over {result.consistency_pairs} pairs: median {result.consistency_median_cm:.2f} cm, "
            f"p90 {result.consistency_p90_cm:.2f} cm, {result.consistency_within_5cm:.0%} within 5 cm -> {verdict} "
            f"(target: median <= {TARGET_MEDIAN_CM} cm, >= {TARGET_WITHIN_5CM:.0%} within 5 cm)"
        )
    print(f"Saved {', '.join(str(p) for p in outputs)}")

    if not args.no_window:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([cloud, trajectory_lines(frames), axes], window_name=f"Basir: {root.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
