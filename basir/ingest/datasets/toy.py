"""A tiny synthetic TUM-layout dataset with known geometry, for offline tests."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import shutil

from basir.core.types import CameraIntrinsics
from basir.ingest.datasets.layout import associate_tum_dataset, write_dataset_metadata
from basir.ingest.datasets.tum_format import TUM_DEPTH_FACTOR, write_tum_file
from utilities.config import get_settings
from utilities.utility import write_png_gray16, write_png_rgb

logger = logging.getLogger("basir.datasets")

TOY_WIDTH, TOY_HEIGHT = 64, 48
TOY_BASE_DEPTH_M = 1.5
TOY_DEPTH_STEP_M = 0.05
_TOY_COLORS = [(100, 150, 200), (120, 170, 220), (140, 190, 240), (160, 210, 240), (180, 220, 255)]


def toy_intrinsics(width: int = TOY_WIDTH, height: int = TOY_HEIGHT) -> dict[str, float]:
    """The default TUM / Kinect camera (640x480) scaled to the toy image size."""
    c = get_settings().default_camera_intrinsics
    full = CameraIntrinsics(c["fx"], c["fy"], c["cx"], c["cy"], int(c["width"]), int(c["height"]), TUM_DEPTH_FACTOR)
    return asdict(full.scaled(width, height))


def create_toy_dataset(target_dir: Path, num_frames: int = 5, overwrite: bool = False) -> Path:
    """Create a tiny synthetic TUM RGB-D dataset.

    64x48 frames at ~30 FPS; frame i has constant depth 1.5 m + 0.05 m * i (stored with
    the TUM factor 5000); intrinsics.json holds the matching scaled camera matrix.
    """
    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        if not overwrite:
            logger.info("Toy dataset already exists at %s.", target_dir)
            return target_dir
        shutil.rmtree(target_dir)
    (target_dir / "rgb").mkdir(parents=True)
    (target_dir / "depth").mkdir()

    rgb_lines, depth_lines, gt_lines = [], [], []
    for i in range(num_frames):
        t_rgb = 1305031453.0 + i * 0.033333
        t_depth = t_rgb + 0.0021  # real sensors are a few ms apart
        rgb_rel, depth_rel = f"rgb/{t_rgb:.6f}.png", f"depth/{t_depth:.6f}.png"
        depth_m = TOY_BASE_DEPTH_M + i * TOY_DEPTH_STEP_M
        write_png_rgb(target_dir / rgb_rel, TOY_WIDTH, TOY_HEIGHT, _TOY_COLORS[i % len(_TOY_COLORS)])
        write_png_gray16(target_dir / depth_rel, TOY_WIDTH, TOY_HEIGHT, round(depth_m * TUM_DEPTH_FACTOR))
        rgb_lines.append(f"{t_rgb:.6f} {rgb_rel}")
        depth_lines.append(f"{t_depth:.6f} {depth_rel}")
        gt_lines.append(f"{t_rgb:.6f} {0.05 * i:.4f} {0.01 * i:.4f} {0.5 + 0.02 * i:.4f} 0 0 0 1")

    write_tum_file(target_dir / "rgb.txt", "# timestamp filename", rgb_lines)
    write_tum_file(target_dir / "depth.txt", "# timestamp filename", depth_lines)
    write_tum_file(target_dir / "groundtruth.txt", "# timestamp tx ty tz qx qy qz qw", gt_lines)
    (target_dir / "intrinsics.json").write_text(json.dumps(toy_intrinsics(), indent=2))

    associate_tum_dataset(target_dir)
    write_dataset_metadata(
        target_dir,
        target_dir.name,
        source="synthetic_toy_generator",
        description=f"Synthetic {num_frames}-frame TUM-compatible dataset for offline unit tests",
    )
    return target_dir
