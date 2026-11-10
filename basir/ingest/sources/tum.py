"""Frame source for TUM RGB-D sequences (and anything saved in the same layout).

Reads associations_with_gt.txt (written by scripts/download_dataset.py): one line per
frame with the RGB file, the depth file and the ground-truth pose T_world_cam.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from basir.core.types import CameraIntrinsics, Frame, Pose
from basir.ingest.datasets.tum_format import TUM_DEPTH_FACTOR, calibration_for
from utilities.config import get_settings


@dataclass(frozen=True)
class _Entry:
    t_rgb: float
    rgb_path: str
    depth_path: str
    pose: Pose | None


class TUMDatasetSource:
    """Iterate a TUM-layout dataset as `Frame`s with depth (metres) and ground-truth pose."""

    def __init__(
        self,
        root: str | Path,
        stride: int = 1,
        max_frames: int | None = None,
        require_pose: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self._intrinsics, self.intrinsics_source = _load_intrinsics(self.root)
        entries = _read_associations(self.root)
        if require_pose:
            entries = [e for e in entries if e.pose is not None]
        if not entries:
            raise ValueError(
                f"No usable frames in {self.root}: expected associations_with_gt.txt "
                "(run scripts/download_dataset.py on it to create one)"
            )
        self._entries = entries[::stride][:max_frames]

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Frame]:
        for index, entry in enumerate(self._entries):
            yield self._load(index, entry)

    def close(self) -> None:  # nothing held open; present for the FrameSource interface
        pass

    def _load(self, index: int, entry: _Entry) -> Frame:
        bgr = cv2.imread(str(self.root / entry.rgb_path), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(str(self.root / entry.depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw_depth is None:
            raise OSError(f"Could not read frame {index} ({entry.rgb_path}, {entry.depth_path})")
        return Frame(
            id=index,
            timestamp=entry.t_rgb,
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth=raw_depth.astype(np.float32) / self._intrinsics.depth_scale,
            intrinsics=self._intrinsics,
            pose=entry.pose,
        )


def _load_intrinsics(root: Path) -> tuple[CameraIntrinsics, str]:
    calib, source = calibration_for(root)
    if calib is None:
        calib, source = get_settings().default_camera_intrinsics, "settings_default"
    return (
        CameraIntrinsics(
            fx=calib["fx"],
            fy=calib["fy"],
            cx=calib["cx"],
            cy=calib["cy"],
            width=int(calib["width"]),
            height=int(calib["height"]),
            depth_scale=calib.get("depth_scale", TUM_DEPTH_FACTOR),
        ),
        source,
    )


def _read_associations(root: Path) -> list[_Entry]:
    """Rows of associations_with_gt.txt, or associations.txt without poses."""
    with_gt = root / "associations_with_gt.txt"
    path = with_gt if with_gt.is_file() else root / "associations.txt"
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#") or len(parts) < 4:
            continue
        pose = None
        if len(parts) >= 12:  # t_rgb rgb t_depth depth t_gt tx ty tz qx qy qz qw
            t_gt, *values = (float(p) for p in parts[4:12])
            pose = Pose.from_tum(t_gt, *values)
        entries.append(_Entry(float(parts[0]), parts[1], parts[3], pose))
    return entries
