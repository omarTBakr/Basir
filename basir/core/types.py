"""Core data types shared by every Basir stage.

Conventions (used everywhere in Basir):
    * Camera frame: OpenCV / TUM optical frame, x right, y down, z forward (metres).
    * Pose: T_world_cam, the 4x4 transform taking camera-frame points to the world frame.
    * Depth: float32 metres, 0 means "no reading".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from basir.core.geometry import invert_transform, make_transform, quaternion_to_matrix


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera parameters plus the factor that turns raw depth values into metres."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 5000.0  # raw depth value / depth_scale = metres (TUM: 5000)

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled(self, width: int, height: int) -> CameraIntrinsics:
        """Intrinsics for the same camera at another image size (pixel-centre convention)."""
        sx, sy = width / self.width, height / self.height
        return replace(
            self,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
            width=width,
            height=height,
        )


@dataclass(frozen=True)
class Pose:
    """Camera pose T_world_cam (camera -> world) at a timestamp."""

    matrix: np.ndarray = field(repr=False)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.matrix.shape != (4, 4):
            raise ValueError(f"Pose matrix must be 4x4, got {self.matrix.shape}")

    @classmethod
    def from_tum(cls, t: float, tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> Pose:
        """From a TUM groundtruth line: translation + unit quaternion (x, y, z, w)."""
        return cls(make_transform(quaternion_to_matrix(qx, qy, qz, qw), [tx, ty, tz]), t)

    def inverse(self) -> np.ndarray:
        """T_cam_world (world -> camera)."""
        return invert_transform(self.matrix)

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return self.matrix[:3, 3]


@dataclass
class Frame:
    """One camera frame: colour, optional depth and pose, and the camera that took it."""

    id: int
    timestamp: float
    rgb: np.ndarray  # HxWx3 uint8, RGB order
    intrinsics: CameraIntrinsics
    depth: np.ndarray | None = None  # HxW float32 metres, 0 = no reading
    pose: Pose | None = None
