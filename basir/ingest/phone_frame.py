"""Turn one phone message (wire header + JPEG + uint16 depth) into a `basir.core.types.Frame`.

This is the only place that crosses from the wire format (`basir.api.protocol`) into Basir's
domain types, so mapping, memory and perception all see the same `Frame`.

Depth scale: two conventions, opposite ways round
--------------------------------------------------
* Wire (`header.depth["scale"]`): **metres per raw unit**, `metres = raw * scale`.
  ARCore DEPTH16 millimetres: 0.001.
* `CameraIntrinsics.depth_scale`: **raw units per metre**, `metres = raw / depth_scale`.
  TUM RGB-D PNGs: 5000.

So `depth_scale = 1 / scale` (`depth_scale_from_wire` / `wire_scale_from_depth_scale`).
`Frame.depth` is always float32 metres, already converted, so most code never needs either.

The intrinsics in the header describe the colour image. When depth is sent, the frame is
built at the depth resolution: the colour is resized to it and the intrinsics are scaled
to it (`CameraIntrinsics.scaled`), so `frame.intrinsics.K` back-projects `frame.depth`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

import cv2
import numpy as np

from basir.core.types import CameraIntrinsics, Frame, Pose


class PhoneHeader(Protocol):
    """The parts of `basir.api.protocol.FrameHeader` this adapter reads (kept structural so
    `basir.ingest` does not import `basir.api`)."""

    @property
    def seq(self) -> int: ...
    @property
    def timestamp(self) -> float: ...
    @property
    def pose(self) -> list[float] | None: ...
    @property
    def intrinsics(self) -> Mapping[str, float] | None: ...
    @property
    def depth(self) -> Mapping[str, float] | None: ...


def depth_scale_from_wire(scale: float) -> float:
    """Wire `depth.scale` (metres per unit) -> `CameraIntrinsics.depth_scale` (units per metre)."""
    if not scale > 0:
        raise ValueError(f"depth scale must be positive, got {scale}")
    return 1.0 / scale


def wire_scale_from_depth_scale(depth_scale: float) -> float:
    """`CameraIntrinsics.depth_scale` (units per metre) -> wire `depth.scale` (metres per unit)."""
    if not depth_scale > 0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")
    return 1.0 / depth_scale


def depth_metres(raw: bytes, width: int, height: int, scale: float) -> np.ndarray:
    """uint16 little-endian depth -> HxW float32 metres (wire convention: metres = raw * scale)."""
    if len(raw) != width * height * 2:
        raise ValueError(f"depth needs {width * height * 2} bytes for {width}x{height}, got {len(raw)}")
    return np.frombuffer(raw, "<u2").reshape(height, width).astype(np.float32) * np.float32(scale)


def decode_rgb(jpeg: bytes) -> np.ndarray:
    """JPEG bytes -> HxWx3 uint8 RGB."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("colour is not a decodable JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frame_from_message(header: PhoneHeader, jpeg: bytes, depth: bytes | None) -> Frame:
    """A `Frame` from one decoded phone message; needs `intrinsics` in the header.

    With depth: depth in metres at its own resolution, colour resized to match, intrinsics
    scaled to it and `depth_scale` = 1 / wire scale. Without depth: colour at its decoded size
    and intrinsics scaled to that size. `pose` becomes a `Pose` when present.
    """
    k = header.intrinsics
    if k is None:
        raise ValueError("frame needs intrinsics")
    rgb = decode_rgb(jpeg)
    colour = CameraIntrinsics(k["fx"], k["fy"], k["cx"], k["cy"], int(k["width"]), int(k["height"]))
    depth_m: np.ndarray | None = None
    if depth is not None:
        if header.depth is None:
            raise ValueError("depth bytes need 'depth' in the header")
        w, h, scale = int(header.depth["width"]), int(header.depth["height"]), float(header.depth["scale"])
        depth_m = depth_metres(depth, w, h, scale)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        intrinsics = replace(colour.scaled(w, h), depth_scale=depth_scale_from_wire(scale))
    else:
        h, w = rgb.shape[:2]
        intrinsics = colour.scaled(w, h)
    pose = None
    if header.pose is not None:
        pose = Pose(np.asarray(header.pose, dtype=np.float64).reshape(4, 4), header.timestamp)
    return Frame(header.seq, header.timestamp, rgb, intrinsics, depth_m, pose)
