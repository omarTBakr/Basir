"""Tests for basir/ingest/phone_frame.py: one phone message -> one `Frame`, with one depth-scale convention."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from basir.api.protocol import FrameHeader, decode_frame, encode_frame
from basir.core.geometry import depth_to_points
from basir.core.types import CameraIntrinsics
from basir.ingest.phone_frame import (
    depth_metres,
    depth_scale_from_wire,
    frame_from_message,
    wire_scale_from_depth_scale,
)

COLOUR = CameraIntrinsics(fx=500.0, fy=500.0, cx=319.5, cy=239.5, width=640, height=480)


def jpeg(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), np.uint8)
    image[:, : width // 2] = (0, 0, 255)  # left half red (BGR)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def header(depth: dict[str, float] | None = None, pose: list[float] | None = None) -> FrameHeader:
    k = COLOUR
    intrinsics = {"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy, "width": k.width, "height": k.height}
    return FrameHeader(seq=9, timestamp=12.5, pose=pose, intrinsics=intrinsics, depth=depth)


@pytest.mark.parametrize(
    ("name", "depth_scale"),
    [("ARCore DEPTH16, millimetres", 1000.0), ("TUM PNG, 1/5000 m", 5000.0)],
)
def test_depth_round_trips_through_the_wire_in_both_conventions(name: str, depth_scale: float) -> None:
    """A depth image in metres -> raw uint16 (CameraIntrinsics convention) -> wire -> Frame: same metres."""
    metres = np.array([[0.5, 1.0, 2.0, 0.0], [1.25, 3.0, 4.0, 6.5]], np.float32)  # 0 = no reading
    raw = np.round(metres * depth_scale).astype("<u2")  # raw = metres * depth_scale (TUM convention)
    wire_scale = wire_scale_from_depth_scale(depth_scale)  # metres per unit: 0.001 (ARCore) or 0.0002 (TUM)
    assert wire_scale == pytest.approx(1 / depth_scale)
    message = encode_frame(header({"width": 4, "height": 2, "scale": wire_scale}), jpeg(64, 48), raw.tobytes())
    frame = frame_from_message(*decode_frame(message))

    assert frame.depth is not None and frame.depth.dtype == np.float32
    np.testing.assert_allclose(frame.depth, metres, atol=0.5 / depth_scale)
    assert frame.intrinsics.depth_scale == pytest.approx(depth_scale)
    assert depth_scale_from_wire(wire_scale) == pytest.approx(depth_scale)
    # the Frame convention reproduces the same metres from the raw values
    np.testing.assert_allclose(raw.astype(np.float32) / frame.intrinsics.depth_scale, frame.depth, rtol=1e-6)


def test_the_frame_is_at_the_depth_resolution_with_scaled_intrinsics() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    raw = np.full((120, 160), 1000, "<u2")  # 1 m everywhere
    message = encode_frame(
        header({"width": 160, "height": 120, "scale": 0.001}, pose.ravel().tolist()), jpeg(640, 480), raw.tobytes()
    )
    frame = frame_from_message(*decode_frame(message))

    assert frame.id == 9 and frame.timestamp == 12.5
    assert frame.rgb.shape == (120, 160, 3) and frame.depth is not None and frame.depth.shape == (120, 160)
    assert frame.intrinsics == CameraIntrinsics(
        fx=125.0, fy=125.0, cx=79.5, cy=59.5, width=160, height=120, depth_scale=1000.0
    )
    assert frame.pose is not None and np.array_equal(frame.pose.matrix, pose) and frame.pose.timestamp == 12.5
    assert frame.rgb[60, 10, 0] > 200 and frame.rgb[60, 150, 0] < 50  # left half red, after the resize
    # the principal point back-projects onto the optical axis
    points, rows, cols = depth_to_points(frame.depth, frame.intrinsics.K)
    centre = points[(rows == 60) & (cols == 80)][0]
    assert centre[2] == pytest.approx(1.0) and abs(centre[0]) < 0.005 and abs(centre[1]) < 0.005


def test_colour_only_frames_keep_the_colour_resolution() -> None:
    frame = frame_from_message(header(), jpeg(320, 240), None)
    assert frame.depth is None and frame.pose is None
    assert (frame.intrinsics.width, frame.intrinsics.height) == (320, 240)
    assert frame.intrinsics.fx == pytest.approx(250.0)


def test_bad_inputs_are_refused() -> None:
    no_intrinsics = FrameHeader(seq=1, timestamp=0.0)
    with pytest.raises(ValueError, match="intrinsics"):
        frame_from_message(no_intrinsics, jpeg(8, 8), None)
    with pytest.raises(ValueError, match="'depth'"):
        frame_from_message(header(), jpeg(8, 8), b"\x00\x00")
    with pytest.raises(ValueError, match="JPEG"):
        frame_from_message(header(), b"not a jpeg", None)
    with pytest.raises(ValueError, match="bytes"):
        depth_metres(b"\x00" * 6, 2, 2, 0.001)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            depth_scale_from_wire(bad)
        with pytest.raises(ValueError):
            wire_scale_from_depth_scale(bad)
