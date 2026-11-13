"""Tests for basir/api/rgbd.py: depth decoding, the depth PNG and the browser point format."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from basir.api.protocol import FrameHeader
from basir.api.rgbd import decode_colour, decode_depth, depth_png, pack_points, unpack_points


def test_decode_depth_applies_the_scale() -> None:
    raw = np.array([[0, 1000], [1500, 4000]], "<u2").tobytes()
    header = FrameHeader(seq=0, timestamp=0.0, depth={"width": 2, "height": 2, "scale": 0.001})
    assert np.allclose(decode_depth(raw, header), [[0, 1.0], [1.5, 4.0]])
    with pytest.raises(ValueError):
        decode_depth(raw, FrameHeader(seq=0, timestamp=0.0))


def test_depth_png_marks_missing_depth_dark() -> None:
    png = depth_png(np.array([[0.0, 0.5], [2.0, 9.0]], np.float32), max_m=4.0)
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (2, 2, 3) and tuple(image[0, 0]) == (24, 24, 24)
    assert tuple(image[0, 1]) != tuple(image[1, 1])  # near and far get different colours


def test_decode_colour_round_trip_and_garbage() -> None:
    _, jpeg = cv2.imencode(".jpg", np.full((4, 6, 3), (0, 0, 255), np.uint8))  # BGR red
    rgb = decode_colour(jpeg.tobytes())
    assert rgb.shape == (4, 6, 3) and rgb[0, 0, 0] > 200 and rgb[0, 0, 2] < 50
    with pytest.raises(ValueError):
        decode_colour(b"not a jpeg")


def test_points_message_round_trip() -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [-1.5, 0.0, 0.25]], np.float32)
    rgb = np.array([[1, 2, 3], [250, 128, 0]], np.uint8)
    message = pack_points(xyz, rgb)
    assert message[:1] == b"P" and len(message) == 5 + 2 * 15
    back_xyz, back_rgb = unpack_points(message)
    assert np.array_equal(back_xyz, xyz) and np.array_equal(back_rgb, rgb)
