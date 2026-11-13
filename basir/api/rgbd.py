"""Decode what phones send (JPEG colour, uint16 depth) and encode what browsers get (depth PNG, point batches)."""

from __future__ import annotations

import struct

import cv2
import numpy as np

from basir.api.protocol import FrameHeader

POINTS_TAG = b"P"  # binary map message: tag | u32 count | float32 xyz * count | uint8 rgb * count
NO_DEPTH_BGR = (24, 24, 24)


def decode_colour(jpeg: bytes) -> np.ndarray:
    """JPEG bytes -> RGB uint8 image."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("colour is not a decodable JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_depth(raw: bytes, header: FrameHeader) -> np.ndarray:
    """uint16 little-endian depth (described by `header.depth`) -> float32 metres, 0 = no reading."""
    if header.depth is None:
        raise ValueError("frame has no depth")
    width, height = int(header.depth["width"]), int(header.depth["height"])
    values = np.frombuffer(raw, "<u2").reshape(height, width)
    return values.astype(np.float32) * float(header.depth["scale"])


def depth_png(depth_m: np.ndarray, max_m: float) -> bytes:
    """Colourised depth (turbo, near = red, far = blue) as PNG; pixels without a reading are dark grey."""
    scaled = np.clip(depth_m / max(max_m, 1e-3), 0, 1)
    bgr = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bgr[depth_m <= 0] = NO_DEPTH_BGR
    ok, png = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("could not encode depth PNG")
    return png.tobytes()


def pack_points(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    """One binary map message for the browser."""
    return (
        POINTS_TAG
        + struct.pack("<I", len(xyz))
        + np.ascontiguousarray(xyz, "<f4").tobytes()
        + np.ascontiguousarray(rgb, np.uint8).tobytes()
    )


def unpack_points(message: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack_points` (used by tests and tools)."""
    if not message.startswith(POINTS_TAG):
        raise ValueError("not a points message")
    (count,) = struct.unpack_from("<I", message, 1)
    xyz_end = 5 + count * 12
    xyz = np.frombuffer(message[5:xyz_end], "<f4").reshape(count, 3)
    rgb = np.frombuffer(message[xyz_end : xyz_end + count * 3], np.uint8).reshape(count, 3)
    return xyz, rgb
