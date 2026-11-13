"""Tests for basir/api/protocol.py (the phone <-> server wire format)."""

from __future__ import annotations

import json
import struct

import pytest

from basir.api.protocol import (
    FrameHeader,
    FrameResult,
    ProtocolError,
    decode_frame,
    encode_frame,
    valid_device_id,
)

JPEG = b"\xff\xd8fake-jpeg\xff\xd9"


def test_round_trip_with_pose_and_intrinsics() -> None:
    pose = [float(i) for i in range(16)]
    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "width": 640.0, "height": 480.0}
    header = FrameHeader(seq=7, timestamp=1305031453.25, pose=pose, intrinsics=intrinsics)
    decoded, jpeg, depth = decode_frame(encode_frame(header, JPEG))
    assert decoded == header and jpeg == JPEG and depth is None


def test_minimal_header() -> None:
    header, jpeg, depth = decode_frame(encode_frame(FrameHeader(seq=0, timestamp=1.5), JPEG))
    assert header.pose is None and header.intrinsics is None and header.depth is None
    assert jpeg == JPEG and depth is None


def test_round_trip_with_depth() -> None:
    depth = bytes(range(24))  # 4 x 3 uint16 values
    header = FrameHeader(seq=1, timestamp=2.0, depth={"width": 4.0, "height": 3.0, "scale": 0.001})
    decoded, jpeg, raw = decode_frame(encode_frame(header, JPEG, depth))
    assert decoded.jpeg_bytes == len(JPEG) and decoded.depth == header.depth
    assert jpeg == JPEG and raw == depth


def test_depth_without_header_metadata_is_refused() -> None:
    with pytest.raises(ProtocolError):
        encode_frame(FrameHeader(seq=1, timestamp=2.0), JPEG, b"\x00\x00")


def _raw(header: object) -> bytes:
    payload = json.dumps(header).encode()
    return struct.pack(">I", len(payload)) + payload + JPEG


@pytest.mark.parametrize(
    "message",
    [
        b"\x00\x01",  # shorter than the length prefix
        struct.pack(">I", 999) + b"{}",  # length beyond the message
        struct.pack(">I", 3) + b"{x}" + JPEG,  # not JSON
        _raw([1, 2]),  # not an object
        _raw({"timestamp": 1.0}),  # no seq
        _raw({"seq": 1, "timestamp": 1.0, "pose": [1, 2, 3]}),  # pose not 4x4
        _raw({"seq": 1, "timestamp": 1.0, "intrinsics": {"fx": 1}}),  # intrinsics incomplete
        _raw({"seq": 1, "timestamp": 1.0, "depth": {"width": 2}}),  # depth incomplete
        _raw({"seq": 1, "timestamp": 1.0, "depth": {"width": 2, "height": 2, "scale": 1}}),  # no jpeg_bytes
        _raw({"seq": 1, "timestamp": 1.0, "depth": {"width": 64, "height": 64, "scale": 1}, "jpeg_bytes": 1}),
    ],
)
def test_malformed_messages_are_rejected(message: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_frame(message)


GOOD_INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
GOOD_DEPTH = {"width": 4, "height": 3, "scale": 0.001}


def _with(**fields: object) -> bytes:
    """A header-only message (no depth bytes) with `fields` merged into a valid header."""
    return _raw({"seq": 1, "timestamp": 1.0, **fields})


def _depth_message(depth: dict[str, object], depth_bytes: int = 24) -> bytes:
    payload = json.dumps({"seq": 1, "timestamp": 1.0, "depth": depth, "jpeg_bytes": len(JPEG)}).encode()
    return struct.pack(">I", len(payload)) + payload + JPEG + b"\x00" * depth_bytes


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (_with(seq=1.5), "fractional seq"),
        (_with(seq=True), "bool seq"),
        (_with(seq="3"), "string seq"),
        (_with(timestamp=NAN), "NaN timestamp"),
        (_with(timestamp=INF), "infinite timestamp"),
        (_with(timestamp="now"), "string timestamp"),
        (_with(pose=[NAN] + [0.0] * 15), "NaN in pose"),
        (_with(pose=[INF] + [0.0] * 15), "Infinity in pose"),
        (_with(pose=["1"] + [0.0] * 15), "string in pose"),
        (_with(intrinsics={**GOOD_INTRINSICS, "fx": 0}), "fx = 0"),
        (_with(intrinsics={**GOOD_INTRINSICS, "fy": -1}), "negative fy"),
        (_with(intrinsics={**GOOD_INTRINSICS, "cx": NAN}), "NaN cx"),
        (_with(intrinsics={**GOOD_INTRINSICS, "width": 0}), "zero colour width"),
        (_with(intrinsics={**GOOD_INTRINSICS, "height": 480.5}), "fractional colour height"),
        (_with(intrinsics={**GOOD_INTRINSICS, "width": 5000}), "colour width over the bound"),
        (_depth_message({**GOOD_DEPTH, "width": -1}, 0), "negative depth width"),
        (_depth_message({**GOOD_DEPTH, "width": 0}, 0), "zero depth width"),
        (_depth_message({**GOOD_DEPTH, "height": 1.5}), "fractional depth height"),
        (_depth_message({**GOOD_DEPTH, "scale": 0}), "zero depth scale"),
        (_depth_message({**GOOD_DEPTH, "scale": -0.001}), "negative depth scale"),
        (_depth_message({**GOOD_DEPTH, "scale": NAN}), "NaN depth scale"),
        (_depth_message({**GOOD_DEPTH, "width": 4097, "height": 1}, 4097 * 2), "depth width over the bound"),
        (_raw({"seq": 1, "timestamp": 1.0, "depth": GOOD_DEPTH, "jpeg_bytes": True}), "bool jpeg_bytes"),
    ],
)
def test_nonsense_headers_are_rejected(message: bytes, reason: str) -> None:
    with pytest.raises(ProtocolError):
        decode_frame(message)


def test_the_size_bound_comes_from_the_caller() -> None:
    message = _depth_message({"width": 64, "height": 2, "scale": 0.001}, 64 * 2 * 2)
    decode_frame(message)  # within the default bound
    with pytest.raises(ProtocolError, match="1 to 32"):
        decode_frame(message, max_side=32)


def test_whole_number_floats_are_accepted_as_sizes() -> None:
    """JavaScript may send 640.0 as 640 or 640.0; both are a whole number."""
    header, _, depth = decode_frame(_depth_message({"width": 4.0, "height": 3, "scale": 0.001}))
    assert header.depth == {"width": 4.0, "height": 3.0, "scale": 0.001} and depth is not None


@pytest.mark.parametrize(
    ("device", "ok"),
    [
        ("phone-1", True),
        ("a.b_c", True),
        ("x" * 64, True),
        ("", False),
        ("x" * 65, False),
        ("../etc", False),
        ("<script>", False),
        ("phone 1", False),
        ("phone-1\n", False),
    ],
)
def test_device_ids(device: str, ok: bool) -> None:
    assert valid_device_id(device) is ok


def test_result_json() -> None:
    result = FrameResult("phone-1", 3, 2, 1.0, 5.0, 8.5, 1)
    assert json.loads(result.to_json()) == {
        "device": "phone-1",
        "seq": 3,
        "batch_size": 2,
        "queue_ms": 1.0,
        "inference_ms": 5.0,
        "latency_ms": 8.5,
        "dropped": 1,
    }
