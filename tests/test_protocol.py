"""Tests for basir/api/protocol.py (the phone <-> server wire format)."""

from __future__ import annotations

import json
import struct

import pytest

from basir.api.protocol import FrameHeader, FrameResult, ProtocolError, decode_frame, encode_frame

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
