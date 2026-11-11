"""The phone <-> server wire format on `/ws/stream/{device_id}`.

Phone -> server, one binary WebSocket message per frame:

    +--------------+--------------------+--------------------+---------------------------+
    | header length| JSON header (UTF-8)| JPEG bytes         | depth (optional):         |
    | (u32, big-   |                    | (colour image)     | uint16 little-endian,     |
    | endian)      |                    |                    | width x height, row-major |
    +--------------+--------------------+--------------------+---------------------------+

    header = {
        "seq": 17,                       # per-phone frame counter (required)
        "timestamp": 1305031453.36,      # sensor time in seconds (required)
        "pose": [16 floats] | null,      # T_world_cam, row-major 4x4 (optional)
        "intrinsics": {"fx", "fy", "cx", "cy", "width", "height"} | null,  # of the colour image (optional)
        "depth": {"width", "height", "scale"} | null,  # scale = metres per unit (ARCore DEPTH16: 0.001)
        "jpeg_bytes": 48213 | null       # length of the JPEG part; required when "depth" is set
    }

Without "depth" everything after the header is the JPEG (protocol v1), so colour-only
clients keep working.

Server -> phone, one text WebSocket message (JSON) per processed frame:

    {"device": "phone-1", "seq": 17, "batch_size": 3, "queue_ms": 4.1,
     "inference_ms": 6.3, "latency_ms": 12.0, "dropped": 2}

`latency_ms` runs from receipt on the server to the result being sent back;
`dropped` is how many of this phone's frames were replaced by newer ones so far.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import struct
from typing import Any

_LENGTH = struct.Struct(">I")
_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy", "width", "height")
_DEPTH_KEYS = ("width", "height", "scale")


class ProtocolError(ValueError):
    """A frame message that does not follow the wire format."""


@dataclass(frozen=True)
class FrameHeader:
    """Metadata sent with each frame."""

    seq: int
    timestamp: float
    pose: list[float] | None = None
    intrinsics: dict[str, float] | None = None
    depth: dict[str, float] | None = None
    jpeg_bytes: int | None = None


@dataclass(frozen=True)
class FrameResult:
    """What the server sends back for one processed frame."""

    device: str
    seq: int
    batch_size: int
    queue_ms: float
    inference_ms: float
    latency_ms: float
    dropped: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def encode_frame(header: FrameHeader, jpeg: bytes, depth: bytes | None = None) -> bytes:
    """Pack a header, a JPEG and optional uint16 depth (described by `header.depth`) into one message."""
    if depth is not None:
        if header.depth is None:
            raise ProtocolError("depth bytes need 'depth' in the header")
        header = replace(header, jpeg_bytes=len(jpeg))
    payload = json.dumps(asdict(header), separators=(",", ":")).encode()
    return _LENGTH.pack(len(payload)) + payload + jpeg + (depth or b"")


def decode_frame(message: bytes) -> tuple[FrameHeader, bytes, bytes | None]:
    """Split a binary message into its validated header, JPEG bytes and depth bytes (None without depth)."""
    if len(message) < _LENGTH.size:
        raise ProtocolError("message shorter than the length prefix")
    (length,) = _LENGTH.unpack_from(message)
    end = _LENGTH.size + length
    if end > len(message):
        raise ProtocolError("header length exceeds the message")
    try:
        raw = json.loads(message[_LENGTH.size : end])
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"header is not JSON: {e}") from e
    header = _validate(raw)
    body = message[end:]
    if header.depth is None:
        return header, body, None
    if header.jpeg_bytes is None:
        raise ProtocolError("'jpeg_bytes' is required when 'depth' is set")
    expected = int(header.depth["width"]) * int(header.depth["height"]) * 2
    if len(body) != header.jpeg_bytes + expected:
        raise ProtocolError(f"depth needs {expected} bytes after the {header.jpeg_bytes}-byte JPEG")
    return header, body[: header.jpeg_bytes], body[header.jpeg_bytes :]


def _validate(raw: Any) -> FrameHeader:
    if not isinstance(raw, dict):
        raise ProtocolError("header must be a JSON object")
    try:
        seq, timestamp = int(raw["seq"]), float(raw["timestamp"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProtocolError("header needs a numeric 'seq' and 'timestamp'") from e
    pose = raw.get("pose")
    if pose is not None and (not isinstance(pose, list) or len(pose) != 16):
        raise ProtocolError("'pose' must be 16 numbers (row-major 4x4)")
    intrinsics = raw.get("intrinsics")
    if intrinsics is not None and (not isinstance(intrinsics, dict) or set(_INTRINSIC_KEYS) - intrinsics.keys()):
        raise ProtocolError(f"'intrinsics' needs {', '.join(_INTRINSIC_KEYS)}")
    depth = raw.get("depth")
    if depth is not None and (not isinstance(depth, dict) or set(_DEPTH_KEYS) - depth.keys()):
        raise ProtocolError(f"'depth' needs {', '.join(_DEPTH_KEYS)}")
    jpeg_bytes = raw.get("jpeg_bytes")
    if jpeg_bytes is not None and (not isinstance(jpeg_bytes, int) or jpeg_bytes < 0):
        raise ProtocolError("'jpeg_bytes' must be a non-negative integer")
    return FrameHeader(
        seq=seq,
        timestamp=timestamp,
        pose=[float(v) for v in pose] if pose is not None else None,
        intrinsics={k: float(intrinsics[k]) for k in _INTRINSIC_KEYS} if intrinsics is not None else None,
        depth={k: float(depth[k]) for k in _DEPTH_KEYS} if depth is not None else None,
        jpeg_bytes=jpeg_bytes,
    )
