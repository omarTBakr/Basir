"""The phone <-> server wire format on `/ws/stream/{device_id}`.

Phone -> server, one binary WebSocket message per frame:

    +----------------------+---------------------------+------------------+
    | header length (u32,  | JSON header (UTF-8)       | JPEG bytes       |
    | big-endian)          |                           | (colour image)   |
    +----------------------+---------------------------+------------------+

    header = {
        "seq": 17,                       # per-phone frame counter (required)
        "timestamp": 1305031453.36,      # sensor time in seconds (required)
        "pose": [16 floats] | null,      # T_world_cam, row-major 4x4 (optional)
        "intrinsics": {"fx", "fy", "cx", "cy", "width", "height"} | null  (optional)
    }

Server -> phone, one text WebSocket message (JSON) per processed frame:

    {"device": "phone-1", "seq": 17, "batch_size": 3, "queue_ms": 4.1,
     "inference_ms": 6.3, "latency_ms": 12.0, "dropped": 2}

`latency_ms` runs from receipt on the server to the result being sent back;
`dropped` is how many of this phone's frames were replaced by newer ones so far.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import struct
from typing import Any

_LENGTH = struct.Struct(">I")
_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy", "width", "height")


class ProtocolError(ValueError):
    """A frame message that does not follow the wire format."""


@dataclass(frozen=True)
class FrameHeader:
    """Metadata sent with each JPEG."""

    seq: int
    timestamp: float
    pose: list[float] | None = None
    intrinsics: dict[str, float] | None = None


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


def encode_frame(header: FrameHeader, jpeg: bytes) -> bytes:
    """Pack a header and a JPEG into one binary message."""
    payload = json.dumps(asdict(header), separators=(",", ":")).encode()
    return _LENGTH.pack(len(payload)) + payload + jpeg


def decode_frame(message: bytes) -> tuple[FrameHeader, bytes]:
    """Split a binary message into its validated header and JPEG bytes."""
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
    return _validate(raw), message[end:]


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
    return FrameHeader(
        seq=seq,
        timestamp=timestamp,
        pose=[float(v) for v in pose] if pose is not None else None,
        intrinsics={k: float(intrinsics[k]) for k in _INTRINSIC_KEYS} if intrinsics is not None else None,
    )
