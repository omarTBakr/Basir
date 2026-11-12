"""The phone <-> server wire format on `/ws/stream/{device_id}`.

Phone -> server, one binary WebSocket message per frame:

    +--------------+--------------------+--------------------+---------------------------+
    | header length| JSON header (UTF-8)| JPEG bytes         | depth (optional):         |
    | (u32, big-   |                    | (colour image)     | uint16 little-endian,     |
    | endian)      |                    |                    | width x height, row-major |
    +--------------+--------------------+--------------------+---------------------------+

    header = {
        "seq": 17,                       # per-phone frame counter (required)
        "timestamp": 1305031453.36,      # seconds on the phone's own clock (required; the web page sends
                                         # performance.now(), not wall time: the server keeps its own receipt time)
        "pose": [16 floats] | null,      # T_world_cam, row-major 4x4 (optional)
        "intrinsics": {"fx", "fy", "cx", "cy", "width", "height"} | null,  # of the colour image (optional)
        "depth": {"width", "height", "scale"} | null,  # scale = metres per unit (ARCore DEPTH16: 0.001)
        "jpeg_bytes": 48213 | null       # length of the JPEG part; required when "depth" is set
    }

Every number must be finite (no NaN or Infinity); `fx`, `fy` and the depth `scale` must be
positive, and image `width`/`height` whole numbers in 1..`max_side` (INGEST_MAX_IMAGE_SIDE).
The depth-scale convention is the opposite of `CameraIntrinsics.depth_scale`; see
`basir.ingest.phone_frame`, which turns a message into a `basir.core.types.Frame`.

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
import math
import re
import struct
from typing import Any

_LENGTH = struct.Struct(">I")
_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy", "width", "height")
_DEPTH_KEYS = ("width", "height", "scale")

DEVICE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # every route and socket that takes a device id checks it
MAX_IMAGE_SIDE = 4096  # default bound on image width/height (Settings.ingest_max_image_side)


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


def valid_device_id(device_id: str) -> bool:
    """True when `device_id` is 1-64 characters of letters, digits, '.', '_' or '-'."""
    return DEVICE_ID.fullmatch(device_id) is not None


def decode_frame(message: bytes, max_side: int = MAX_IMAGE_SIDE) -> tuple[FrameHeader, bytes, bytes | None]:
    """Split a binary message into its validated header, JPEG bytes and depth bytes (None without depth).

    `max_side` bounds every image width and height in the header.
    """
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
    header = _validate(raw, max_side)
    body = message[end:]
    if header.depth is None:
        return header, body, None
    if header.jpeg_bytes is None:
        raise ProtocolError("'jpeg_bytes' is required when 'depth' is set")
    expected = int(header.depth["width"]) * int(header.depth["height"]) * 2
    if len(body) != header.jpeg_bytes + expected:
        raise ProtocolError(f"depth needs {expected} bytes after the {header.jpeg_bytes}-byte JPEG")
    return header, body[: header.jpeg_bytes], body[header.jpeg_bytes :]


def _number(value: Any, name: str) -> float:
    """A finite JSON number (bools are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"'{name}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"'{name}' must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ProtocolError(f"'{name}' must be positive")
    return number


def _side(value: Any, name: str, max_side: int) -> float:
    """An image width or height: a whole number in 1..max_side."""
    number = _number(value, name)
    if not number.is_integer() or not 1 <= number <= max_side:
        raise ProtocolError(f"'{name}' must be a whole number from 1 to {max_side}")
    return number


def _validate(raw: Any, max_side: int = MAX_IMAGE_SIDE) -> FrameHeader:
    if not isinstance(raw, dict):
        raise ProtocolError("header must be a JSON object")
    seq = raw.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or "timestamp" not in raw:
        raise ProtocolError("header needs an integer 'seq' and a numeric 'timestamp'")
    timestamp = _number(raw["timestamp"], "timestamp")
    pose = raw.get("pose")
    if pose is not None:
        if not isinstance(pose, list) or len(pose) != 16:
            raise ProtocolError("'pose' must be 16 numbers (row-major 4x4)")
        pose = [_number(v, "pose") for v in pose]
    intrinsics = raw.get("intrinsics")
    if intrinsics is not None:
        if not isinstance(intrinsics, dict) or set(_INTRINSIC_KEYS) - intrinsics.keys():
            raise ProtocolError(f"'intrinsics' needs {', '.join(_INTRINSIC_KEYS)}")
        intrinsics = {
            "fx": _positive(intrinsics["fx"], "fx"),
            "fy": _positive(intrinsics["fy"], "fy"),
            "cx": _number(intrinsics["cx"], "cx"),
            "cy": _number(intrinsics["cy"], "cy"),
            "width": _side(intrinsics["width"], "intrinsics.width", max_side),
            "height": _side(intrinsics["height"], "intrinsics.height", max_side),
        }
    depth = raw.get("depth")
    if depth is not None:
        if not isinstance(depth, dict) or set(_DEPTH_KEYS) - depth.keys():
            raise ProtocolError(f"'depth' needs {', '.join(_DEPTH_KEYS)}")
        depth = {
            "width": _side(depth["width"], "depth.width", max_side),
            "height": _side(depth["height"], "depth.height", max_side),
            "scale": _positive(depth["scale"], "depth.scale"),
        }
    jpeg_bytes = raw.get("jpeg_bytes")
    if jpeg_bytes is not None and (isinstance(jpeg_bytes, bool) or not isinstance(jpeg_bytes, int) or jpeg_bytes < 0):
        raise ProtocolError("'jpeg_bytes' must be a non-negative integer")
    return FrameHeader(
        seq=seq, timestamp=timestamp, pose=pose, intrinsics=intrinsics, depth=depth, jpeg_bytes=jpeg_bytes
    )
