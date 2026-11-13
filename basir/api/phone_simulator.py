"""Simulated phones: replay recorded frames to the server over `/ws/stream/{device_id}`, like ARCore would.

Each phone sends header + JPEG + uint16 depth messages (protocol v2) at a fixed rate and records
what comes back: round-trip time measured on the phone, the server's own latency, batch sizes
and drops. Depth is sent in millimetres at ARCore's resolution (160 x 120) by default.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import json
import statistics
import time

import cv2
import numpy as np
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from basir.api.protocol import FrameHeader, encode_frame
from basir.ingest.sources.base import FrameSource

JPEG_QUALITY = 80
DEPTH_SCALE = 0.001  # metres per unit: millimetres, like ARCore DEPTH16
ARCORE_DEPTH_WIDTH = 160


@dataclass(frozen=True)
class RecordedFrame:
    header: FrameHeader
    jpeg: bytes
    depth: bytes | None = None


@dataclass
class PhoneStats:
    """What one simulated phone sent and got back."""

    device: str
    sent: int = 0
    errors: int = 0
    seconds: float = 0.0
    dropped: int = 0  # the server's running count, from the latest result
    rtt_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def results(self) -> int:
        return len(self.rtt_ms)

    def summary(self) -> str:
        fps = self.sent / self.seconds if self.seconds else 0.0
        return (
            f"{self.device}: sent {self.sent} ({fps:.1f} fps), results {self.results}, dropped {self.dropped}, "
            f"errors {self.errors}, batch mean {_mean(self.batch_sizes):.2f} max {max(self.batch_sizes, default=0)}, "
            f"server latency p50 {_pct(self.server_ms, 50):.1f} ms p95 {_pct(self.server_ms, 95):.1f} ms, "
            f"round trip p50 {_pct(self.rtt_ms, 50):.1f} ms p95 {_pct(self.rtt_ms, 95):.1f} ms"
        )


def _depth_mm(depth_m: np.ndarray, width: int | None) -> np.ndarray:
    """Metres -> uint16 millimetres, optionally resized (nearest, so no invented depths) to `width`."""
    if width and width < depth_m.shape[1]:
        height = round(depth_m.shape[0] * width / depth_m.shape[1])
        depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.clip(depth_m / DEPTH_SCALE, 0, 65535).astype("<u2")


def record(
    source: FrameSource, max_frames: int | None = None, depth_width: int | None = ARCORE_DEPTH_WIDTH
) -> list[RecordedFrame]:
    """Encode a source's frames as they would leave a phone: JPEG colour, depth, pose and intrinsics.

    `depth_width` resizes depth like ARCore's small depth image (None keeps the source size).
    """
    frames: list[RecordedFrame] = []
    for frame in source:
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            continue
        k = frame.intrinsics
        depth = _depth_mm(frame.depth, depth_width) if frame.depth is not None else None
        header = FrameHeader(
            seq=len(frames),
            timestamp=frame.timestamp,
            pose=frame.pose.matrix.flatten().tolist() if frame.pose is not None else None,
            intrinsics={"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy, "width": k.width, "height": k.height},
            depth={"width": depth.shape[1], "height": depth.shape[0], "scale": DEPTH_SCALE}
            if depth is not None
            else None,
        )
        frames.append(RecordedFrame(header, jpeg.tobytes(), depth.tobytes() if depth is not None else None))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


async def run_phone(
    base_url: str,
    device: str,
    frames: list[RecordedFrame],
    fps: float,
    duration: float,
    delay: float = 0.0,
    token: str | None = None,
) -> PhoneStats:
    """Stream `frames` (looping) at `fps` for `duration` seconds; fps <= 0 sends as fast as possible.

    `token` (for the LAN listener) is sent as the first message, `{"type": "auth", "token": ...}`.
    """
    stats = PhoneStats(device)
    sent_at: dict[int, float] = {}
    await asyncio.sleep(delay)
    async with connect(f"{base_url.rstrip('/')}/ws/stream/{device}", max_size=None) as ws:
        if token is not None:
            await ws.send(json.dumps({"type": "auth", "token": token}))
        receiver = asyncio.create_task(_receive(ws, stats, sent_at))
        start = time.perf_counter()
        seq = 0
        while (elapsed := time.perf_counter() - start) < duration:
            recorded = frames[seq % len(frames)]
            header = replace(recorded.header, seq=seq)
            sent_at[seq] = time.perf_counter()
            await ws.send(encode_frame(header, recorded.jpeg, recorded.depth))
            stats.sent += 1
            seq += 1
            await asyncio.sleep(max(0.0, seq / fps - elapsed) if fps > 0 else 0)
        stats.seconds = time.perf_counter() - start
        await asyncio.sleep(0.5)  # let in-flight results arrive
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
    return stats


async def simulate(
    base_url: str, phones: int, frames: list[RecordedFrame], fps: float, duration: float
) -> list[PhoneStats]:
    """Run `phones` phones at once, start times staggered across one frame period."""
    period = 1 / fps if fps > 0 else 0.0
    return list(
        await asyncio.gather(
            *(
                run_phone(base_url, f"phone-{i + 1}", frames, fps, duration, delay=i * period / phones)
                for i in range(phones)
            )
        )
    )


async def _receive(ws: ClientConnection, stats: PhoneStats, sent_at: dict[int, float]) -> None:
    try:
        async for message in ws:
            now = time.perf_counter()
            result = json.loads(message)
            if "error" in result:
                stats.errors += 1
                continue
            stats.rtt_ms.append((now - sent_at.pop(result["seq"], now)) * 1000)
            for old in [s for s in sent_at if s < result["seq"]]:  # older frames were dropped
                del sent_at[old]
            stats.server_ms.append(result["latency_ms"])
            stats.batch_sizes.append(result["batch_size"])
            stats.dropped = result["dropped"]
    except ConnectionClosed:
        pass


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(values: list[float], q: int) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=100)[q - 1]
