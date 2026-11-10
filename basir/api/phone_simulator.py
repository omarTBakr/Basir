"""Simulated phones: replay recorded frames to the server over `/ws/stream/{device_id}`, like ARCore would.

Each phone sends JPEG + header messages at a fixed rate and records what comes back:
round-trip time measured on the phone, the server's own latency, batch sizes and drops.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import statistics
import time

import cv2
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from basir.api.protocol import FrameHeader, encode_frame
from basir.ingest.sources.base import FrameSource

JPEG_QUALITY = 80


@dataclass(frozen=True)
class RecordedFrame:
    header: FrameHeader
    jpeg: bytes


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


def record(source: FrameSource, max_frames: int | None = None) -> list[RecordedFrame]:
    """Encode a source's frames as they would leave a phone: JPEG colour, pose and intrinsics."""
    frames: list[RecordedFrame] = []
    for frame in source:
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            continue
        k = frame.intrinsics
        header = FrameHeader(
            seq=len(frames),
            timestamp=frame.timestamp,
            pose=frame.pose.matrix.flatten().tolist() if frame.pose is not None else None,
            intrinsics={"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy, "width": k.width, "height": k.height},
        )
        frames.append(RecordedFrame(header, jpeg.tobytes()))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


async def run_phone(
    base_url: str, device: str, frames: list[RecordedFrame], fps: float, duration: float, delay: float = 0.0
) -> PhoneStats:
    """Stream `frames` (looping) at `fps` for `duration` seconds; fps <= 0 sends as fast as possible."""
    stats = PhoneStats(device)
    sent_at: dict[int, float] = {}
    await asyncio.sleep(delay)
    async with connect(f"{base_url.rstrip('/')}/ws/stream/{device}", max_size=None) as ws:
        receiver = asyncio.create_task(_receive(ws, stats, sent_at))
        start = time.perf_counter()
        seq = 0
        while (elapsed := time.perf_counter() - start) < duration:
            recorded = frames[seq % len(frames)]
            header = FrameHeader(seq, recorded.header.timestamp, recorded.header.pose, recorded.header.intrinsics)
            sent_at[seq] = time.perf_counter()
            await ws.send(encode_frame(header, recorded.jpeg))
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
