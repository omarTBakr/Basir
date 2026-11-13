"""Micro-batching: drain the frame bus across phones into one batch, run it, route the results.

A batch closes at `max_size` frames or `max_wait` seconds after its first frame, whichever
comes first, so the GPU sees several phones' frames per call without adding much latency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import time

from basir.api.framebus import FrameBus, QueuedFrame
from basir.api.metrics import BATCH_SIZE, FRAME_LATENCY, INFERENCE_SECONDS, QUEUE_DEPTH
from basir.api.worker import InferenceBackend, WorkerDown

logger = logging.getLogger("basir.batcher")


@dataclass(frozen=True)
class ProcessedFrame:
    """One frame after inference, with where its time went."""

    frame: QueuedFrame
    batch_size: int
    queue_s: float  # receipt -> batch sent to the worker
    inference_s: float  # the whole batch inside the worker
    latency_s: float  # receipt -> result ready


class Batcher:
    """Pull batches from a `FrameBus`, run them on an `InferenceBackend`, hand each result to `deliver`."""

    def __init__(
        self,
        bus: FrameBus,
        backend: InferenceBackend,
        deliver: Callable[[ProcessedFrame], None],
        max_size: int,
        max_wait: float,
    ) -> None:
        self.bus = bus
        self.backend = backend
        self.deliver = deliver
        self.max_size = max_size
        self.max_wait = max_wait
        self.skipped = 0  # batches skipped while the worker was down

    async def run(self) -> None:
        """Batch forever (until cancelled)."""
        while True:
            batch = await self.bus.next_batch(self.max_size, self.max_wait)
            if batch:
                await self.process(batch)

    async def process(self, batch: list[QueuedFrame]) -> None:
        """Run one batch and deliver a result per frame; a failed batch is logged and skipped."""
        for device in {f.device for f in batch}:
            QUEUE_DEPTH.labels(device).set(self.bus.depth(device))
        sent = time.perf_counter()
        try:
            inference_s = await self.backend.infer([f.jpeg for f in batch])
        except WorkerDown:  # the backend already logged the death once; do not add a traceback per batch
            if self.skipped == 0:
                logger.warning("GPU worker is down: skipping batches until it is back")
            self.skipped += 1
            return
        except Exception:
            logger.exception("Batch of %d frames failed", len(batch))
            return
        if self.skipped:
            logger.info("GPU worker is back after %d skipped batches", self.skipped)
            self.skipped = 0
        done = time.perf_counter()
        BATCH_SIZE.observe(len(batch))
        INFERENCE_SECONDS.observe(inference_s)
        for frame in batch:
            latency_s = done - frame.received_at
            FRAME_LATENCY.observe(latency_s)
            self.deliver(ProcessedFrame(frame, len(batch), sent - frame.received_at, inference_s, latency_s))
