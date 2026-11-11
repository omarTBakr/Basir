"""The GPU worker: one separate process, started with `spawn`, that runs batches of frames.

The server process never touches CUDA: torch is imported, and the model loaded, only
inside the spawned child, once. Batches go over a pipe as lists of JPEG bytes; the
child answers with the batch's inference time.

`StubModel` stands in for perception until Phase 1: it decodes and resizes each JPEG and
runs one convolution on the GPU (CPU if there is none), so timings are realistic.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import multiprocessing
from multiprocessing.connection import Connection
import threading
import time
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger("basir.worker")

MODEL_INPUT = 224  # square input side, like a typical vision backbone


class InferenceBackend(Protocol):
    """Runs batches of JPEG frames; `start` before the first batch, `stop` at shutdown."""

    device: str  # where inference runs, for display

    def start(self) -> None: ...

    async def infer(self, jpegs: list[bytes]) -> float:
        """Process one batch; return its inference time in seconds."""
        ...

    def stop(self) -> None: ...


class StubModel:
    """Decode, resize and one 3x3 convolution: perception-shaped work with no real output."""

    def __init__(self, device: str) -> None:
        import torch

        self._torch = torch
        self.device = _resolve_device(torch, device)
        self._weight = torch.randn(16, 3, 3, 3, device=self.device)

    def __call__(self, jpegs: list[bytes]) -> list[float]:
        """One score per frame (the mean activation), so the work cannot be skipped."""
        images = [_decode(jpeg) for jpeg in jpegs]
        x = self._torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        y = self._torch.nn.functional.conv2d(x, self._weight, padding=1).relu().mean(dim=(1, 2, 3))
        return [float(v) for v in y.cpu()]  # .cpu() waits for the GPU to finish


class ProcessBackend:
    """`InferenceBackend` that runs `StubModel` in a spawned child process."""

    def __init__(self, device: str = "auto", startup_timeout: float = 120.0) -> None:
        self.device = device
        self.startup_timeout = startup_timeout
        self._conn: Connection | None = None
        self._process: Any = None
        self._lock = threading.Lock()  # one batch in flight on the pipe at a time

    def start(self) -> None:
        """Spawn the worker and block until its model is loaded."""
        context = multiprocessing.get_context("spawn")
        self._conn, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child, self.device), name="basir-gpu-worker", daemon=True)
        self._process.start()
        child.close()
        if not self._conn.poll(self.startup_timeout):
            self.stop()
            raise RuntimeError(f"GPU worker did not start within {self.startup_timeout:.0f} s")
        self.device = self._conn.recv()
        logger.info("GPU worker ready on %s (pid %d)", self.device, self._process.pid)

    async def infer(self, jpegs: list[bytes]) -> float:
        return await asyncio.to_thread(self._roundtrip, jpegs)

    def stop(self) -> None:
        """Ask the worker to exit; terminate it if it does not."""
        if self._conn is not None:
            with suppress(OSError):  # the worker may already be gone
                self._conn.send(None)
            self._conn.close()
            self._conn = None
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    def _roundtrip(self, jpegs: list[bytes]) -> float:
        if self._conn is None:
            raise RuntimeError("GPU worker is not running")
        with self._lock:
            try:
                self._conn.send(jpegs)
                status, value = self._conn.recv()
            except (EOFError, OSError) as e:
                raise RuntimeError("GPU worker exited") from e
        if status != "ok":
            raise RuntimeError(f"GPU worker failed: {value}")
        return float(value)


def _serve(conn: Connection, device: str) -> None:
    """Child process: load the model once, then answer batches until told to stop."""
    model = StubModel(device)
    model([_blank_jpeg()])  # warm-up: first CUDA call pays kernel and allocator setup
    conn.send(model.device)
    while (jpegs := conn.recv()) is not None:
        start = time.perf_counter()
        try:
            model(jpegs)
            conn.send(("ok", time.perf_counter() - start))
        except Exception as e:
            conn.send(("error", repr(e)))


def _resolve_device(torch: Any, device: str) -> str:
    target = device.strip().lower()
    if target != "auto":
        return target
    return "cuda" if torch.cuda.is_available() else "cpu"


def _decode(jpeg: bytes) -> np.ndarray:
    """JPEG -> MODEL_INPUT x MODEL_INPUT RGB; an undecodable frame becomes black."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((MODEL_INPUT, MODEL_INPUT, 3), np.uint8)
    return cv2.cvtColor(cv2.resize(bgr, (MODEL_INPUT, MODEL_INPUT)), cv2.COLOR_BGR2RGB)


def _blank_jpeg() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((MODEL_INPUT, MODEL_INPUT, 3), np.uint8))
    return buffer.tobytes() if ok else b""
