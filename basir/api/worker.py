"""The GPU worker: one separate process, started with `spawn`, that runs batches of frames.

The server process never touches CUDA: torch is imported, and the model loaded, only
inside the spawned child, once. Batches go over a pipe as lists of JPEG bytes; the
child answers with the batch's inference time.

`StubModel` stands in for perception until Phase 1: it decodes and resizes each JPEG and
runs one convolution on the GPU (CPU if there is none), so timings are realistic.

Lifecycle: `start` raises a clear `RuntimeError` (and cleans up the pipe and process) when the
child does not come up within `startup_timeout` (WORKER_STARTUP_TIMEOUT_S) or dies while
loading. If the child dies later, the backend logs it once, sets `basir_worker_alive` to 0 and
restarts it once; while it is down every batch fails fast with `WorkerDown`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
import logging
import multiprocessing
from multiprocessing.connection import Connection
import threading
import time
from typing import Any, Protocol

import cv2
import numpy as np
from prometheus_client import Gauge

logger = logging.getLogger("basir.worker")

MODEL_INPUT = 224  # square input side, like a typical vision backbone
DEFAULT_STARTUP_TIMEOUT_S = 120.0  # Settings.worker_startup_timeout_s
STOP_LOCK_TIMEOUT_S = 10.0  # stop() waits this long for a batch in flight before terminating

WORKER_ALIVE = Gauge("basir_worker_alive", "1 while the GPU worker process is up, 0 when it is down")


class WorkerDown(RuntimeError):
    """The GPU worker process is not running (it died and could not be restarted, or was stopped)."""


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
    """`InferenceBackend` that runs `StubModel` in a spawned child process.

    `target(conn, device)` is the child's main function (tests pass small fakes). It must send
    `("ready", device)` or `("error", message)` once, then answer each batch with
    `("ok", seconds)` or `("error", message)` until it receives None.
    """

    def __init__(
        self,
        device: str = "auto",
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
        restarts: int = 1,
        target: Callable[[Connection, str], None] | None = None,
    ) -> None:
        self.device = device
        self.requested_device = device
        self.startup_timeout = startup_timeout
        self.restarts_left = restarts
        self._target = target or _serve
        self._conn: Connection | None = None
        self._process: Any = None
        self._lock = threading.Lock()  # one batch in flight on the pipe at a time; stop() takes it too
        self.alive = False

    def start(self) -> None:
        """Spawn the worker and block until its model is loaded; RuntimeError (cleaned up) if it is not."""
        context = multiprocessing.get_context("spawn")
        self._conn, child = context.Pipe()
        self._process = context.Process(
            target=self._target, args=(child, self.requested_device), name="basir-gpu-worker", daemon=True
        )
        self._process.start()
        child.close()
        try:
            if not self._conn.poll(self.startup_timeout):
                raise RuntimeError(f"GPU worker did not start within {self.startup_timeout:.0f} s")
            status, value = self._conn.recv()
        except (EOFError, OSError, ValueError, TypeError) as e:  # died while loading, or a malformed hello
            code = self._process.exitcode if self._process is not None else None
            self._shutdown()
            raise RuntimeError(f"GPU worker exited during startup (exit code {code})") from e
        except RuntimeError:
            self._shutdown()
            raise
        if status != "ready":
            self._shutdown()
            raise RuntimeError(f"GPU worker failed to start: {value}")
        self.device = str(value)
        self.alive = True
        WORKER_ALIVE.set(1)
        logger.info("GPU worker ready on %s (pid %d)", self.device, self._process.pid)

    async def infer(self, jpegs: list[bytes]) -> float:
        return await asyncio.to_thread(self._roundtrip, jpegs)

    def stop(self) -> None:
        """Ask the worker to exit; terminate it if it does not. Waits for a batch in flight (bounded)."""
        locked = self._lock.acquire(timeout=STOP_LOCK_TIMEOUT_S)
        try:
            if not locked:
                logger.warning("A batch is still in flight after %.0f s; terminating the worker", STOP_LOCK_TIMEOUT_S)
            self.restarts_left = 0
            self._shutdown()
        finally:
            if locked:
                self._lock.release()

    def _shutdown(self) -> None:
        """Close the pipe and end the process (caller holds the lock, or nothing else uses the pipe yet)."""
        self.alive = False
        WORKER_ALIVE.set(0)
        if self._conn is not None:
            with suppress(OSError, ValueError):  # the worker may already be gone
                self._conn.send(None)
            self._conn.close()
            self._conn = None
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            self._process = None

    def _roundtrip(self, jpegs: list[bytes]) -> float:
        with self._lock:
            if self._conn is None:
                raise WorkerDown("GPU worker is not running")
            try:
                self._conn.send(jpegs)
                status, value = self._conn.recv()
            except (EOFError, OSError) as e:
                self._on_death()
                raise WorkerDown("GPU worker exited") from e
        if status != "ok":
            raise RuntimeError(f"GPU worker failed: {value}")
        return float(value)

    def _on_death(self) -> None:
        """The child died mid-batch: log once, mark it down and restart it once (caller holds the lock)."""
        code = self._process.exitcode if self._process is not None else None
        logger.error("GPU worker died (exit code %s)", code)
        self._shutdown()
        if self.restarts_left <= 0:
            logger.error("GPU worker is down; batches will be skipped until the server restarts")
            return
        self.restarts_left -= 1
        try:
            self.start()
        except RuntimeError as e:
            logger.error("Restarting the GPU worker failed: %s", e)


def _serve(conn: Connection, device: str) -> None:
    """Child process: load the model once, then answer batches until told to stop."""
    try:
        model = StubModel(device)
        model([_blank_jpeg()])  # warm-up: first CUDA call pays kernel and allocator setup
    except Exception as e:  # e.g. CUDA out of memory: tell the parent why instead of just dying
        conn.send(("error", repr(e)))
        return
    conn.send(("ready", model.device))
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
