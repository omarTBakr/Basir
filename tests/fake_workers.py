"""Child-process mains for `ProcessBackend(target=...)` tests: tiny, no torch, importable by `spawn`."""

from __future__ import annotations

from multiprocessing.connection import Connection
import os
import time


def crash_on_start(conn: Connection, device: str) -> None:
    """Dies while 'loading the model' (like a CUDA import error), without saying why."""
    os._exit(3)


def refuse_on_start(conn: Connection, device: str) -> None:
    conn.send(("error", "CUDA out of memory"))


def never_ready(conn: Connection, device: str) -> None:
    time.sleep(30)


def serve(conn: Connection, device: str) -> None:
    """Answers every batch with its size in milliseconds."""
    conn.send(("ready", "fake"))
    while (jpegs := conn.recv()) is not None:
        conn.send(("ok", len(jpegs) / 1000))


def die_after_one_batch(conn: Connection, device: str) -> None:
    conn.send(("ready", "fake"))
    jpegs = conn.recv()
    conn.send(("ok", len(jpegs) / 1000))
    conn.recv()
    os._exit(1)
