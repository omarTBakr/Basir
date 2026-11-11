# Async processing: asyncio for frames, Temporal for pipelines

Basir has two kinds of background work, and they want opposite things. This page records the decision and why.

| | Live frames | Pipelines |
|---|---|---|
| Example | A phone's frame at 15 fps, on its way to the GPU | Download a dataset, fuse two rooms, gate on the targets |
| Worth something for | About 100 ms | Hours; the result must not be lost |
| Right failure mode | Drop it; a fresher frame is coming | Retry the failed step, keep the finished ones |
| Volume | Many small items per second per phone | A handful of long steps per run |
| Built on | **In-process asyncio queues + one GPU process** ([Server and UI](09-server-and-ui.md)) | **Temporal workflows** ([Pipelines](07-pipelines.md)) |

## Live frames: asyncio

A frame is worth something for about 100 ms. Celery and Temporal are built for durable work that must not be lost: they persist every task, retry it and pay a broker or history round trip per task. For a 15 fps stream from each phone that's the wrong trade: the overhead is larger than the work, and a retried frame from two seconds ago is useless.

In-process asyncio queues cost microseconds, dropping the oldest frame is the right failure mode, and one process owns the GPU. Latency stays bounded because a phone's queue never holds more than `INGEST_QUEUE_SIZE` frames.

**Temporal's payload limit rules it out on its own.** Temporal caps a payload at 2 MB, and every payload is written to its history. A batch of JPEG frames (let alone depth) would hit that limit and fill the database with frames nobody will read again.

## Durable pipelines: Temporal

Pipelines are the opposite: long, few, and expensive to redo. Temporal gives them durable state, per-step retries with policies, heartbeats that detect a dead worker, a UI with each step's inputs and results, and a queryable workflow. The pipeline activities pass only paths and numbers, which keeps them far below the 2 MB limit.

## Why not Celery

Celery would sit in between and fit neither side well:

- **Prefork vs CUDA.** Celery's default worker pool forks child processes. CUDA can't be used in a process forked after CUDA was initialised, so a GPU model either loads once per child (several copies in 8 GB of VRAM) or the pool has to be switched to threads or solo, giving up the reason to use it. Basir instead runs one GPU worker started with `spawn`, which loads the model once and is fed micro-batches across phones.
- **For frames** it has the same per-task broker round trip and persistence as Temporal, with nothing gained.
- **For pipelines** Temporal already gives more: a DAG of dependent steps, per-step history and a UI, without a separate broker and result backend.

## What this means in code

- The live path never imports Temporal or a broker: [`framebus.py`](../basir/api/framebus.py), [`batcher.py`](../basir/api/batcher.py), [`worker.py`](../basir/api/worker.py).
- The pipeline path never carries frames: [`phase0_steps.py`](../basir/pipelines/phase0_steps.py) exchanges dataclasses of paths and numbers.
- **Room to scale.** If one server is no longer enough, the `FrameBus` interface lets Redis Streams replace the in-process queues, and `InferenceBackend` lets a second GPU worker join, without touching the WebSocket code.
