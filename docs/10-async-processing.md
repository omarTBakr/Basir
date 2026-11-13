# Async processing: asyncio for frames, Airflow for pipelines

Basir has two kinds of background work, and they want opposite things. This page records the decision and why.

| | Live frames | Pipelines |
|---|---|---|
| Example | A phone's frame at 15 fps, on its way to the GPU | Download a dataset, fuse two rooms, gate on the targets |
| Worth something for | About 100 ms | Hours; the result must not be lost |
| Right failure mode | Drop it; a fresher frame is coming | Retry the failed step, keep the finished ones |
| Volume | Many small items per second per phone | A handful of long steps per run |
| Built on | **In-process asyncio queues + one GPU process** ([Server and UI](09-server-and-ui.md)) | **Apache Airflow DAGs** ([Pipelines](07-pipelines.md)) |

## Live frames: asyncio

A frame is worth something for about 100 ms. Celery, Airflow and other job systems are built for durable work that must not be lost: they persist every task, retry it, and pay a database or broker round trip per task. For a 15 fps stream from each phone that's the wrong trade. The overhead is larger than the work, and a retried frame from two seconds ago is useless.

In-process asyncio queues cost microseconds, dropping the oldest frame is the right failure mode, and one process owns the GPU. Latency stays bounded because a phone's queue never holds more than `INGEST_QUEUE_SIZE` frames.

**A scheduler can't carry frames anyway.** Airflow starts every task as its own process and stores each task's state and XComs in its database, so seconds of latency per task and a JPEG batch per XCom row would both be absurd.

## Durable pipelines: Airflow

Pipelines are the opposite: long, few, and expensive to redo. Airflow gives them:

- a DAG of dependent tasks, with mapped tasks per dataset;
- per-task retries, timeouts and logs;
- params with a trigger form;
- schedules and asset triggers for the DAGs to come;
- a UI and a REST API;
- StatsD metrics.

The tasks pass only paths and numbers as XComs.

**Why Airflow:** it is the standard tool for DAGs of batch jobs (dataset preparation, post-session processing, evaluations), with scheduling, a DAG-first UI and a REST API built in. Basir's pipelines are exactly that, not long-lived stateful workflows.

## Why not Celery

Celery would sit in between and fit neither side well:

- **Prefork vs CUDA.** Celery's default worker pool forks child processes. CUDA can't be used in a process forked after CUDA was initialised, so a GPU model either loads once per child (several copies in 8 GB of VRAM) or the pool has to be switched to threads or solo, giving up the reason to use it. Basir instead runs one GPU worker started with `spawn`, which loads the model once and is fed micro-batches across phones.
- **For frames** it has a broker round trip and persistence per task, with nothing gained.
- **For pipelines** Airflow already gives more: a DAG of dependent steps, per-task history, logs and a UI, without a separate broker and result backend.

## What this means in code

- The live path never imports Airflow or a broker: [`framebus.py`](../basir/api/framebus.py), [`batcher.py`](../basir/api/batcher.py), [`worker.py`](../basir/api/worker.py).
- The pipeline path never carries frames: [`phase0_steps.py`](../basir/pipelines/phase0_steps.py) exchanges dataclasses of paths and numbers.
- Airflow never shares a Python environment with Basir: tasks run the steps in Basir's own environment ([Pipelines](07-pipelines.md#how-its-built)).
- **Room to scale.** If one server is no longer enough, the `FrameBus` interface lets Redis Streams replace the in-process queues, and `InferenceBackend` lets a second GPU worker join, without touching the WebSocket code. Pipelines scale by switching Airflow's LocalExecutor for the Celery or Kubernetes executor; the DAGs don't change.
