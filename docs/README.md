# Basir documentation

One page per feature that's built, and a placeholder for each one that isn't. Start with [Setup](01-setup.md), then [Replay and point clouds](05-replay-and-pointcloud.md) for the Phase 0 check. The UML (user stories, use cases, sequence and class diagrams) lives on [the `software_engineering` branch](https://github.com/omarTBakr/Basir/tree/software_engineering/software_engineering).

## Built

| Page | Covers |
|---|---|
| [01 Setup and configuration](01-setup.md) | Prerequisites, `uv sync`, `check_env.py`, settings precedence and every setting key |
| [02 Geometry and camera conventions](02-geometry.md) | x right / y down / z forward, `T_world_cam`, depth factor 5000, per-sensor calibration, the consistency score |
| [03 Datasets](03-datasets.md) | `download_dataset.py`, the 18 TUM presets, the TUM folder format, `TUMDatasetSource`, the toy dataset |
| [04 Data versioning with DVC](04-data-versioning.md) | Tracked datasets, the local remote, the hardlink cache and `dvc unprotect` |
| [05 Replay and the Phase 0 point-cloud check](05-replay-and-pointcloud.md) | `replay.py`, `visualize_pointcloud.py`, fusion, the depth-consistency gate and its results |
| [06 Experiment tracking and tracing](06-experiment-tracking.md) | MLflow experiments, standard tags, parent / child pipeline runs and their tag schema; Langfuse tracing |
| [07 Pipelines](07-pipelines.md) | The Airflow DAG `phase0`, its tasks and retries, `run_pipeline.py`, `dag.json`, adding a DAG |
| [08 Local stack](08-local-stack.md) | Docker Compose services and ports, native Docker Engine for the GPU, stack settings, Grafana dashboards and panel ids |
| [09 Server and UI](09-server-and-ui.md) | FastAPI on :8000, the `/ws/stream/{device_id}` protocol, latest-frame queues, micro-batching, the GPU worker, `basir_*` metrics, the four pages |
| [10 Async processing](10-async-processing.md) | Why live frames use asyncio and pipelines use Airflow, and why not Celery |
| [11 Testing and code quality](11-testing.md) | pytest and the `network` marker, fixtures, ruff, mypy, CI and pre-commit |
| [12 Architecture and project structure](12-architecture.md) | Module map, the design rules, the file tree |

## Planned, not built

| Page | Covers |
|---|---|
| [20 Phase 1: L1 temporal memory](20-planned-phase1-temporal-memory.md) | Keyframes, SigLIP embeddings in Qdrant, "seen 42 s ago" |
| [21 Phases 2 to 5](21-planned-perception-memory-query.md) | Object-level memory, 3D spatial memory and directions, re-acquisition, smart querying |
| [22 ARCore phone app](22-planned-arcore-phone-app.md) | The Android app that streams pose, depth and video to the server |
| [23 Robot integration](23-planned-robot-integration.md) | ROS 2 and Nav2 paths to remembered objects |

## Status by phase

| Phase | Status |
|---|---|
| 0 Setup: environment, TUM data, geometry, frame source, replay, point-cloud check, MLflow + Langfuse wiring, DVC, Airflow pipeline, local stack | **Built**, final visual check pending |
| 1 L1 temporal memory: keyframes, SigLIP embeddings in Qdrant, "seen 42 s ago" answers, Gradio UI | Planned |
| 2 Object-level memory: open-vocabulary detection, SAM masks, per-object embeddings, 2D fusion | Planned |
| 3 L2 spatial memory: 3D object positions, 3D fusion, bearing + distance directions | Planned |
| 4 Re-acquisition and segmentation: live guidance, "Found it!" mask, tracking, moved-object handling | Planned |
| 5 Smart querying: LLM query parsing, hybrid retrieval, VLM reranking | Planned |
| 6 Real-time live pipeline: ARCore phone app streaming to the laptop, async workers, latency budget | Server side built: multi-phone ingest, GPU worker, metrics and UI ([09](09-server-and-ui.md)); phone app planned |
| 7 Robot integration: ROS 2, Nav2 paths (L3) | Planned |
| 8 Evaluation and hardening (ongoing) | Planned |

## Conventions for these pages

- Every command shown exists in the repo; run Python scripts as `uv run python scripts/<name>.py`.
- Paths are relative to the repository root.
- When a feature lands, its page is updated in the same change, and a placeholder page becomes a real one.
