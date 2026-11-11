<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/basir-logo-dark.svg">
    <img src="logo/basir-logo-light.svg" alt="Basir logo: the scanned room with a pin on the found object and a chat between you and Basir" width="160">
  </picture>
</p>

<h1 align="center">Basir</h1>

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omarrbakr/)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bakro0298@gmail.com)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/omar_bakr/)
[![Portfolio](https://img.shields.io/badge/Portfolio-333333?style=for-the-badge&logo=googlechrome&logoColor=white)](https://omartbakr.github.io/)

</div>

> **Basir** (بصير, "the one who sees): a camera that remembers where it saw your things and walks you back to them.

**▶ [See the end-to-end flow](https://github.com/omarTBakr/Basir/blob/software_engineering/software_engineering/3_sequence_diagrams/system_overview.png)** (remember → ask → directions → found it) · [Jump to the Phase 0 check](docs/05-replay-and-pointcloud.md)

Basir watches a moving camera and builds a **spatial memory** of the objects it sees. Ask *"Where is my water bottle?"* and it finds the bottle in memory, even if it was seen minutes ago and is nowhere in view now. It answers with directions from where you stand ("turn right ~120°, go about 2.3 m, it's on the desk, last seen 45 s ago"), updates them as you move, and outlines the object on screen once it's back in view.

The deployment is a **phone + laptop** split. An Android phone running **ARCore** (realme 14 / 15) records video, its own 6-DoF pose and depth, and shows the answers. The laptop (RTX 4060, 8 GB) is the server that runs the models. No model is trained from scratch: detection, segmentation, embeddings and depth all come from pretrained models (Grounding DINO / YOLO-World, SAM 2/3, SigLIP 2, Depth Anything). The new work is the memory, the 3D fusion and the guidance.

## Status

The stage built today is **Phase 0, the foundation**: replay a recorded RGB-D scene, turn depth and camera poses into a correct 3D point cloud, and *measure* that it's correct. Every direction Basir gives later depends on this geometry. Around it sit the tools every later phase uses: DVC-versioned data, MLflow runs, a Temporal pipeline, a monitoring stack, and a server that already takes frames from several phones at once.

| Phase 0 check (TUM RGB-D) | Median residual | Within 5 cm | Target |
|---|---|---|---|
| `fr1/desk` | 1.89 cm | 84 % | ≤ 3 cm, ≥ 80 % |
| `fr1/room` | 1.81 cm | 82 % | ≤ 3 cm, ≥ 80 % |

Both pass the numbers; the final visual check is pending. The later phases are fully designed but not built yet: see [Status by phase](docs/README.md#status-by-phase). The UML (user stories, use cases, sequence and class diagrams) is on [the `software_engineering` branch](https://github.com/omarTBakr/Basir/blob/software_engineering/software_engineering/README.md).

## Quickstart

```bash
uv sync                                                # everything, incl. torch with CUDA (several GB)
cp env/.env.example env/.env                           # then adjust (see docs/01-setup.md)
uv run python scripts/check_env.py                     # versions, CUDA check, resolved paths

uv run python scripts/download_dataset.py fr1/room     # 746 MB (or, on this machine: uv run dvc pull)
uv run python scripts/visualize_pointcloud.py fr1/room # the Phase 0 check

scripts/stack.sh up                                    # MLflow, Temporal, Prometheus, Grafana (native Docker Engine)
uv run python scripts/worker.py                        # in a second terminal: the pipeline worker
uv run python scripts/run_pipeline.py phase0 --wait    # the whole check as a Temporal workflow, both rooms

uv run python scripts/serve.py                         # server and UI on http://127.0.0.1:8000
uv run python scripts/simulate_phones.py 2             # two simulated phones streaming fr1/desk
```

**Prerequisites:** Linux, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12), an NVIDIA GPU with a recent driver, and Docker Engine for the stack. Developed on an RTX 4060 Laptop (8 GB).

## Documentation

Everything else is in [`docs/`](docs/README.md), one page per feature:

| Page | Covers |
|---|---|
| [Setup and configuration](docs/01-setup.md) | Install, `check_env.py`, settings precedence and every setting |
| [Geometry and camera conventions](docs/02-geometry.md) | Camera axes, `T_world_cam`, depth factor, per-sensor calibration, the consistency score |
| [Datasets](docs/03-datasets.md) | TUM presets, download and prepare, the TUM format, the toy dataset |
| [Data versioning with DVC](docs/04-data-versioning.md) | Tracked datasets, the local remote, the hardlink cache |
| [Replay and the Phase 0 check](docs/05-replay-and-pointcloud.md) | Playback, point-cloud fusion and the depth-consistency gate |
| [Experiment tracking and tracing](docs/06-experiment-tracking.md) | MLflow runs and tags, Langfuse |
| [Pipelines](docs/07-pipelines.md) | The Temporal `Phase0Pipeline` and its DAG |
| [Local stack](docs/08-local-stack.md) | Docker Compose services, ports, Grafana dashboards |
| [Server and UI](docs/09-server-and-ui.md) | Multi-phone WebSocket ingest, GPU worker, metrics, web UI |
| [Async processing](docs/10-async-processing.md) | asyncio for frames, Temporal for pipelines, no Celery |
| [Testing and code quality](docs/11-testing.md) | pytest, ruff, mypy, CI |
| [Architecture and project structure](docs/12-architecture.md) | Module map, design rules, file tree |
| Planned: [Phase 1](docs/20-planned-phase1-temporal-memory.md) · [Phases 2 to 5](docs/21-planned-perception-memory-query.md) · [phone app](docs/22-planned-arcore-phone-app.md) · [robot](docs/23-planned-robot-integration.md) | Goals of what isn't built yet |

## License

**Proprietary — All Rights Reserved.** Copyright (c) 2026 Omar Ayman Bakr.

This repository is public so that it can be **viewed**. Viewing it grants no rights. Without prior written permission you may not use, run, clone, copy, modify, fork outside GitHub, or redistribute any part of it, including the code, design documents, the Basir name and logo, and the 3D reconstructions. You also may not use it, or anything it produces, to train, fine-tune, evaluate or prompt AI / ML models, or include it in any dataset. A GitHub fork carries no right to use it.

See [LICENSE](LICENSE) for the full terms; to request permission, contact [@omarTBakr](https://github.com/omarTBakr).
