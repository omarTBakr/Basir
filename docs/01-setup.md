# Setup and configuration

How to install Basir on a fresh machine, check that it can see the GPU, and configure it. Every setting the built phases read is listed here.

## Prerequisites

| Need | Why |
|---|---|
| Linux | Developed and run on Fedora; the stack needs the native Docker Engine |
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.12 (pinned in `.python-version`) and every dependency from `uv.lock` |
| NVIDIA GPU with a recent driver | The models run on CUDA; developed on an RTX 4060 Laptop (8 GB) |
| Docker Engine + NVIDIA Container Toolkit | Only for the [local stack](08-local-stack.md) |

## Install

```bash
uv sync                                    # everything, incl. torch with CUDA (several GB)
cp env/.env.example env/.env               # then adjust (see the table below)
uv run python scripts/check_env.py         # versions, CUDA check, resolved paths
uv run python scripts/check_env.py --langfuse   # also send one smoke trace to Langfuse
```

`uv sync` installs the runtime dependencies plus the `dev` group (DVC, pytest, ruff, SciPy for the geometry cross-checks). Every script is run as `uv run python scripts/<name>.py`, so nothing is installed outside the project's `.venv`.

**`check_env.py`** prints the Python and package versions (torch, numpy, opencv-python, open3d, pydantic, tqdm, mlflow, langfuse), runs a small fp16 matmul on CUDA, and shows the resolved device, datasets folder, MLflow store and whether Langfuse is on. It exits with 1 when CUDA is missing, or when `--langfuse` is given and the smoke trace can't be sent.

## How settings are resolved

All settings live in one pydantic model, `Settings`, in [`utilities/config.py`](../utilities/config.py). Code reads them through `get_settings()`, which caches one instance (call `get_settings.cache_clear()` to reload).

Precedence, lowest to highest:

```text
field default  <  env file  <  <FIELD> environment variable  <  BASIR_<FIELD> environment variable  <  keyword arguments
```

- **Env file.** `env/.env` (git-ignored), copied from `env/.env.example`. Set `ENV_FILE` to read a different file instead.
- **`BASIR_` prefix.** `BASIR_DEVICE=cpu` wins over `DEVICE=cuda`. Use it to pin generic names such as `DEVICE` that other tools may also set.
- **Names are case-insensitive**, and an **empty value means "unset"**: `DATA_DIR=` falls back to the default rather than to an empty path.
- **Keyword arguments** (`Settings(device="cpu")`) are for tests.

Only what the built phases use is configurable. Each later phase adds its own settings when it's built.

## Every setting

| Setting | Default | Does what |
|---|---|---|
| `APP_ENV` | `development` | `development`, `staging` or `production`; also tags Langfuse traces |
| `DEVICE` | `auto` | `auto`, `cuda`, `cpu` or `mps`; `auto` picks CUDA, then MPS, then CPU |
| `DATA_DIR` | the project folder | Root for `datasets/`, `outputs/`, `mlflow.db` and `mlartifacts/`; a relative path starts at the project root |
| `MLFLOW_TRACKING_URI` | `sqlite:///<DATA_DIR>/mlflow.db` | Where runs are recorded; `http://127.0.0.1:5000` for the local stack. An `http://` URI also turns on the UI's run links |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | unset | Runtime tracing; off until **both** are set |
| `LANGFUSE_BASE_URL` | Langfuse Cloud | Langfuse server URL |
| `AIRFLOW_URL` | `http://127.0.0.1:8080` | Airflow API server and UI that `run_pipeline.py` and the Console's Pipelines tab use |
| `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | `admin`, blank | Airflow user for the REST API; a blank password means `AIRFLOW_ADMIN_PASSWORD` from `env/stack.env` |
| `GRAFANA_URL` | `http://127.0.0.1:3000` | Grafana, embedded on the Dashboards page |
| `PROMETHEUS_URL` | `http://127.0.0.1:9090` | Prometheus of the local stack |
| `INGEST_QUEUE_SIZE` | `2` | Frames kept per phone before the oldest is dropped (≥ 1) |
| `BATCH_MAX_SIZE` | `8` | Most frames per GPU micro-batch (≥ 1) |
| `BATCH_MAX_WAIT_MS` | `20` | Longest wait after a batch's first frame before it's sent (≥ 0) |
| `INGEST_HOST`, `INGEST_PORT` | `0.0.0.0`, `8001` | The phone-only listener (phone page, pairing, stream); the Console stays on `127.0.0.1:8000` |
| `INGEST_TLS` | `true` | HTTPS with a self-signed certificate on the phone listener (phone browsers need it for the camera and AR) |
| `MLFLOW_EMBED_PORT`, `AIRFLOW_EMBED_PORT` | `8002`, `8003` | Loopback ports that serve frameable copies of the MLflow and Airflow UIs |
| `VIEW_MAX_FPS` | `15` | Most frames per second sent to a Live view |
| `MAP_VOXEL_M` | `0.02` | Live map voxel size in metres |
| `MAP_KEYFRAME_M`, `MAP_KEYFRAME_DEG` | `0.05`, `5` | A frame is fused into the live map after the camera moved this far or turned this much |
| `MAP_MAX_POINTS` | `2000000` | Most voxels per live map |

Derived values are read-only properties on `Settings`, not settings: `datasets_dir`, `outputs_dir`, `mlflow_uri`, `mlflow_artifacts_dir`, `mlflow_ui_url`, `devices_file`, `certs_dir`, `live_map_dir`, `langfuse_enabled`, `effective_device` (with `auto` resolved) and `default_camera_intrinsics` (the TUM / Kinect v1 camera, used when a dataset has no calibration of its own; see [Geometry](02-geometry.md#camera-intrinsics)).

## Stack settings

The Docker stack has its own env file, `env/stack.env` (git-ignored). `scripts/stack.sh up` creates it from `env/stack.env.example` the first time, with random passwords. It's covered in [Local stack](08-local-stack.md#stack-settings).

Notes:

- **Secrets stay local.** `env/.env` and `env/stack.env` are git-ignored; only the `.example` files are committed.
- **Without the stack** everything still works: runs go to the local SQLite file, Langfuse stays off, and the server UI marks the missing services as *not reachable*.
