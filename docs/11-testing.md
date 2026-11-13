# Testing and code quality

## Tests

```bash
uv run pytest -q                              # offline suite (toy dataset, local HTTP server, temporary MLflow store)
uv run pytest -q -m "not network and not slow" # quick loop: skips spawned processes, live servers, Temporal
uv run pytest -q -m network                   # live checks: every TUM preset URL, Langfuse auth, local stack targets
```

The suite is configured in [`pyproject.toml`](../pyproject.toml) (`[tool.pytest.ini_options]`): tests live in `tests/`, the project root is on the import path, and **`network` tests are deselected by default** (`addopts = "-m 'not network'"`).

| Marker | Runs | Needs |
|---|---|---|
| (none) | By default | Nothing: no internet, no GPU required, no running services |
| `slow` | By default (skip with `-m "not network and not slow"`) | Time: the spawned GPU worker (imports torch), uvicorn live servers, the phone simulator, LAN pairing, map save (open3d + MLflow), the Temporal test server in `test_phase0` (downloaded once, then cached) |
| `gpu` | By default | A CUDA GPU. No test needs one yet; reserved for Phase 1 perception |
| `network` | Only with `-m network` | The internet, real Langfuse keys in `env/.env` (skipped without them), and the [local stack](08-local-stack.md) running |

The `network` tests check that every TUM preset URL still answers, that the Langfuse keys authenticate, and that every Prometheus target except the Basir server is up.

### How the offline suite stays offline

Shared fixtures are in [`tests/conftest.py`](../tests/conftest.py):

| Fixture | Gives |
|---|---|
| `isolated_settings` (autouse) | Every test: `ENV_FILE=/dev/null`, `BASIR_DATA_DIR` in a temporary folder and `BASIR_MLFLOW_TRACKING_URI` pointing at a SQLite file there, so no test can reach the developer's `env/.env`, MLflow server or data drive even without `clean_env` |
| `clean_env` | An environment with every Basir setting (plain, `BASIR_`-prefixed and `ENV_FILE`) removed and the settings cache reset, so a developer's environment never leaks into a test |
| `file_server` | A local HTTP server with in-memory files, `Range` support and dropped connections on demand, for the download and resume code |
| `backend` | A fake GPU worker (no process, no torch) that records batch sizes and can hold batches back |
| `client` | A FastAPI `TestClient` on the app with the fake worker and a temporary data folder |
| `live_server` | The app served by uvicorn on a free port, for real WebSocket clients such as the phone simulator |

Datasets come from the [toy dataset](03-datasets.md#the-toy-dataset), MLflow writes to a temporary SQLite store, and Langfuse stays off.

There's **one test file per module**. **The geometry tests are mandatory**: every later phase depends on them (see [Geometry](02-geometry.md#tests)).

## Lint and format

```bash
uv run ruff check .          # lint (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, ruff-specific)
uv run ruff format --check . # formatting
```

Ruff is pinned in the `dev` group and configured in `pyproject.toml`: line length 120, target Python 3.12, rule sets `E`, `F`, `W`, `I`, `B`, `UP`, `SIM`, `RUF`, and isort with `force-sort-within-sections`.

## Type checking

The code is fully type-annotated (`from __future__ import annotations` everywhere, `Protocol`s for the swappable seams such as `FrameSource`, `FrameBus` and `InferenceBackend`). **mypy** is in the `dev` group and configured in `pyproject.toml` (`[tool.mypy]`: Python 3.12, `warn_unused_ignores`, `check_untyped_defs`; `ignore_missing_imports` only for the untyped third-party packages listed there, and `types-tqdm` for tqdm):

```bash
uv run mypy basir utilities scripts   # must report no issues; CI fails otherwise
```

## CI and pre-commit

- **GitHub Actions** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)): on every push and pull request, except the docs-only `software_engineering` branch. Three jobs:
  - **ruff:** `ruff check` and `ruff format --check`.
  - **pytest:** the full suite (including `slow`), with no GPU and an empty `BASIR_DATA_DIR`. It installs `libegl1 libgl1 libgomp1` first, which open3d and OpenCV need to import on `ubuntu-latest`. The environment uses CPU-only torch ([`.github/scripts/sync-cpu-env.sh`](../.github/scripts/sync-cpu-env.sh)), which skips about 2.9 GB of CUDA wheels; the local lock is unchanged.
  - **mypy:** blocking, `uv run mypy basir utilities scripts` with the settings above.
- **pre-commit** ([`.pre-commit-config.yaml`](../.pre-commit-config.yaml)): ruff (fix + format) and file hygiene (trailing whitespace, final newline, YAML, large files, private keys) on every commit; the full test suite on every push. Turn it on once per clone:

  ```bash
  uv run pre-commit install --hook-type pre-commit --hook-type pre-push
  ```
