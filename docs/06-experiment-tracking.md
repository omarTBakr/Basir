# Experiment tracking and tracing

Every run leaves a record. **MLflow** answers "did this change make Basir better?" and **Langfuse** will answer "what happened inside this one request?". Scripts never import either directly: they go through [`utilities/experiment.py`](../utilities/experiment.py) and [`utilities/tracing.py`](../utilities/tracing.py), so the store, the standard tags and the failure handling live in one place.

## Observability at a glance

| Tool | Answers | Records | Where |
|---|---|---|---|
| **MLflow** | "Did this change make Basir better?" | One run per script or evaluation: params, metrics, artifacts, git commit, GPU | The stack's MLflow server on Postgres, or local SQLite (`mlflow.db`); artifacts in `mlartifacts/` either way |
| **Airflow** | "Where is this pipeline, and what failed?" | One DAG run per pipeline run: each task's state, attempts, timing and log | The [local stack](08-local-stack.md), UI on port 8080 ([Pipelines](07-pipelines.md)) |
| **Langfuse** | "What happened in this one request?" | One trace per query: each step, LLM / VLM calls, latency and cost | Langfuse Cloud for now; self-hosting later |
| **Prometheus + Grafana** | "Is the machine keeping up?" | Host, GPU, Postgres, Airflow and Basir server metrics over time | The [local stack](08-local-stack.md) |

## MLflow

### Where runs go

| `MLFLOW_TRACKING_URI` | Store | Artifacts |
|---|---|---|
| unset (default) | `sqlite:///<DATA_DIR>/mlflow.db` | `<DATA_DIR>/mlartifacts/` |
| `http://127.0.0.1:5000` | The stack's MLflow server, on Postgres | Stored by the server in a host folder mounted into its container (`mlartifacts/` by default, `MLFLOW_ARTIFACTS_DIR` in the stack settings) |

With an `http://` URI the server UI and `run_pipeline.py` also print direct links to runs. The tests always use a temporary SQLite store.

### Experiments

| Experiment | Written by | One run per |
|---|---|---|
| `phase0-replay` | `scripts/replay.py` | Replay of one sequence |
| `phase0-pointcloud` | `scripts/visualize_pointcloud.py` | Point-cloud check of one sequence |
| `phase0-pipeline` | The Airflow DAG `phase0` | Pipeline run (parent) and each step (child) |

### API

```python
from utilities.experiment import start_run, log_params, log_metrics, log_artifact

with start_run("phase0-pointcloud", run_name="fr1_room", tags={"dataset": "fr1/room"}):
    log_params({"every": 15})
    log_metrics({"points": 123456})
    log_artifact("outputs/fr1_room.ply")
```

| Function | Does |
|---|---|
| `start_run(experiment, run_name, tags, parent_run_id)` | Context manager; creates the experiment if needed, yields the run ID, ends the run `FAILED` if the block raises. With `parent_run_id` the run is nested |
| `log_params`, `log_metrics`, `log_artifact`, `set_tags` | Log to the current thread's run |
| `open_run(...)` / `close_run(run_id, failed, tags)` | A run that stays open across calls, for a pipeline's parent run |
| `run_url(run_id)` | Web link to a run when tracking goes to an MLflow server |

**Logging never stops a run.** If the database is locked or the server is unreachable, Basir logs one warning and carries on without tracking. Runs are tracked per thread, so parallel pipeline steps in one worker don't interfere.

### Standard tags

Every run gets these, enough to reproduce it:

| Tag | Value |
|---|---|
| `basir_version` | The package version |
| `device` | The resolved compute device |
| `git_commit` | Short commit hash (which also pins the [DVC](04-data-versioning.md) dataset pointers) |
| `git_dirty` | Whether the working tree had uncommitted changes |
| `gpu` | GPU name, when CUDA is available |

### Pipeline runs: parent and child

A pipeline run is one **parent** run with one nested **child** run per step. The tags are the contract the server UI reads to draw the Pipelines page.

| Run | Tags | Artifacts |
|---|---|---|
| parent | `basir.pipeline` (`phase0`), `basir.workflow_id` (the Airflow DAG run's `run_id`), `basir.status` (`running` → `succeeded` / `failed`) | `dag.json`, `dag.mmd` |
| child (per step) | `mlflow.parentRunId`, `basir.step` (e.g. `pointcloud:fr1/room`), `basir.depends_on` (comma-separated step names), `basir.status`, `basir.dataset` (when the step has one) | What the step produces, e.g. the `.ply`, top-down image and trajectory of a `pointcloud` step |

`dag.json` has the shape `{"nodes": [{"id", "activity", "dataset"}], "edges": [{"source", "target"}]}`; `dag.mmd` is the same DAG as a Mermaid flowchart. See [Pipelines](07-pipelines.md).

## Langfuse tracing

```python
from utilities.tracing import observe, flush

@observe(name="query.retrieve")
def retrieve(text): ...
```

- **Off without keys.** Tracing turns on only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Without them `@observe` returns the function unchanged and nothing leaves the machine, so the same code runs with or without tracing.
- **No frames, ever.** Input / output capture is off by default because Basir functions often receive frames and images. Turn it on per function with `@observe(..., capture=True)`.
- **Never breaks the app.** A client that fails to start logs a warning and tracing stays off.
- **`APP_ENV`** is sent as the trace environment; `LANGFUSE_BASE_URL` picks the server (default: Langfuse Cloud).
- Short-lived scripts call `flush()` before exiting.

`uv run python scripts/check_env.py --langfuse` checks the keys and sends one smoke trace, `basir.smoke`. The first real traces come with the query path in a later phase.
