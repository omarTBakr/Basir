# Software engineering docs

UML for Basir, written in PlantUML. Every `.puml` has its rendered `.png` committed next to it.
Phase 0 code (`basir/core`, `basir/ingest`, `utilities/`) is in place on the [`phase_0` branch](https://github.com/omarTBakr/Basir/tree/phase_0); the other diagrams describe the **planned** design from the project's design plan,
plus the deployment decision taken since: a realme phone running ARCore records and displays, and
the laptop (RTX 4060, 8 GB) is the server. Class and module names match the planned package layout. Update a
diagram in the same change as the code that makes it true or different.

## Implementation status (living section, updated as work lands)

_Last updated: 2026-11-10._ ✅ built on [`phase_0`](https://github.com/omarTBakr/Basir/tree/phase_0) · 🔄 being built on a work branch · ⏳ designed, not built.

| Area | Diagrams | Status |
|---|---|---|
| Core data model and geometry | [core_types](4_class_diagrams/core_types.png) | ✅ `Frame`, `Pose`, `CameraIntrinsics`, geometry; `Observation` / `ObjectTrack` / `QueryResult` ⏳ |
| Frame sources and datasets | [ingest_and_spatial](4_class_diagrams/ingest_and_spatial.png) | ✅ `FrameSource`, `TUMDatasetSource`, dataset prep; `ARCoreStreamSource` / `RecordingSource` ⏳ |
| Data versioning | — | 🔄 DVC done on `dvc_setup` (awaiting review): one `.dvc` per TUM sequence, local remote `dvc-remotes/Basir` |
| Replay and evaluation | [replay_eval](3_sequence_diagrams/replay_eval.png), [evaluation](2_use_case_diagrams/evaluation.png) | ✅ replay, point-cloud check with MLflow; 🔄 Temporal pipeline DAG (`pipeline_dag`) |
| Multi-phone streaming | [phone_stream](3_sequence_diagrams/phone_stream.png), [phone_app](2_use_case_diagrams/phone_app.png) | 🔄 server-side ingest + GPU micro-batching (`server_ui`); phone app ⏳ |
| Server, UI and observability | system overviews | 🔄 FastAPI UI, Prometheus, Grafana (`server_ui`, `infra_stack`) |
| Memory building, querying, guidance | [ingest_keyframe](3_sequence_diagrams/ingest_keyframe.png), [query_where_is](3_sequence_diagrams/query_where_is.png), [guidance_reacquire](3_sequence_diagrams/guidance_reacquire.png), [perception](4_class_diagrams/perception.png), [memory](4_class_diagrams/memory.png), [query_and_guidance](4_class_diagrams/query_and_guidance.png) | ⏳ Phases 1–5 |
| Robot integration | [06_robot_integration](1_user_stories/06_robot_integration.png) | ⏳ Phase 7 |

**Diagrams to add once the work branches merge:** the async ingest design (per-phone latest-frame queues → spawned GPU worker with micro-batching), the Temporal pipeline DAG and its MLflow parent/child runs, and the deployment view (Docker Compose: Postgres, MLflow, Temporal, Prometheus, Grafana; FastAPI server; phones).

## Observability

Two tools, two questions. **MLflow** answers "did this change make Basir better?": one run per script or evaluation, with params, metrics and artifacts (`utilities/experiment.py`). **Langfuse** answers "what happened in this one request?": a trace per query with each step, LLM/VLM calls, latency and cost (`utilities/tracing.py`, off until keys are set).

## Rendering

```bash
software_engineering/render.sh                       # every diagram
software_engineering/render.sh 3_sequence_diagrams/query_where_is.puml
```

Rendering needs `java` and graphviz (`dot`). On first run `render.sh` downloads the pinned
PlantUML jar (1.2026.9, checked against its published SHA-256) into `.tools/`, which is
gitignored, so nothing is installed outside the repository. Set `PLANTUML_JAR` to use a
different jar. `_style.iuml` holds the shared look.

## 1. User stories — `1_user_stories/`

| Epic | Diagram |
| --- | --- |
| Building the spatial memory | [01_memory_building](1_user_stories/01_memory_building.png) |
| Asking "where is my ...?" (L1 temporal, L2 relative) | [02_finding_objects](1_user_stories/02_finding_objects.png) |
| Guidance, re-acquisition, moved objects | [03_guidance_and_reacquisition](1_user_stories/03_guidance_and_reacquisition.png) |
| Phone app: record, stream, display | [04_phone_app](1_user_stories/04_phone_app.png) |
| Evaluation and development on the laptop | [05_evaluation_and_development](1_user_stories/05_evaluation_and_development.png) |
| Robot integration (later phase, L3) | [06_robot_integration](1_user_stories/06_robot_integration.png) |

## 2. Use case diagrams — `2_use_case_diagrams/`

| Scope | Diagram |
| --- | --- |
| **Whole system** | [system_overview](2_use_case_diagrams/system_overview.png) |
| Building the spatial memory | [memory_building](2_use_case_diagrams/memory_building.png) |
| Asking where an object is | [querying](2_use_case_diagrams/querying.png) |
| Guidance and re-acquisition | [guidance](2_use_case_diagrams/guidance.png) |
| Phone app | [phone_app](2_use_case_diagrams/phone_app.png) |
| Evaluation and development | [evaluation](2_use_case_diagrams/evaluation.png) |

## 3. Sequence diagrams — `3_sequence_diagrams/`

| Flow | Diagram |
| --- | --- |
| **Whole system**: remember → ask → directions → found it | [system_overview](3_sequence_diagrams/system_overview.png) |
| Frame → keyframe → detect / segment / embed → 3D → fusion | [ingest_keyframe](3_sequence_diagrams/ingest_keyframe.png) |
| Query: parse → retrieve → VLM verify → rank → directions | [query_where_is](3_sequence_diagrams/query_where_is.png) |
| Guidance loop: live directions, re-acquisition, tracking, stale | [guidance_reacquire](3_sequence_diagrams/guidance_reacquire.png) |
| Phone app: ARCore session, WebSocket stream, answers back | [phone_stream](3_sequence_diagrams/phone_stream.png) |
| Replay a session and score it | [replay_eval](3_sequence_diagrams/replay_eval.png) |

## 4. Class diagrams — `4_class_diagrams/`

| Scope | Diagram |
| --- | --- |
| **Whole system**, packages and main classes | [system_overview](4_class_diagrams/system_overview.png) |
| Core data model and geometry (`basir.core`) | [core_types](4_class_diagrams/core_types.png) |
| Frame sources, keyframes, localization (`basir.ingest`, `basir.spatial`) | [ingest_and_spatial](4_class_diagrams/ingest_and_spatial.png) |
| Model adapters (`basir.perception`) | [perception](4_class_diagrams/perception.png) |
| Fusion, Qdrant store, keyframe store (`basir.memory`) | [memory](4_class_diagrams/memory.png) |
| Query engine, LLM clients, guidance, server (`basir.query`, `basir.guidance`, `basir.api`) | [query_and_guidance](4_class_diagrams/query_and_guidance.png) |
