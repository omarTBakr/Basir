# Phase 1: L1 temporal memory

> **Planned, not built.** This page holds the goal only. It's filled in when the phase lands.

**Goal.** Give Basir its first memory: remember *when* it saw something. The camera stream is reduced to **keyframes**, each keyframe gets a **SigLIP** image embedding stored in **Qdrant**, and a text question is matched against them, so Basir can answer "your bottle was **seen 42 s ago**". A small **Gradio** UI is planned for asking questions.

**Status.** Not started. The groundwork is built: frames arrive through the `FrameSource` interface ([Datasets](03-datasets.md#reading-frames)) or the phone WebSocket ([Server and UI](09-server-and-ui.md)), the GPU worker's stub model is the slot the first real model goes into, and Langfuse tracing is wired in and waiting for the first query ([Experiment tracking](06-experiment-tracking.md#langfuse-tracing)).
