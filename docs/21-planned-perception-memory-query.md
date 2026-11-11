# Phases 2 to 5: perception, spatial memory, query and guidance

> **Planned, not built.** This page holds the goals only. Each section is filled in when its phase lands.

No model is trained from scratch: detection, segmentation, embeddings and depth come from pretrained models (Grounding DINO / YOLO-World, SAM 2/3, SigLIP 2, Depth Anything). The new work is the memory, the 3D fusion and the guidance.

## Phase 2: object-level memory

**Goal.** Remember objects, not whole frames: open-vocabulary detection, SAM masks, per-object embeddings and 2D fusion of repeated sightings.

**Status.** Planned.

## Phase 3: L2 spatial memory

**Goal.** Remember *where* each object is: 3D object positions, 3D fusion, and directions as a bearing and distance from where you stand ("turn right ~120°, go about 2.3 m").

**Status.** Planned. It builds directly on the Phase 0 geometry ([Geometry](02-geometry.md), [Replay and point clouds](05-replay-and-pointcloud.md)).

## Phase 4: re-acquisition and segmentation

**Goal.** Live guidance that updates as you move, a "Found it!" mask once the object is back in view, tracking, and handling of objects that have moved.

**Status.** Planned.

## Phase 5: smart querying

**Goal.** Understand free-form questions: LLM query parsing, hybrid retrieval and VLM reranking.

**Status.** Planned.
