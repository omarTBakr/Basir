# ARCore phone app

> **Planned, not built.** This page holds the goal only. It's filled in when the app lands.

**Goal.** An Android app on an ARCore phone (realme 14 / 15) that records video together with its own 6-DoF pose and depth, streams them to the laptop server, and shows Basir's answers and directions. This is the phone side of Phase 6, the real-time live pipeline.

The phone needs ARCore with the **Depth API** for phases 3 and later (realme 14 / 15 qualify). A phone with ARCore but no Depth API (for example the Redmi Note 10S) still gives the pose, and the laptop then estimates depth with Depth Anything.

**Status.** The server side is built: multi-phone WebSocket ingest, the wire format the app will speak, the GPU worker, metrics and the UI, tested with simulated phones ([Server and UI](09-server-and-ui.md)). The app itself is planned.
