# ARCore phone app

> **Planned, not built.** This page holds the goal only. It's filled in when the app lands.

**Goal.** An Android app on an ARCore phone (realme 14 / 15) that records video together with its own 6-DoF pose and depth, streams them to the laptop server, and shows Basir's answers and directions. This is the phone side of Phase 6, the real-time live pipeline.

The phone needs ARCore with the **Depth API** for phases 3 and later (realme 14 / 15 qualify). A phone with ARCore but no Depth API (for example the Redmi Note 10S) still gives the pose, and the laptop then estimates depth with Depth Anything.

**Status.** No app is needed to start: the laptop serves a **phone page** that pairs and streams from Chrome, with colour, depth and position through WebXR on ARCore phones ([Connect a phone](09-server-and-ui.md#connect-a-phone)). The server side is built: the phone-only Wi-Fi listener (HTTPS, per-phone tokens, pairing by code), the wire format (protocol v2: JPEG colour + 16-bit depth in millimetres + pose + intrinsics, see [Wire protocol](09-server-and-ui.md#wire-protocol)), the GPU worker, the live colour + depth view, the live 3D map and the Console. The native app is planned for what a browser can't do well: higher frame rates, full-resolution ARCore depth, streaming with the screen off; it will use the same pairing (`POST /pair`, `GET /pair/{id}`) and protocol.
