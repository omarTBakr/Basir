"""The Basir Console: multi-phone WebSocket ingest, a GPU worker process, live view and map, metrics and the web UI.

Modules:
    app             the Console app factory and lifespan (GPU worker, batcher, live maps)
    protocol        the phone <-> server wire format (JSON header + JPEG + optional uint16 depth), device ids
    ingest          /ws/stream/{device_id}, and the phone-only LAN app (phone page, pairing, token check)
    framebus        per-phone latest-frame-wins queues behind a swappable FrameBus interface
    batcher         micro-batching across phones and result routing
    worker          the spawned GPU process (InferenceBackend, restart, basir_worker_alive) and the stub model
    devices         connected phones and their rolling statistics
    rgbd            decode phone colour/depth; encode depth PNGs and point batches for browsers
    viewer          /ws/view/{device_id}: a phone's newest colour + depth for the Live tab
    mapping         /ws/map/{device_id} and /api/map/*: per-phone live voxel maps, save as MLflow runs
    pairing         pairing requests: a phone asks over Wi-Fi, the Devices tab approves it
    tokens          per-phone tokens for the LAN listener, stored as SHA-256 hashes, with expiry
    tls             the phone listener's self-signed certificate
    lan             this laptop's LAN / hotspot addresses (for the QR codes)
    devices_page    the Devices tab: QR codes, pairing approvals, paired and connected phones
    phone_simulator simulated phones replaying a dataset over /ws/stream
    embed           frameable loopback proxies of the MLflow and Airflow UIs
    tabs            the Console's tab registry (sidebar, routes, "coming later" pages)
    metrics         Prometheus metric definitions, HTTP timing and /metrics
    middleware      request IDs, Host/Origin checks, log redaction
    dag             pipeline DAG parsing and layout for the UI
    mlflow_view     the MLflow summary layer (running, latest, pipelines)
    airflow_view    running and recent Airflow DAG runs and their task states
    services        Grafana / Prometheus reachability
    grafana         embedded dashboard and panel configuration
    ui              page and HTMX fragment routes

Phone messages become domain `Frame`s through `basir.ingest.phone_frame.frame_from_message`.
"""
