"""The Basir server: multi-phone WebSocket ingest, a GPU worker process, Prometheus metrics and the web UI.

Modules:
    protocol       the phone <-> server wire format (binary JPEG + JSON header)
    framebus       per-phone latest-frame-wins queues behind a swappable FrameBus interface
    worker         the spawned GPU process (InferenceBackend) and the stub model
    batcher        micro-batching across phones and result routing
    devices        connected phones and their rolling statistics
    ingest         the /ws/stream/{device_id} endpoint
    metrics        Prometheus metric definitions, HTTP timing and /metrics
    middleware     request IDs for HTTP and WebSocket
    dag            pipeline DAG parsing and layout
    mlflow_view    the MLflow summary layer (running, latest, pipelines)
    temporal_view  running and recent Temporal workflows
    services       Grafana / Prometheus reachability
    grafana        embedded dashboard and panel configuration
    ui             page and HTMX fragment routes
    app            the app factory and lifespan
"""
