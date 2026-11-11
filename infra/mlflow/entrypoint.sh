#!/bin/sh
# Migrate the Postgres backend to this MLflow version, then serve.
# Artifacts are proxied by the server into /mlartifacts (a host folder).
set -eu

mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"

exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --serve-artifacts \
    --artifacts-destination /mlartifacts \
    --default-artifact-root mlflow-artifacts:/ \
    --host 0.0.0.0 \
    --port 5000
