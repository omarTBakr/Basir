#!/bin/sh
# Migrate the Postgres backend to this MLflow version, then serve.
# Artifacts are proxied by the server into /mlartifacts (a host folder).
# --allowed-hosts: MLflow rejects unknown Host headers (DNS-rebinding guard); Airflow's tasks
# reach it as mlflow:5000 inside the compose network, everything else as 127.0.0.1 / localhost.
set -eu

mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"

exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --serve-artifacts \
    --artifacts-destination /mlartifacts \
    --default-artifact-root mlflow-artifacts:/ \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"
