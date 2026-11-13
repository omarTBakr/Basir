#!/usr/bin/env bash
# Create the MLflow and Temporal databases on the first start of an empty Postgres volume.
set -euo pipefail

for db in mlflow temporal temporal_visibility; do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -c "CREATE DATABASE ${db} OWNER \"${POSTGRES_USER}\";"
done
