#!/usr/bin/env bash
# Create the MLflow and Airflow databases on the first start of an empty Postgres volume.
# (airflow-init also creates `airflow` on a volume made before Airflow joined the stack.)
set -euo pipefail

for db in mlflow airflow; do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -c "CREATE DATABASE ${db} OWNER \"${POSTGRES_USER}\";"
done
