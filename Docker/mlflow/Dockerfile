# MLflow tracking server with the Postgres driver (the upstream image ships without it).
FROM ghcr.io/mlflow/mlflow:v3.18.0

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache psycopg2-binary==2.9.13

COPY entrypoint.sh /usr/local/bin/mlflow-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/mlflow-entrypoint.sh"]
