# Airflow 3 with Basir's dependencies in a second Python environment.
#
# Airflow's own environment (from the official image) parses dags/ and runs the task wrappers;
# Basir's (/opt/basir-venv, from uv.lock, CPU torch) runs the steps, so the two never share a
# dependency (Airflow pins fastapi, pydantic and SQLAlchemy versions Basir has moved past).
# Basir's code is not baked in: compose mounts basir/, utilities/ and dags/ at /opt/basir.
# Build context = the project root (see compose.yml); Dockerfile.dockerignore keeps it to the lock files.
FROM apache/airflow:3.3.2-python3.12

USER root
# open3d needs EGL/GL, OpenMP and libusb (RealSense support); OpenCV needs GL and GLib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 libglib2.0-0 libgomp1 libusb-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/basir-venv \
    UV_PYTHON=/usr/python/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1
WORKDIR /opt/basir-build
COPY pyproject.toml uv.lock ./
COPY --chmod=0755 Docker/airflow/install-basir-env.sh /usr/local/bin/install-basir-env.sh
RUN install-basir-env.sh && rm -rf /opt/basir-build

COPY --chmod=0755 Docker/airflow/entrypoint.sh /usr/local/bin/basir-airflow-entrypoint.sh
COPY Docker/airflow/ensure_database.py /usr/local/bin/ensure_database.py

USER airflow
# pytest in Airflow's environment, for tests/test_dags.py (DagBag import checks); pinning
# apache-airflow keeps pip from touching anything Airflow depends on.
RUN pip install --no-cache-dir "apache-airflow==3.3.2" "pytest==9.2.0"
WORKDIR /opt/airflow
ENV BASIR_PYTHON=/opt/basir-venv/bin/python \
    PYTHONPATH=/opt/basir
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/usr/local/bin/basir-airflow-entrypoint.sh"]
