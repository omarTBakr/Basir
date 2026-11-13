#!/usr/bin/env bash
# Manage Basir's local stack (MLflow, Airflow, Postgres, Prometheus, Grafana).
#
#   scripts/stack.sh up        # build, start and wait until healthy
#   scripts/stack.sh down      # stop (volumes are kept)
#   scripts/stack.sh status    # containers, URLs and Prometheus target health
#   scripts/stack.sh logs [service...]
#
# The stack runs on the native Docker Engine (context "default"), which has the NVIDIA
# runtime; Docker Desktop's VM cannot reach the GPU. Override with BASIR_DOCKER_CONTEXT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/Docker/docker-compose.yml"
ENV_FILE="$ROOT/env/stack.env"
CONTEXT="${BASIR_DOCKER_CONTEXT:-default}"
export HOST_UID="${HOST_UID:-$(id -u)}"

compose() {
    ensure_env_file
    export_artifacts_dir
    export_data_dir
    docker --context "$CONTEXT" compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
}

# Compose resolves relative bind paths from each service's folder (Docker/<service>/), so a
# relative MLFLOW_ARTIFACTS_DIR from env/stack.env is made absolute from the project root
# here. A variable exported in the shell wins over the --env-file value.
export_artifacts_dir() {
    local dir
    dir="${MLFLOW_ARTIFACTS_DIR:-$(sed -n 's/^MLFLOW_ARTIFACTS_DIR=//p' "$ENV_FILE" 2>/dev/null | tail -n 1)}"
    dir="${dir:-mlartifacts}"
    [[ "$dir" == /* ]] || dir="$ROOT/${dir#./}"
    export MLFLOW_ARTIFACTS_DIR="$dir"
}

# Airflow mounts datasets/ and outputs/ at their host paths, so paths it logs to MLflow are
# valid on the host: DATA_DIR from env/.env (relative = from the project root), else the root.
export_data_dir() {
    local dir
    dir="${BASIR_DATA_DIR:-$(sed -n 's/^DATA_DIR=//p' "$ROOT/env/.env" 2>/dev/null | tail -n 1)}"
    dir="${dir:-$ROOT}"
    [[ "$dir" == /* ]] || dir="$ROOT/${dir#./}"
    mkdir -p "$dir/datasets" "$dir/outputs"
    export BASIR_DATA_DIR="$dir"
}

ensure_engine() {
    if ! docker --context "$CONTEXT" info >/dev/null 2>&1; then
        echo "Docker engine (context '$CONTEXT') is not reachable. Start it with:" >&2
        echo "  sudo systemctl start docker" >&2
        exit 1
    fi
}

ensure_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Creating env/stack.env with random passwords"
        sed -e "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" \
            -e "s/^GRAFANA_ADMIN_PASSWORD=.*/GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 12)/" \
            "$ROOT/env/stack.env.example" >"$ENV_FILE"
        chmod 600 "$ENV_FILE"
    fi
    ensure_airflow_secrets
}

# Fill in Airflow's secrets: on a new env/stack.env (the example has them blank) and on one made
# before Airflow joined the stack (the keys are missing). Existing values are never changed.
ensure_airflow_secrets() {
    local key value
    for key in AIRFLOW_ADMIN_PASSWORD AIRFLOW_FERNET_KEY AIRFLOW_JWT_SECRET AIRFLOW_API_SECRET_KEY; do
        [[ -n "$(sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1)" ]] && continue
        case "$key" in
            AIRFLOW_ADMIN_PASSWORD) value="$(openssl rand -hex 12)" ;;
            AIRFLOW_FERNET_KEY) value="$(openssl rand -base64 32 | tr '+/' '-_')" ;;  # url-safe base64, 32 bytes
            *) value="$(openssl rand -hex 32)" ;;
        esac
        if grep -q "^${key}=" "$ENV_FILE"; then
            sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        else
            printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
        fi
        echo "Generated $key in env/stack.env"
    done
}

print_urls() {
    local pg_port
    pg_port="$(sed -n 's/^POSTGRES_PORT=//p' "$ENV_FILE")"
    cat <<EOF

  MLflow          http://127.0.0.1:5000
  Airflow         http://127.0.0.1:8080   (user admin, AIRFLOW_ADMIN_PASSWORD in env/stack.env)
  Grafana         http://127.0.0.1:3000   (admin login in env/stack.env)
  Prometheus      http://127.0.0.1:9090
  Postgres        127.0.0.1:${pg_port:-5433}
EOF
}

print_targets() {
    local targets
    echo
    echo "Prometheus targets:"
    if ! targets="$(curl -sf http://127.0.0.1:9090/api/v1/targets)"; then
        echo "  Prometheus is not answering on :9090"
        return
    fi
    python3 -c '
import json, sys
for t in sorted(json.load(sys.stdin)["data"]["activeTargets"], key=lambda t: t["labels"]["job"]):
    job, health, url, err = t["labels"]["job"], t["health"], t["scrapeUrl"], t["lastError"]
    note = "  (" + err + ")" if health != "up" and err else ""
    print(f"  {job:<11} {health:<7} {url}{note}")
' <<<"$targets"
}

usage() {
    echo "usage: $0 up|down|status|logs [service...]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
command="$1"
shift

case "$command" in
    up)
        ensure_engine
        compose up -d --build --wait --wait-timeout 600
        print_urls
        ;;
    down)
        ensure_engine
        compose down
        ;;
    status)
        ensure_engine
        compose ps --format 'table {{.Service}}\t{{.Image}}\t{{.Status}}'
        print_urls
        print_targets
        ;;
    logs)
        ensure_engine
        compose logs --tail 100 -f "$@"
        ;;
    *)
        usage
        ;;
esac
