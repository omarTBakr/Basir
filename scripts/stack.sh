#!/usr/bin/env bash
# Manage Basir's local stack (MLflow, Temporal, Postgres, Prometheus, Grafana).
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
ENV_FILE="$ROOT/infra/stack.env"
CONTEXT="${BASIR_DOCKER_CONTEXT:-default}"
export HOST_UID="${HOST_UID:-$(id -u)}"

compose() {
    docker --context "$CONTEXT" compose --project-directory "$ROOT" --env-file "$ENV_FILE" "$@"
}

ensure_engine() {
    if ! docker --context "$CONTEXT" info >/dev/null 2>&1; then
        echo "Docker engine (context '$CONTEXT') is not reachable. Start it with:" >&2
        echo "  sudo systemctl start docker" >&2
        exit 1
    fi
}

ensure_env_file() {
    [[ -f "$ENV_FILE" ]] && return
    echo "Creating infra/stack.env with random passwords"
    sed -e "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" \
        -e "s/^GRAFANA_ADMIN_PASSWORD=.*/GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 12)/" \
        "$ROOT/infra/stack.env.example" >"$ENV_FILE"
    chmod 600 "$ENV_FILE"
}

print_urls() {
    local pg_port
    pg_port="$(sed -n 's/^POSTGRES_PORT=//p' "$ENV_FILE")"
    cat <<EOF

  MLflow          http://127.0.0.1:5000
  Temporal UI     http://127.0.0.1:8233   (gRPC 127.0.0.1:7233)
  Grafana         http://127.0.0.1:3000   (admin login in infra/stack.env)
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
        ensure_env_file
        compose up -d --build --wait --wait-timeout 300
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
