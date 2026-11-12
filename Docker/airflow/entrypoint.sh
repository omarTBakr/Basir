#!/usr/bin/env bash
# Write the admin's password (from env/stack.env) where Airflow's simple auth manager reads it,
# then hand over to the official entrypoint. `init` creates the `airflow` database when the
# Postgres volume predates it, then migrates it.
set -euo pipefail

python - <<'EOF'
import json, os
path = os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"]
with open(path, "w") as f:
    json.dump({"admin": os.environ["AIRFLOW_ADMIN_PASSWORD"]}, f)
os.chmod(path, 0o600)
EOF

if [[ "${1:-}" == "init" ]]; then
    python /usr/local/bin/ensure_database.py
    exec /entrypoint airflow db migrate
fi
exec /entrypoint "$@"
