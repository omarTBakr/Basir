"""Create Airflow's metadata database on the stack's Postgres if it is missing.

Docker/postgres/init-databases.sh creates it on a fresh volume; this covers volumes made
before Airflow joined the stack. Connects with the same superuser as MLflow.
"""

from __future__ import annotations

import os
import time

import psycopg2
from psycopg2 import sql

DATABASE = "airflow"


def main() -> None:
    params = {
        "host": os.environ.get("AIRFLOW_DB_HOST", "postgres"),
        "port": int(os.environ.get("AIRFLOW_DB_PORT", "5432")),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": "postgres",
    }
    for attempt in range(30):
        try:
            conn = psycopg2.connect(**params)
            break
        except psycopg2.OperationalError:
            if attempt == 29:
                raise
            time.sleep(2)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,))
        if cur.fetchone():
            print(f"database '{DATABASE}' exists")
        else:
            cur.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(DATABASE), sql.Identifier(params["user"]))
            )
            print(f"created database '{DATABASE}'")
    conn.close()


if __name__ == "__main__":
    main()
