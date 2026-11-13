"""Create a new isolated synthetic PostgreSQL database for graph integration checks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from local_app.config import Settings
from local_app.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-file", type=Path, required=True)
    args = parser.parse_args()
    if args.name_file.exists():
        raise FileExistsError(args.name_file)
    dbname = "nirs_personal_graph_it_" + uuid4().hex[:12]
    settings = Settings.from_env()
    conn = psycopg2.connect(settings.database_url, connect_timeout=10)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {} ").format(sql.Identifier(dbname)))
    finally:
        conn.close()
    args.name_file.parent.mkdir(parents=True, exist_ok=True)
    args.name_file.write_text(dbname + "\n", encoding="ascii")
    store = Store(make_dsn(settings.database_url, dbname=dbname))
    applied = store.migrate()
    print(f"Created isolated synthetic database {dbname}; migrations applied: {applied}; schema: {store.schema_version()}")


if __name__ == "__main__":
    main()
