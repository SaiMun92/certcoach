from __future__ import annotations

from pathlib import Path

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv
import os

load_dotenv()


def get_conn() -> psycopg2.extensions.connection:
    url = os.environ["DATABASE_URL"]
    return psycopg2.connect(url)


def apply_migrations(
    conn: psycopg2.extensions.connection,
    migrations_dir: Path = Path("migrations"),
) -> None:
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
