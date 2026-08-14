import os
import pytest
import psycopg2
from pathlib import Path

from backend.db import apply_migrations, get_conn

# Skip all tests in this file if the DB is unreachable (e.g. CI without Docker).
def _db_reachable() -> bool:
    try:
        conn = get_conn()
        conn.close()
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason="certcoach-db container not running",
)


@pytest.fixture
def conn():
    c = get_conn()
    yield c
    c.close()


def test_get_conn_returns_open_connection(conn):
    assert conn.closed == 0


def test_apply_migrations_creates_chunks_table(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'chunks' AND table_schema = 'public'"
        )
        assert cur.fetchone()[0] == 1


def test_chunks_table_has_required_columns(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND table_schema = 'public'"
        )
        cols = {row[0] for row in cur.fetchall()}
    required = {
        "id", "tenant", "source_id", "source_version", "source_date",
        "citation", "doc_type", "parent_doc_id", "prev_chunk_id",
        "next_chunk_id", "chunk_index", "content", "token_count",
        "embedding", "fts", "created_at",
    }
    assert required <= cols, f"missing columns: {required - cols}"


def test_embedding_column_is_vector_1536(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "vector"
    # Verify dimension via pg_attribute.
    # For pgvector, atttypmod stores the dimension directly (e.g. 1536 for vector(1536)).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "JOIN pg_class ON pg_class.oid = pg_attribute.attrelid "
            "WHERE pg_class.relname = 'chunks' AND pg_attribute.attname = 'embedding'"
        )
        dim = cur.fetchone()[0]
    assert dim == 1536, f"expected vector(1536), got vector({dim})"


def test_apply_migrations_is_idempotent(conn):
    apply_migrations(conn, Path("migrations"))
    # Running a second time must not raise
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = '001'")
        assert cur.fetchone()[0] == 1


def test_insert_and_retrieve_chunk(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks
                (tenant, source_id, citation, doc_type, chunk_index, content)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            ("aws-saa", "faq-ec2", "Amazon EC2 FAQs", "faq", 0, "EC2 is a virtual machine service."),
        )
        inserted_id = cur.fetchone()[0]
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT tenant, content FROM chunks WHERE id = %s", (inserted_id,))
        row = cur.fetchone()
    assert row == ("aws-saa", "EC2 is a virtual machine service.")
    # Clean up
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE id = %s", (inserted_id,))
    conn.commit()


def test_fts_generated_column_is_populated(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks
                (tenant, source_id, citation, doc_type, chunk_index, content)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            ("aws-saa", "faq-s3", "Amazon S3 FAQs", "faq", 0, "S3 stores objects in buckets."),
        )
        inserted_id = cur.fetchone()[0]
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fts IS NOT NULL FROM chunks WHERE id = %s", (inserted_id,)
        )
        assert cur.fetchone()[0] is True
    # Clean up
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE id = %s", (inserted_id,))
    conn.commit()


def test_tenant_isolation_index_is_used(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname LIKE '%tenant%'"
        )
        indexes = {row[0] for row in cur.fetchall()}
    assert "chunks_tenant_idx" in indexes
    assert "chunks_tenant_source_idx" in indexes
