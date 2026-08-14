from __future__ import annotations
import pytest


def _db_reachable() -> bool:
    try:
        from backend.db import get_conn
        conn = get_conn()
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(), reason="Postgres container not running"
)


@pytest.fixture
def conn():
    from backend.db import get_conn, apply_migrations
    from pathlib import Path
    c = get_conn()
    apply_migrations(c, Path("migrations"))
    yield c
    c.rollback()
    c.close()


def _make_doc():
    from ingestion.sources import DocSpec
    return DocSpec(
        id="test-source",
        title="Test Source",
        url="https://example.com",
        type="faq",
        format="html",
        citation="Test Citation",
        corpus=True,
        source_version=None,
    )


def _make_chunk_rows(n: int = 3):
    from ingestion.store import ChunkRow
    return [
        ChunkRow(
            content=f"chunk content {i} " + "word " * 20,
            chunk_index=i,
            token_count=25,
            embedding=[0.1] * 1536,
        )
        for i in range(n)
    ]


def test_store_inserts_rows(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(3)
    count = store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    assert count == 3
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 3


def test_store_is_idempotent(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(3)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 3


def test_neighbor_links_are_set(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(3)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_index, prev_chunk_id, next_chunk_id "
            "FROM chunks WHERE tenant='aws-saa' AND source_id='test-source' "
            "ORDER BY chunk_index"
        )
        rows = cur.fetchall()
    assert rows[0][1] is None        # first: no prev
    assert rows[0][2] is not None    # first: has next
    assert rows[1][1] is not None    # middle: has prev
    assert rows[1][2] is not None    # middle: has next
    assert rows[2][2] is None        # last: no next


def test_tenant_isolation(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(2)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    store_chunks(conn, tenant="gcp-ace", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='gcp-ace' AND source_id='test-source'")
        assert cur.fetchone()[0] == 2


def test_empty_chunks_returns_zero(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    # Ensure clean slate so the post-call count assertion is unambiguous
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
    count = store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=[])
    assert count == 0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 0


def test_cleanup(conn):
    # Clean up test rows so they don't pollute other tests
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id='test-source'")
    conn.commit()
