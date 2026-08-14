from __future__ import annotations
import pytest
import psycopg2.extras


def _db_reachable() -> bool:
    try:
        from backend.db import get_conn
        c = get_conn(); c.close(); return True
    except Exception:
        return False


# Use a dedicated test tenant so test rows never compete with real corpus data.
_TEST_TENANT = "_test_"

# Marker for DB-dependent tests only (pure unit tests don't use this)
_db_required = pytest.mark.skipif(
    not _db_reachable(), reason="Postgres container not available"
)


def _insert_chunks(conn, rows):
    """Insert raw chunk rows for tests. rows: list of dicts with keys matching INSERT."""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO chunks
                (tenant, source_id, citation, doc_type, chunk_index, content, token_count, embedding)
            VALUES %s
        """, [
            (r["tenant"], r["source_id"], r["citation"], r["doc_type"],
             r["chunk_index"], r["content"], r["token_count"], r["embedding"])
            for r in rows
        ])
    conn.commit()


def _cleanup(conn, source_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
    conn.commit()


RIGHT = [1.0] + [0.0] * 1535
UP    = [0.0, 1.0] + [0.0] * 1534


@_db_required
def test_dense_search_ranks_similar_chunk_first():
    from backend.db import get_conn
    from backend.retrieval import dense_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": _TEST_TENANT, "source_id": "test-dense", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "about compute", "token_count": 2, "embedding": RIGHT},
            {"tenant": _TEST_TENANT, "source_id": "test-dense", "citation": "T", "doc_type": "faq",
             "chunk_index": 1, "content": "about storage", "token_count": 2, "embedding": UP},
        ])
        results = dense_search(conn, RIGHT, _TEST_TENANT, k=5)
        assert results[0].content == "about compute"
        assert results[0].score > results[1].score
    finally:
        _cleanup(conn, "test-dense")
        conn.close()


@_db_required
def test_dense_search_tenant_isolation():
    from backend.db import get_conn
    from backend.retrieval import dense_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "_test_other_", "source_id": "test-dense-iso", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "gcp only", "token_count": 2, "embedding": RIGHT},
        ])
        results = dense_search(conn, RIGHT, _TEST_TENANT, k=5)
        contents = [r.content for r in results]
        assert "gcp only" not in contents
    finally:
        _cleanup(conn, "test-dense-iso")
        conn.close()


@_db_required
def test_sparse_search_finds_keyword_match():
    from backend.db import get_conn
    from backend.retrieval import sparse_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": _TEST_TENANT, "source_id": "test-sparse", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "EC2 instance types for memory", "token_count": 5,
             "embedding": RIGHT},
            {"tenant": _TEST_TENANT, "source_id": "test-sparse", "citation": "T", "doc_type": "faq",
             "chunk_index": 1, "content": "S3 bucket policies", "token_count": 3,
             "embedding": UP},
        ])
        results = sparse_search(conn, "EC2 instance", _TEST_TENANT, k=5)
        assert any(r.content == "EC2 instance types for memory" for r in results)
        # The EC2 chunk should rank above the S3 chunk
        ec2_idx = next(i for i, r in enumerate(results) if "EC2" in r.content)
        s3_results = [i for i, r in enumerate(results) if "S3" in r.content]
        if s3_results:
            assert ec2_idx < s3_results[0]
    finally:
        _cleanup(conn, "test-sparse")
        conn.close()


@_db_required
def test_sparse_search_no_match_returns_empty():
    from backend.db import get_conn
    from backend.retrieval import sparse_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": _TEST_TENANT, "source_id": "test-sparse-empty", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "hello world", "token_count": 2, "embedding": RIGHT},
        ])
        results = sparse_search(conn, "xyzzy quux frobnitz", _TEST_TENANT, k=5)
        assert results == []
    finally:
        _cleanup(conn, "test-sparse-empty")
        conn.close()


# RRF fusion tests (pure unit tests, no DB)

def _make_chunk(id: int, content: str, score: float = 0.5) -> "ChunkResult":
    from backend.retrieval import ChunkResult
    return ChunkResult(
        id=id, tenant="aws-saa", source_id="src", chunk_index=id,
        content=content, token_count=5, citation="T", doc_type="faq",
        source_date=None, prev_chunk_id=None, next_chunk_id=None, score=score,
    )


def test_rrf_shared_chunk_scores_higher_than_single_list():
    from backend.retrieval import rrf_fuse
    shared = _make_chunk(1, "shared")
    dense_only = _make_chunk(2, "dense only")
    sparse_only = _make_chunk(3, "sparse only")

    result = rrf_fuse([shared, dense_only], [shared, sparse_only])
    ids = [r.id for r in result]
    # shared appears in both lists → higher RRF score → ranks first
    assert ids[0] == 1


def test_rrf_deduplicates_by_id():
    from backend.retrieval import rrf_fuse
    chunk = _make_chunk(1, "same chunk")
    result = rrf_fuse([chunk], [chunk])
    assert len(result) == 1
    expected = 2.0 / (60 + 1)
    assert abs(result[0].score - expected) < 1e-9


def test_rrf_score_is_set_on_output():
    from backend.retrieval import rrf_fuse
    c1, c2 = _make_chunk(1, "a"), _make_chunk(2, "b")
    result = rrf_fuse([c1], [c2])
    expected = 1.0 / (60 + 1)
    for r in result:
        assert abs(r.score - expected) < 1e-9


def test_rrf_empty_lists_returns_empty():
    from backend.retrieval import rrf_fuse
    assert rrf_fuse([], []) == []


def test_rrf_single_list_still_ranked():
    from backend.retrieval import rrf_fuse
    c1, c2 = _make_chunk(1, "first"), _make_chunk(2, "second")
    result = rrf_fuse([c1, c2], [])
    assert [r.id for r in result] == [1, 2]
    assert result[0].score > result[1].score


@_db_required
def test_retrieve_returns_result_with_stub_rerank():
    """Full pipeline with stubbed embed + rerank — verifies orchestration, not AI quality."""
    from unittest.mock import patch
    from backend.db import get_conn
    from backend.retrieval import retrieve
    from backend.ai_client import RerankResult

    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "aws-saa", "source_id": "test-retrieve", "citation": "AWS EC2 FAQs",
             "doc_type": "faq", "chunk_index": 0,
             "content": "EC2 instance types provide different compute capacities",
             "token_count": 7, "embedding": RIGHT},
            {"tenant": "aws-saa", "source_id": "test-retrieve", "citation": "AWS EC2 FAQs",
             "doc_type": "faq", "chunk_index": 1,
             "content": "Memory optimised instances are suited for in-memory databases",
             "token_count": 8, "embedding": UP},
        ])

        stub_embed = lambda texts, **kw: [RIGHT]
        stub_rerank = lambda query, docs, **kw: [
            RerankResult(index=0, score=0.85),
            RerankResult(index=1, score=0.4),
        ]

        with patch("backend.retrieval.embed", side_effect=stub_embed), \
             patch("backend.retrieval.rerank", side_effect=stub_rerank):
            result = retrieve(conn, "EC2 compute", "aws-saa")

        assert not result.below_threshold
        assert result.confidence == pytest.approx(0.85)
        assert len(result.chunks) > 0
        assert result.query == "EC2 compute"
        assert result.tenant == "aws-saa"
    finally:
        _cleanup(conn, "test-retrieve")
        conn.close()


@_db_required
def test_retrieve_below_threshold_when_low_score():
    from unittest.mock import patch
    from backend.db import get_conn
    from backend.retrieval import retrieve
    from backend.ai_client import RerankResult

    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "aws-saa", "source_id": "test-retrieve-low", "citation": "T",
             "doc_type": "faq", "chunk_index": 0, "content": "unrelated content here",
             "token_count": 3, "embedding": RIGHT},
        ])

        stub_embed = lambda texts, **kw: [RIGHT]
        stub_rerank = lambda query, docs, **kw: [RerankResult(index=0, score=0.05)]

        with patch("backend.retrieval.embed", side_effect=stub_embed), \
             patch("backend.retrieval.rerank", side_effect=stub_rerank):
            result = retrieve(conn, "obscure query", "aws-saa", confidence_threshold=0.1)

        assert result.below_threshold
        assert result.confidence == pytest.approx(0.05)
    finally:
        _cleanup(conn, "test-retrieve-low")
        conn.close()


@_db_required
def test_retrieve_empty_corpus_returns_below_threshold():
    from unittest.mock import patch
    from backend.db import get_conn
    from backend.retrieval import retrieve

    conn = get_conn()
    try:
        stub_embed = lambda texts, **kw: [RIGHT]
        with patch("backend.retrieval.embed", side_effect=stub_embed):
            result = retrieve(conn, "anything", "aws-saa",
                              dense_k=0, sparse_k=0)
        assert result.below_threshold
        assert result.confidence == 0.0
        assert result.chunks == []
    finally:
        conn.close()


@_db_required
def test_expand_chunks_fetches_neighbors():
    from backend.db import get_conn
    from backend.retrieval import expand_chunks, ChunkResult
    from ingestion.store import ChunkRow, store_chunks
    from ingestion.sources import DocSpec

    doc = DocSpec(id="test-expand", title="T", url="https://x.com",
                  type="faq", format="html", citation="T", corpus=True)
    rows = [
        ChunkRow(content=f"chunk {i}", chunk_index=i, token_count=2,
                 embedding=[0.0] * 1536)
        for i in range(3)
    ]
    conn = get_conn()
    try:
        store_chunks(conn, tenant="aws-saa", doc=doc,
                     source_date="2026-08-08", chunks=rows)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, prev_chunk_id, next_chunk_id FROM chunks "
                "WHERE source_id='test-expand' ORDER BY chunk_index"
            )
            db_rows = cur.fetchall()

        # Middle chunk (index 1) has both prev and next
        mid_id, mid_prev, mid_next = db_rows[1]
        mid_chunk = ChunkResult(
            id=mid_id, tenant="aws-saa", source_id="test-expand",
            chunk_index=1, content="chunk 1", token_count=2,
            citation="T", doc_type="faq", source_date=None,
            prev_chunk_id=mid_prev, next_chunk_id=mid_next, score=1.0,
        )

        expanded = expand_chunks(conn, [mid_chunk])
        assert len(expanded) == 1
        assert expanded[0].prev_content == "chunk 0"
        assert expanded[0].next_content == "chunk 2"
    finally:
        _cleanup(conn, "test-expand")
        conn.close()


def test_retrieve_dense_only_skips_sparse():
    """dense_only mode calls dense_search but not sparse_search."""
    import backend.retrieval as ret
    from backend.retrieval import ChunkResult, ExpandedChunk
    from backend.ai_client import RerankResult
    from unittest.mock import MagicMock, patch

    chunk = ChunkResult(
        id=1, tenant="aws-saa", source_id="faq-ec2", chunk_index=0,
        content="text", token_count=5, citation="cite", doc_type="faq",
        source_date=None, prev_chunk_id=None, next_chunk_id=None, score=0.9,
    )

    with patch.object(ret, "embed", return_value=[[0.0] * 1536]), \
         patch.object(ret, "dense_search", return_value=[chunk]) as mock_dense, \
         patch.object(ret, "sparse_search") as mock_sparse, \
         patch.object(ret, "rerank", return_value=[RerankResult(index=0, score=0.9)]), \
         patch.object(ret, "expand_chunks", return_value=[ExpandedChunk(chunk, None, None)]):
        conn = MagicMock()
        result = ret.retrieve(conn, "q", "aws-saa", retrieval_mode="dense_only")

    mock_dense.assert_called_once()
    mock_sparse.assert_not_called()
    assert not result.below_threshold
    assert result.confidence == pytest.approx(0.9)


def test_retrieve_hybrid_no_rerank_skips_rerank():
    """hybrid_no_rerank mode fuses dense+sparse but skips rerank."""
    import backend.retrieval as ret
    from backend.retrieval import ChunkResult, ExpandedChunk
    from unittest.mock import MagicMock, patch

    chunk = ChunkResult(
        id=2, tenant="aws-saa", source_id="faq-s3", chunk_index=0,
        content="s3 text", token_count=4, citation="s3", doc_type="faq",
        source_date=None, prev_chunk_id=None, next_chunk_id=None, score=0.016,
    )

    with patch.object(ret, "embed", return_value=[[0.0] * 1536]), \
         patch.object(ret, "dense_search", return_value=[chunk]), \
         patch.object(ret, "sparse_search", return_value=[chunk]), \
         patch.object(ret, "rerank") as mock_rerank, \
         patch.object(ret, "expand_chunks", return_value=[ExpandedChunk(chunk, None, None)]):
        conn = MagicMock()
        result = ret.retrieve(conn, "s3", "aws-saa", retrieval_mode="hybrid_no_rerank")

    mock_rerank.assert_not_called()
    assert len(result.chunks) == 1
