# Retrieval Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement hybrid retrieval `(query, tenant) → expanded chunks` — dense pgvector search + sparse BM25 search → RRF fusion → Cohere rerank → confidence gate → expand-to-context.

**Architecture:** A single `certcoach/retrieval.py` module exposes a `retrieve()` function that orchestrates five stages: embed the query, run dense + sparse search in Postgres, fuse with Reciprocal Rank Fusion, rerank top candidates with cohere-rerank-pro (via ai-core gateway), apply a confidence gate, and expand each top chunk to include its prev/next neighbors. The Cohere rerank call is added to `certcoach/ai_client.py` alongside `embed()`.

**Tech Stack:** Python ≥ 3.11, psycopg2, pgvector Python adapter (`pgvector.psycopg2.register_vector`), httpx (already a dep), pytest with live-DB integration tests.

## Global Constraints

- Python ≥ 3.11.
- **No mocks for DB tests** — integration tests hit the real running Postgres container.
- No Azure. No exam-dump sources.
- Tenant values: exactly `"aws-saa"` and `"gcp-ace"`. Every query filters on `tenant` first.
- Embedding model: `text-embedding-3-large`, `dimensions=1536`. Dense search uses cosine distance (`<=>` operator).
- Sparse search uses `plainto_tsquery('english', ...)` against the `fts tsvector` generated column.
- RRF constant `k=60` (standard default). Formula: `score(d) = Σ 1/(k + rank_L(d))` over both lists.
- Rerank model: `"cohere-rerank-pro"`. Called via POST `{AICORE_BASE_URL}/rerank` with `Authorization: Bearer {AICORE_API_KEY}`. Request body: `{"model": ..., "query": ..., "documents": [...], "top_n": ...}`. Response: `{"results": [{"index": int, "relevance_score": float}, ...]}`.
- Confidence gate threshold default: `0.1`. If top reranker score < threshold, `RetrievalResult.below_threshold = True`.
- `embed` and `rerank` must be imported at **module level** in `certcoach/retrieval.py` (for test patchability via `patch("certcoach.retrieval.embed")` and `patch("certcoach.retrieval.rerank")`).
- All model IDs are ai-core gateway aliases — use exactly as listed.
- `pyproject.toml` deps: httpx already present; `pgvector>=0.3` already present. No new deps needed.

---

## File map

| File | Create / Modify | Responsibility |
|---|---|---|
| `certcoach/ai_client.py` | Modify | Add `RerankResult` dataclass + `rerank(query, documents, *, model, top_n) → list[RerankResult]` |
| `certcoach/retrieval.py` | Create | `ChunkResult`, `ExpandedChunk`, `RetrievalResult`; `dense_search`, `sparse_search`, `rrf_fuse`, `expand_chunks`, `retrieve` |
| `tests/test_ai_client.py` | Modify | Add 3 tests for `rerank()` |
| `tests/test_retrieval.py` | Create | Integration tests for dense/sparse/retrieve; unit tests for rrf_fuse |

---

### Task 1: `rerank()` in ai_client

**Files:**
- Modify: `certcoach/ai_client.py`
- Modify: `tests/test_ai_client.py`

**Interfaces:**
- Consumes: `AICORE_BASE_URL`, `AICORE_API_KEY` env vars (already used by `embed`)
- Produces:
  - `RerankResult` dataclass: `index: int`, `score: float`
  - `rerank(query: str, documents: list[str], *, model: str = "cohere-rerank-pro", top_n: int = 5) -> list[RerankResult]` — returns results ordered by `score` descending (as returned by the API). Empty `documents` returns `[]` without calling the API.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ai_client.py`:

```python
def test_rerank_returns_results():
    from unittest.mock import patch, MagicMock
    from certcoach.ai_client import rerank, RerankResult

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response) as mock_post:
        results = rerank("what is EC2?", ["about storage", "about compute"])

    assert len(results) == 2
    assert results[0] == RerankResult(index=1, score=0.9)
    assert results[1] == RerankResult(index=0, score=0.3)
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["json"]["query"] == "what is EC2?"
    assert call_kwargs.kwargs["json"]["documents"] == ["about storage", "about compute"]
    assert call_kwargs.kwargs["json"]["model"] == "cohere-rerank-pro"
    assert call_kwargs.kwargs["json"]["top_n"] == 5


def test_rerank_empty_documents_returns_empty():
    from certcoach.ai_client import rerank
    from unittest.mock import patch
    with patch("httpx.post") as mock_post:
        result = rerank("query", [])
    assert result == []
    mock_post.assert_not_called()


def test_rerank_uses_correct_url_and_auth(monkeypatch):
    from unittest.mock import patch, MagicMock
    from certcoach.ai_client import rerank

    monkeypatch.setenv("AICORE_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("AICORE_API_KEY", "test-key-xyz")

    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.7}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response) as mock_post:
        rerank("q", ["doc1"], top_n=1)

    call_kwargs = mock_post.call_args
    assert call_kwargs.args[0] == "https://gateway.example.com/v1/rerank"
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-key-xyz"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ai_client.py -v -k "rerank"
```
Expected: ImportError (`RerankResult` not defined yet).

- [ ] **Step 3: Implement in `certcoach/ai_client.py`**

Add after the existing `embed` function:

```python
from dataclasses import dataclass

@dataclass
class RerankResult:
    index: int
    score: float


def rerank(
    query: str,
    documents: list[str],
    *,
    model: str = "cohere-rerank-pro",
    top_n: int = 5,
) -> list[RerankResult]:
    if not documents:
        return []
    import httpx
    base_url = os.environ["AICORE_BASE_URL"].rstrip("/")
    api_key = os.environ["AICORE_API_KEY"]
    response = httpx.post(
        f"{base_url}/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "query": query, "documents": documents, "top_n": top_n},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return [RerankResult(index=r["index"], score=r["relevance_score"]) for r in data["results"]]
```

Note: `from dataclasses import dataclass` must be added at the top of the file.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ai_client.py -v
```
Expected: all 6 pass (3 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add certcoach/ai_client.py tests/test_ai_client.py
git commit -m "feat: rerank() wrapper for cohere-rerank-pro via ai-core"
```

---

### Task 2: Dense search + sparse search

**Files:**
- Create: `certcoach/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `certcoach.db.get_conn` (psycopg2 connection); `pgvector.psycopg2.register_vector`
- Produces:
  - `ChunkResult` dataclass: `id: int`, `tenant: str`, `source_id: str`, `chunk_index: int`, `content: str`, `token_count: int`, `citation: str`, `doc_type: str`, `source_date: str | None`, `prev_chunk_id: int | None`, `next_chunk_id: int | None`, `score: float`
  - `dense_search(conn, embedding: list[float], tenant: str, k: int = 20) -> list[ChunkResult]` — cosine similarity via `<=>`, filtered by `tenant`, ordered by ascending distance (= descending similarity), `score = 1 - cosine_distance`
  - `sparse_search(conn, query_text: str, tenant: str, k: int = 20) -> list[ChunkResult]` — `fts @@ plainto_tsquery('english', query_text)`, filtered by `tenant`, `score = ts_rank_cd(fts, query)`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_retrieval.py`:

```python
from __future__ import annotations
import pytest
import psycopg2.extras


def _db_reachable() -> bool:
    try:
        from certcoach.db import get_conn
        c = get_conn(); c.close(); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
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


def test_dense_search_ranks_similar_chunk_first():
    from certcoach.db import get_conn
    from certcoach.retrieval import dense_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "aws-saa", "source_id": "test-dense", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "about compute", "token_count": 2, "embedding": RIGHT},
            {"tenant": "aws-saa", "source_id": "test-dense", "citation": "T", "doc_type": "faq",
             "chunk_index": 1, "content": "about storage", "token_count": 2, "embedding": UP},
        ])
        results = dense_search(conn, RIGHT, "aws-saa", k=5)
        assert results[0].content == "about compute"
        assert results[0].score > results[1].score
    finally:
        _cleanup(conn, "test-dense")
        conn.close()


def test_dense_search_tenant_isolation():
    from certcoach.db import get_conn
    from certcoach.retrieval import dense_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "gcp-ace", "source_id": "test-dense-iso", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "gcp only", "token_count": 2, "embedding": RIGHT},
        ])
        results = dense_search(conn, RIGHT, "aws-saa", k=5)
        contents = [r.content for r in results]
        assert "gcp only" not in contents
    finally:
        _cleanup(conn, "test-dense-iso")
        conn.close()


def test_sparse_search_finds_keyword_match():
    from certcoach.db import get_conn
    from certcoach.retrieval import sparse_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "aws-saa", "source_id": "test-sparse", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "EC2 instance types for memory", "token_count": 5,
             "embedding": RIGHT},
            {"tenant": "aws-saa", "source_id": "test-sparse", "citation": "T", "doc_type": "faq",
             "chunk_index": 1, "content": "S3 bucket policies", "token_count": 3,
             "embedding": UP},
        ])
        results = sparse_search(conn, "EC2 instance", "aws-saa", k=5)
        assert any(r.content == "EC2 instance types for memory" for r in results)
        # The EC2 chunk should rank above the S3 chunk
        ec2_idx = next(i for i, r in enumerate(results) if "EC2" in r.content)
        s3_results = [i for i, r in enumerate(results) if "S3" in r.content]
        if s3_results:
            assert ec2_idx < s3_results[0]
    finally:
        _cleanup(conn, "test-sparse")
        conn.close()


def test_sparse_search_no_match_returns_empty():
    from certcoach.db import get_conn
    from certcoach.retrieval import sparse_search
    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "aws-saa", "source_id": "test-sparse-empty", "citation": "T", "doc_type": "faq",
             "chunk_index": 0, "content": "hello world", "token_count": 2, "embedding": RIGHT},
        ])
        results = sparse_search(conn, "xyzzy quux frobnitz", "aws-saa", k=5)
        assert results == []
    finally:
        _cleanup(conn, "test-sparse-empty")
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v -k "dense or sparse"
```
Expected: ImportError (`certcoach.retrieval` not found).

- [ ] **Step 3: Implement `certcoach/retrieval.py`**

```python
from __future__ import annotations
from dataclasses import dataclass

import psycopg2.extensions
from pgvector.psycopg2 import register_vector

from certcoach.ai_client import embed, rerank


@dataclass
class ChunkResult:
    id: int
    tenant: str
    source_id: str
    chunk_index: int
    content: str
    token_count: int
    citation: str
    doc_type: str
    source_date: str | None
    prev_chunk_id: int | None
    next_chunk_id: int | None
    score: float


@dataclass
class ExpandedChunk:
    chunk: ChunkResult
    prev_content: str | None
    next_content: str | None


@dataclass
class RetrievalResult:
    chunks: list[ExpandedChunk]
    confidence: float
    below_threshold: bool
    query: str
    tenant: str


_CHUNK_COLS = """
    id, tenant, source_id, chunk_index, content, token_count,
    citation, doc_type, source_date::text,
    prev_chunk_id, next_chunk_id
"""


def _row_to_chunk(row, score: float) -> ChunkResult:
    return ChunkResult(
        id=row[0], tenant=row[1], source_id=row[2], chunk_index=row[3],
        content=row[4], token_count=row[5], citation=row[6], doc_type=row[7],
        source_date=row[8], prev_chunk_id=row[9], next_chunk_id=row[10],
        score=score,
    )


def dense_search(
    conn: psycopg2.extensions.connection,
    embedding: list[float],
    tenant: str,
    k: int = 20,
) -> list[ChunkResult]:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_CHUNK_COLS},
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE tenant = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, tenant, embedding, k),
        )
        return [_row_to_chunk(row, row[11]) for row in cur.fetchall()]


def sparse_search(
    conn: psycopg2.extensions.connection,
    query_text: str,
    tenant: str,
    k: int = 20,
) -> list[ChunkResult]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_CHUNK_COLS},
                   ts_rank_cd(fts, plainto_tsquery('english', %s)) AS score
            FROM chunks
            WHERE tenant = %s
              AND fts @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_text, tenant, query_text, k),
        )
        return [_row_to_chunk(row, row[11]) for row in cur.fetchall()]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v -k "dense or sparse"
```
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add certcoach/retrieval.py tests/test_retrieval.py
git commit -m "feat: dense and sparse search in retrieval module"
```

---

### Task 3: RRF fusion

**Files:**
- Modify: `certcoach/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `list[ChunkResult]` from dense and sparse search
- Produces:
  - `rrf_fuse(dense: list[ChunkResult], sparse: list[ChunkResult], *, rrf_k: int = 60) -> list[ChunkResult]` — merges two ranked lists using Reciprocal Rank Fusion. Deduplicates by chunk `id`. Returns list ordered by RRF score descending. Each returned `ChunkResult` has `score` set to its RRF score.

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/test_retrieval.py` (no DB needed — pure unit tests, no skipif needed):

```python
def _make_chunk(id: int, content: str, score: float = 0.5) -> "ChunkResult":
    from certcoach.retrieval import ChunkResult
    return ChunkResult(
        id=id, tenant="aws-saa", source_id="src", chunk_index=id,
        content=content, token_count=5, citation="T", doc_type="faq",
        source_date=None, prev_chunk_id=None, next_chunk_id=None, score=score,
    )


def test_rrf_shared_chunk_scores_higher_than_single_list():
    from certcoach.retrieval import rrf_fuse
    shared = _make_chunk(1, "shared")
    dense_only = _make_chunk(2, "dense only")
    sparse_only = _make_chunk(3, "sparse only")

    result = rrf_fuse([shared, dense_only], [shared, sparse_only])
    ids = [r.id for r in result]
    # shared appears in both lists → higher RRF score → ranks first
    assert ids[0] == 1


def test_rrf_deduplicates_by_id():
    from certcoach.retrieval import rrf_fuse
    chunk = _make_chunk(1, "same chunk")
    result = rrf_fuse([chunk], [chunk])
    assert len(result) == 1


def test_rrf_score_is_set_on_output():
    from certcoach.retrieval import rrf_fuse
    c1, c2 = _make_chunk(1, "a"), _make_chunk(2, "b")
    result = rrf_fuse([c1], [c2])
    for r in result:
        assert r.score > 0.0


def test_rrf_empty_lists_returns_empty():
    from certcoach.retrieval import rrf_fuse
    assert rrf_fuse([], []) == []


def test_rrf_single_list_still_ranked():
    from certcoach.retrieval import rrf_fuse
    c1, c2 = _make_chunk(1, "first"), _make_chunk(2, "second")
    result = rrf_fuse([c1, c2], [])
    assert [r.id for r in result] == [1, 2]
    assert result[0].score > result[1].score
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v -k "rrf"
```
Expected: ImportError (`rrf_fuse` not defined yet).

- [ ] **Step 3: Implement `rrf_fuse` in `certcoach/retrieval.py`**

Add after `sparse_search`:

```python
def rrf_fuse(
    dense: list[ChunkResult],
    sparse: list[ChunkResult],
    *,
    rrf_k: int = 60,
) -> list[ChunkResult]:
    scores: dict[int, float] = {}
    chunks_by_id: dict[int, ChunkResult] = {}

    for rank, chunk in enumerate(dense, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[chunk.id] = chunk

    for rank, chunk in enumerate(sparse, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[chunk.id] = chunk

    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [
        ChunkResult(
            id=chunks_by_id[cid].id,
            tenant=chunks_by_id[cid].tenant,
            source_id=chunks_by_id[cid].source_id,
            chunk_index=chunks_by_id[cid].chunk_index,
            content=chunks_by_id[cid].content,
            token_count=chunks_by_id[cid].token_count,
            citation=chunks_by_id[cid].citation,
            doc_type=chunks_by_id[cid].doc_type,
            source_date=chunks_by_id[cid].source_date,
            prev_chunk_id=chunks_by_id[cid].prev_chunk_id,
            next_chunk_id=chunks_by_id[cid].next_chunk_id,
            score=scores[cid],
        )
        for cid in sorted_ids
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v -k "rrf"
```
Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add certcoach/retrieval.py tests/test_retrieval.py
git commit -m "feat: RRF fusion for hybrid retrieval"
```

---

### Task 4: `retrieve()` orchestrator — rerank + confidence gate + expand-to-context

**Files:**
- Modify: `certcoach/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes:
  - `certcoach.ai_client.embed` (module-level import — already present)
  - `certcoach.ai_client.rerank` (module-level import — already present)
  - `dense_search`, `sparse_search`, `rrf_fuse` from this module
- Produces:
  - `expand_chunks(conn, chunks: list[ChunkResult]) -> list[ExpandedChunk]` — fetches `prev_chunk_id` and `next_chunk_id` content in a single `WHERE id = ANY(...)` query. `prev_content`/`next_content` are `None` if the neighbor id is `None` or not found.
  - `retrieve(conn, query: str, tenant: str, *, dense_k: int = 20, sparse_k: int = 20, rerank_top_n: int = 5, confidence_threshold: float = 0.1) -> RetrievalResult` — full pipeline. Passes `rerank_top_n * 4` RRF candidates to reranker (but no more than the total fused list). Returns `RetrievalResult` with `below_threshold=True` when `confidence < confidence_threshold` or when no results at all (`confidence=0.0`).

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_retrieval.py`:

```python
def test_retrieve_returns_result_with_stub_rerank():
    """Full pipeline with stubbed embed + rerank — verifies orchestration, not AI quality."""
    from unittest.mock import patch
    from certcoach.db import get_conn
    from certcoach.retrieval import retrieve
    from certcoach.ai_client import RerankResult

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

        with patch("certcoach.retrieval.embed", side_effect=stub_embed), \
             patch("certcoach.retrieval.rerank", side_effect=stub_rerank):
            result = retrieve(conn, "EC2 compute", "aws-saa")

        assert not result.below_threshold
        assert result.confidence == pytest.approx(0.85)
        assert len(result.chunks) > 0
        assert result.query == "EC2 compute"
        assert result.tenant == "aws-saa"
    finally:
        _cleanup(conn, "test-retrieve")
        conn.close()


def test_retrieve_below_threshold_when_low_score():
    from unittest.mock import patch
    from certcoach.db import get_conn
    from certcoach.retrieval import retrieve
    from certcoach.ai_client import RerankResult

    conn = get_conn()
    try:
        _insert_chunks(conn, [
            {"tenant": "aws-saa", "source_id": "test-retrieve-low", "citation": "T",
             "doc_type": "faq", "chunk_index": 0, "content": "unrelated content here",
             "token_count": 3, "embedding": RIGHT},
        ])

        stub_embed = lambda texts, **kw: [RIGHT]
        stub_rerank = lambda query, docs, **kw: [RerankResult(index=0, score=0.05)]

        with patch("certcoach.retrieval.embed", side_effect=stub_embed), \
             patch("certcoach.retrieval.rerank", side_effect=stub_rerank):
            result = retrieve(conn, "obscure query", "aws-saa", confidence_threshold=0.1)

        assert result.below_threshold
        assert result.confidence == pytest.approx(0.05)
    finally:
        _cleanup(conn, "test-retrieve-low")
        conn.close()


def test_retrieve_empty_corpus_returns_below_threshold():
    from unittest.mock import patch
    from certcoach.db import get_conn
    from certcoach.retrieval import retrieve

    conn = get_conn()
    try:
        stub_embed = lambda texts, **kw: [RIGHT]
        with patch("certcoach.retrieval.embed", side_effect=stub_embed):
            result = retrieve(conn, "anything", "aws-saa",
                              dense_k=0, sparse_k=0)
        assert result.below_threshold
        assert result.confidence == 0.0
        assert result.chunks == []
    finally:
        conn.close()


def test_expand_chunks_fetches_neighbors():
    from certcoach.db import get_conn
    from certcoach.retrieval import expand_chunks, ChunkResult
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v -k "retrieve or expand"
```
Expected: ImportError (`retrieve`, `expand_chunks` not defined).

- [ ] **Step 3: Implement `expand_chunks` and `retrieve` in `certcoach/retrieval.py`**

Add after `rrf_fuse`:

```python
def expand_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[ChunkResult],
) -> list[ExpandedChunk]:
    ids_needed: set[int] = set()
    for c in chunks:
        if c.prev_chunk_id is not None:
            ids_needed.add(c.prev_chunk_id)
        if c.next_chunk_id is not None:
            ids_needed.add(c.next_chunk_id)

    neighbor_content: dict[int, str] = {}
    if ids_needed:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content FROM chunks WHERE id = ANY(%s)",
                (list(ids_needed),),
            )
            for row in cur.fetchall():
                neighbor_content[row[0]] = row[1]

    return [
        ExpandedChunk(
            chunk=c,
            prev_content=neighbor_content.get(c.prev_chunk_id) if c.prev_chunk_id else None,
            next_content=neighbor_content.get(c.next_chunk_id) if c.next_chunk_id else None,
        )
        for c in chunks
    ]


def retrieve(
    conn: psycopg2.extensions.connection,
    query: str,
    tenant: str,
    *,
    dense_k: int = 20,
    sparse_k: int = 20,
    rerank_top_n: int = 5,
    confidence_threshold: float = 0.1,
) -> RetrievalResult:
    # 1. Embed query
    query_embedding = embed([query])[0]

    # 2. Dense + sparse
    dense_results = dense_search(conn, query_embedding, tenant, k=dense_k)
    sparse_results = sparse_search(conn, query, tenant, k=sparse_k)

    # 3. RRF fusion
    fused = rrf_fuse(dense_results, sparse_results)

    if not fused:
        return RetrievalResult(
            chunks=[], confidence=0.0, below_threshold=True,
            query=query, tenant=tenant,
        )

    # 4. Rerank — pass up to rerank_top_n * 4 candidates
    candidates = fused[: rerank_top_n * 4]
    rerank_results = rerank(query, [c.content for c in candidates], top_n=rerank_top_n)
    rerank_results_sorted = sorted(rerank_results, key=lambda r: r.score, reverse=True)

    confidence = rerank_results_sorted[0].score if rerank_results_sorted else 0.0
    below_threshold = confidence < confidence_threshold

    # 5. Expand top-n to include prev/next neighbors
    top_chunks = [candidates[r.index] for r in rerank_results_sorted[:rerank_top_n]]
    expanded = expand_chunks(conn, top_chunks)

    return RetrievalResult(
        chunks=expanded,
        confidence=confidence,
        below_threshold=below_threshold,
        query=query,
        tenant=tenant,
    )
```

- [ ] **Step 4: Run full test suite to verify everything passes**

```bash
pytest -v
```
Expected: all tests pass or skip. No new failures.

- [ ] **Step 5: Commit**

```bash
git add certcoach/retrieval.py tests/test_retrieval.py
git commit -m "feat: retrieve() orchestrator — rerank, confidence gate, expand-to-context"
```
