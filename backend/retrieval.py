from __future__ import annotations
from dataclasses import dataclass

import psycopg2.extensions
from pgvector.psycopg2 import register_vector

from backend.ai_client import embed, rerank  # noqa: F401 — imported for test patchability
from backend.tracing import retrieval_span, rerank_span


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
        # IVFFlat default probes=1 is too low for small corpora; 10 gives good recall.
        cur.execute("SET LOCAL ivfflat.probes = 10")
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


def rrf_fuse(
    dense: list[ChunkResult],
    sparse: list[ChunkResult],
    *,
    rrf_k: int = 60,
) -> list[ChunkResult]:
    """Merges dense and sparse search results using Reciprocal Rank Fusion.

    Args:
        dense: List of chunks from dense search, in rank order
        sparse: List of chunks from sparse search, in rank order
        rrf_k: RRF constant (denominator base), default 60

    Returns:
        Merged list of chunks sorted by RRF score (descending).
        Chunks appearing in both lists are deduplicated and scored as the sum
        of their individual RRF contributions.
        Each returned chunk has its `score` field set to its RRF score.
    """
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


def expand_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[ChunkResult],
) -> list[ExpandedChunk]:
    """Fetch prev/next neighbor content for each chunk in a single query."""
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
    query_embedding: list[float] | None = None,
    retrieval_mode: str = "hybrid",  # "dense_only" | "hybrid" | "hybrid_no_rerank"
) -> RetrievalResult:
    """Full retrieval pipeline: embed → dense+sparse → RRF → rerank → expand."""
    with retrieval_span(query, tenant, retrieval_mode) as span:
        # 1. Embed query (skip if pre-computed embedding is provided)
        if query_embedding is None:
            query_embedding = embed([query])[0]

        # 2. Dense search (always)
        dense_results = dense_search(conn, query_embedding, tenant, k=dense_k)

        # 3. Sparse search + RRF fusion (skipped for dense_only)
        if retrieval_mode == "dense_only":
            fused = dense_results
        else:
            sparse_results = sparse_search(conn, query, tenant, k=sparse_k)
            fused = rrf_fuse(dense_results, sparse_results)

        if not fused:
            span.set_attribute("certcoach.chunk_count", 0)
            span.set_attribute("certcoach.confidence", 0.0)
            span.set_attribute("certcoach.below_threshold", True)
            return RetrievalResult(
                chunks=[], confidence=0.0, below_threshold=True,
                query=query, tenant=tenant,
            )

        # 4. hybrid_no_rerank: normalise RRF score to 0-1 against theoretical max.
        #    RRF max: top-1 in both lists → 2 * 1/(k+1) = 2/61 ≈ 0.0328.
        _RRF_MAX = 2.0 / (60 + 1)
        if retrieval_mode == "hybrid_no_rerank":
            top_chunks = fused[:rerank_top_n]
            confidence = min(top_chunks[0].score / _RRF_MAX, 1.0) if top_chunks else 0.0
            expanded = expand_chunks(conn, top_chunks)
            span.set_attribute("certcoach.chunk_count", len(expanded))
            span.set_attribute("certcoach.confidence", confidence)
            span.set_attribute("certcoach.below_threshold", False)
            return RetrievalResult(
                chunks=expanded, confidence=confidence,
                below_threshold=False, query=query, tenant=tenant,
            )

        # 5. Rerank — pass up to rerank_top_n * 4 candidates (capped at fused list length)
        candidates = fused[: rerank_top_n * 4]
        with rerank_span(query, n_candidates=len(candidates)):
            rerank_results = rerank(query, [c.content for c in candidates], top_n=rerank_top_n)
        rerank_results_sorted = sorted(rerank_results, key=lambda r: r.score, reverse=True)

        confidence = rerank_results_sorted[0].score if rerank_results_sorted else 0.0
        below_threshold = confidence < confidence_threshold

        # 6. Expand top-n chunks to include prev/next neighbors
        top_chunks = [candidates[r.index] for r in rerank_results_sorted[:rerank_top_n]]
        expanded = expand_chunks(conn, top_chunks)

        span.set_attribute("certcoach.chunk_count", len(expanded))
        span.set_attribute("certcoach.confidence", confidence)
        span.set_attribute("certcoach.below_threshold", below_threshold)

        return RetrievalResult(
            chunks=expanded,
            confidence=confidence,
            below_threshold=below_threshold,
            query=query,
            tenant=tenant,
        )
