from __future__ import annotations
from dataclasses import dataclass

import psycopg2.extras

from ingestion.sources import DocSpec


@dataclass
class ChunkRow:
    content: str
    chunk_index: int
    token_count: int
    embedding: list[float]


def store_chunks(
    conn,
    *,
    tenant: str,
    doc: DocSpec,
    source_date: str | None,
    chunks: list[ChunkRow],
) -> int:
    if not chunks:
        return 0
    with conn.cursor() as cur:
        # Idempotent: delete existing rows for this (tenant, source_id)
        cur.execute(
            "DELETE FROM chunks WHERE tenant = %s AND source_id = %s",
            (tenant, doc.id),
        )
        # Bulk insert — embedding as list (pgvector accepts Python list)
        # execute_values with fetch=True returns the RETURNING rows directly
        rows = psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks
                (tenant, source_id, source_version, source_date, citation,
                 doc_type, parent_doc_id, chunk_index, content, token_count, embedding)
            VALUES %s
            RETURNING id, chunk_index
            """,
            [
                (
                    tenant,
                    doc.id,
                    doc.source_version,
                    source_date,
                    doc.citation,
                    doc.type,
                    doc.id,
                    c.chunk_index,
                    c.content,
                    c.token_count,
                    c.embedding,
                )
                for c in chunks
            ],
            fetch=True,
        )
        # rows is [(id, chunk_index), ...]
        id_by_idx = {ci: rid for rid, ci in rows}

        # Update neighbor links
        for rid, ci in rows:
            prev_id = id_by_idx.get(ci - 1)
            next_id = id_by_idx.get(ci + 1)
            cur.execute(
                "UPDATE chunks SET prev_chunk_id = %s, next_chunk_id = %s WHERE id = %s",
                (prev_id, next_id, rid),
            )
    return len(chunks)
