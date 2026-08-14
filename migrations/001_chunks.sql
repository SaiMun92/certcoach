-- Enable pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Guard: skip if already applied
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM schema_migrations WHERE version = '001') THEN
        RAISE NOTICE 'Migration 001 already applied, skipping.';
        RETURN;
    END IF;

    -- Main chunks table
    CREATE TABLE chunks (
        id              BIGSERIAL PRIMARY KEY,
        tenant          TEXT        NOT NULL,          -- 'aws-saa' | 'gcp-ace'
        source_id       TEXT        NOT NULL,          -- matches DocSpec.id from sources.yaml
        source_version  TEXT,                          -- e.g. 'SAA-C03', null if unknown
        source_date     DATE,                          -- date source was fetched
        citation        TEXT        NOT NULL,          -- human-readable citation string
        doc_type        TEXT        NOT NULL,          -- 'exam_guide'|'faq'|'whitepaper'|'doc'
        parent_doc_id   TEXT,                          -- source_id of the parent document
        prev_chunk_id   BIGINT,                        -- neighbor for expand-to-context
        next_chunk_id   BIGINT,                        -- neighbor for expand-to-context
        chunk_index     INT         NOT NULL,          -- 0-based position within parent doc
        content         TEXT        NOT NULL,          -- chunk text (Markdown)
        token_count     INT,                           -- approximate token count
        embedding       vector(1536),                  -- text-embedding-3-large with dimensions=1536 (Matryoshka truncation; full 3072 requires pgvector ≥0.9)
        fts             tsvector GENERATED ALWAYS AS   -- sparse BM25 index
                            (to_tsvector('english', content)) STORED,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- Dense vector index: IVFFlat with 1536-dim embeddings
    CREATE INDEX chunks_embedding_ivfflat_idx
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

    -- Sparse / BM25 index (GIN on generated tsvector column)
    CREATE INDEX chunks_fts_gin_idx
        ON chunks USING gin (fts);

    -- Tenant filter index (every retrieval query filters by tenant)
    CREATE INDEX chunks_tenant_idx ON chunks (tenant);

    -- Tenant + source composite (for corpus-gap diagnostics and per-doc upsert)
    CREATE INDEX chunks_tenant_source_idx ON chunks (tenant, source_id);

    INSERT INTO schema_migrations (version) VALUES ('001');
    RAISE NOTICE 'Migration 001 applied.';
END $$;
