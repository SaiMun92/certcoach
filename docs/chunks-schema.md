# The `chunks` Table — Explained

The `chunks` table is the heart of CertCoach. Every piece of knowledge the system can retrieve and cite lives here as a row. This document walks through the schema, what each column is for, and how the table is used at query time.

---

## What is a "chunk"?

When CertCoach ingests a source document (e.g. the AWS EC2 FAQ), it:

1. Parses the HTML/PDF into clean Markdown
2. Splits it into overlapping ~650-token windows (chunks) with 100-token overlap
3. Embeds each chunk with `text-embedding-3-large`
4. Inserts a row into `chunks` for each one

A single source document like `faq-ec2` might produce 200–400 chunk rows.

---

## A concrete row

Here is what a single row looks like for a chunk from the AWS EC2 FAQ:

```
id              | 42
tenant          | aws-saa
source_id       | faq-ec2
source_version  | NULL
source_date     | 2026-07-24
citation        | Amazon EC2 FAQs
doc_type        | faq
parent_doc_id   | faq-ec2
prev_chunk_id   | 41
next_chunk_id   | 43
chunk_index     | 7
content         | ## What is Amazon EC2?
                | Amazon Elastic Compute Cloud (Amazon EC2) is a web service that provides
                | resizable compute capacity in the cloud. It is designed to make web-scale
                | cloud computing easier for developers. EC2 instances can be launched in
                | minutes, and you pay only for the capacity you use...
token_count     | 312
embedding       | [0.021, -0.043, 0.107, ... ]  (1536 numbers)
fts             | 'amazon':1,4 'cloud':6 'comput':3,7 'ec2':2 'elastic':3 ...  (auto-generated)
created_at      | 2026-07-24T08:54:00Z
```

---

## Column-by-column

### Identity

| Column | Type | Example | Purpose |
|---|---|---|---|
| `id` | `BIGSERIAL` | `42` | Auto-incrementing row ID. Used by `prev_chunk_id`/`next_chunk_id` to link neighbors. |

---

### Tenancy — the isolation boundary

| Column | Type | Example | Purpose |
|---|---|---|---|
| `tenant` | `TEXT NOT NULL` | `aws-saa` | Which certification this chunk belongs to. **Every single query filters on this column first.** A GCP ACE user can never see AWS chunks, and vice versa. |

Without `tenant`, asking "what is a VPC?" could return an AWS answer to a GCP learner. The `chunks_tenant_idx` B-tree index makes this filter fast.

---

### Source provenance — citations and version tracking

| Column | Type | Example | Purpose |
|---|---|---|---|
| `source_id` | `TEXT NOT NULL` | `faq-ec2` | The document this chunk came from. Matches `DocSpec.id` in `sources.yaml`. |
| `source_version` | `TEXT` | `SAA-C03` | Exam guide version. `NULL` for sources that don't publish a version (e.g. GCP ACE guide). |
| `source_date` | `DATE` | `2026-07-24` | When the source was fetched. Used in citations like "as of 2026-07-24". |
| `citation` | `TEXT NOT NULL` | `Amazon EC2 FAQs` | Human-readable citation shown directly in answers: *"According to Amazon EC2 FAQs..."* |
| `doc_type` | `TEXT NOT NULL` | `faq` | One of: `exam_guide`, `faq`, `whitepaper`, `doc`. Used to weight retrieval results. |

---

### Chunk navigation — expand-to-context

| Column | Type | Example | Purpose |
|---|---|---|---|
| `parent_doc_id` | `TEXT` | `faq-ec2` | The source document this chunk belongs to. Same as `source_id` for top-level docs. |
| `chunk_index` | `INT NOT NULL` | `7` | 0-based position of this chunk within its parent document. |
| `prev_chunk_id` | `BIGINT` | `41` | Row ID of the chunk immediately before this one in the document. |
| `next_chunk_id` | `BIGINT` | `43` | Row ID of the chunk immediately after this one in the document. |

**Why do we need prev/next?** This enables **retrieve-small, expand-to-context**:

```
Document:  [chunk 40] [chunk 41] [chunk 42] [chunk 43] [chunk 44]
                                     ↑
                              retriever finds this
                              as the best match

Expanded context sent to LLM:
                      [chunk 41] [chunk 42] [chunk 43]
                      (neighbors give the LLM more room to reason)
```

The retriever finds the tight, precise chunk (good for matching). The LLM gets that chunk plus its neighbors (good for answering). Without neighbors, an answer that straddles a chunk boundary would be cut off.

---

### Content

| Column | Type | Example | Purpose |
|---|---|---|---|
| `content` | `TEXT NOT NULL` | `## What is Amazon EC2?\nAmazon Elastic...` | The chunk text in Markdown. All sources (HTML and PDF) are normalized to Markdown at ingestion. |
| `token_count` | `INT` | `312` | Approximate token count. Used to enforce the ~500–800 token budget at ingestion time. |

---

### The two retrieval columns

These are the most important columns — they power the hybrid retrieval pipeline.

#### `embedding vector(1536)` — dense search

```
embedding | [0.021, -0.043, 0.107, 0.089, -0.201, ... ]  ← 1536 numbers
```

This is the output of `text-embedding-3-large` for this chunk's text. Two pieces of text with similar *meaning* will have vectors that point in similar directions — that's how semantic search works.

At query time:
```sql
SET LOCAL ivfflat.probes = 10;  -- increase probe count for better recall on small corpora

SELECT id, tenant, source_id, chunk_index, content, token_count,
       citation, doc_type, source_date::text,
       prev_chunk_id, next_chunk_id,
       1 - (embedding <=> $query_embedding::vector) AS score
FROM chunks
WHERE tenant = 'aws-saa'
ORDER BY embedding <=> $query_embedding::vector
LIMIT 20;
```

The `<=>` operator is provided by pgvector. The IVFFlat index makes this fast by dividing the vector space into 100 buckets (lists) and only searching the nearest ones instead of comparing every row.

**Why 1536 and not 3072?** `text-embedding-3-large` natively outputs 3072 dimensions, but pgvector 0.8.6 (the version in the Docker image) cannot build an ANN index on more than 2000 dimensions. OpenAI built the model with **Matryoshka Representation Learning**, which means you can truncate to 1536 dimensions with only ~2–3% quality loss. So we use 1536 — full IVFFlat support, nearly full quality.

#### `fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED` — sparse/BM25 search

```
fts | 'amazon':1,4 'cloud':6,9 'comput':3,7 'ec2':2 'elast':3 'servic':11 ...
```

This is a **generated column** — Postgres computes it automatically from `content` on every insert or update. You never write to it from application code.

`tsvector` is Postgres's built-in keyword index format. It stores each word's stem and position(s). `to_tsvector('english', content)` applies English stemming: "computing", "computed", "compute" all become the stem `comput`.

At query time (`backend/retrieval.py:sparse_search()`):
```sql
SELECT id, tenant, source_id, chunk_index, content, token_count,
       citation, doc_type, source_date, prev_chunk_id, next_chunk_id,
       ts_rank_cd(fts, plainto_tsquery('english', $query)) AS score
FROM chunks
WHERE tenant = 'aws-saa'
  AND fts @@ plainto_tsquery('english', $query)
ORDER BY score DESC
LIMIT 20;
```

`ts_rank_cd()` scores each matching chunk by keyword density — how often and how closely together the query terms appear. Results are ordered by this score so the most keyword-relevant chunks rank first. All 11 chunk columns are selected because `prev_chunk_id`/`next_chunk_id` are needed downstream for the expand-to-context step.

**How the WHERE clause works:**
- `plainto_tsquery('english', $query)` converts the user's query into a tsquery — a boolean expression of stemmed AND terms. For example, `'EC2 instance types'` → `'ec2' & 'instanc' & 'type'`.
- `fts @@ plainto_tsquery(...)` is the match operator (`@@`). For each row it asks: "does the `fts` column (the pre-computed keyword index for this chunk) contain all the stemmed query terms?" Rows where this is `true` pass through; rows where it's `false` are excluded. Only passing rows are then scored by `ts_rank_cd`.
- The GIN index (`chunks_fts_gin_idx`) accelerates the `@@` filter — without it, Postgres would evaluate `ts_rank_cd` against every row in the table.

**Why do we need both dense and sparse search?** Each has a blind spot the other covers.

**Dense search is blind to exact tokens.** The embedding for `s3:GetObject` is mathematically similar to embeddings for other IAM actions and S3 operations — the vector space is crowded with related concepts. A query specifically about `s3:GetObject` may return chunks about S3 permissions in general rather than that exact action name.

**Sparse search is blind to meaning.** If a user asks "what are the memory-optimised instance families?" but the relevant chunk says "R-series instances are designed for memory-intensive workloads" — the words "memory-optimised" don't appear verbatim, so the keyword filter misses it entirely.

Cloud cert content has both problems in abundance:
- **Exact tokens that matter**: IAM action names (`sts:AssumeRole`), instance types (`r6g.xlarge`), CIDR blocks (`10.0.0.0/16`), CLI flags
- **Paraphrased concepts**: "high availability" vs "fault tolerant", "object storage" vs "S3", "auto scaling" vs "elastic scaling"

Running both and fusing with RRF means a chunk only needs to win on **one** dimension to be retrieved — but chunks that score well on **both** float to the top automatically.

---

## The 4 indexes

```sql
-- 1. Dense vector ANN index
CREATE INDEX chunks_embedding_ivfflat_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```
**IVFFlat = Inverted File with Flat compression.** At build time, k-means clustering divides all vectors into `lists=100` buckets. At query time, `SET LOCAL ivfflat.probes = 10` tells it to scan the 10 nearest buckets instead of all 100 — comparing the query against every vector stored flat in those buckets (exact comparison, no compression).

**How many vectors does it actually compare?**
- 300 chunks ÷ 100 buckets = ~3 vectors per bucket
- `probes=10` → scan 10 buckets → compare ~30 vectors instead of all 300

At this corpus size the speedup is modest — a full scan of 300 vectors is fast anyway. The `lists=100` setting follows the pgvector rule of thumb (`sqrt(num_rows)`), which is calibrated for large corpora. At 300 rows, `lists=17` would be more appropriate; `lists=100` means ~3 vectors per bucket, which barely differs from a full scan.

**Why keep it then?** The index is production-shaped for corpus growth. At 10,000 chunks per tenant, `lists=100` and `probes=10` would genuinely narrow from 10,000 vectors down to ~1,000 — a real 10× speedup. The architecture is correct; the corpus is just portfolio-sized right now.

**The tradeoff:** IVFFlat is *approximate* — if the true nearest vector sits in bucket 11, it gets missed. For RAG this is acceptable; a slightly suboptimal chunk rank is corrected downstream by RRF fusion or the reranker.

**Index staleness and automatic rebuild.** The 100 centroids are computed once at index build time. New vectors inserted afterward are assigned to the nearest existing bucket — the centroids are never recomputed automatically. Over time, as the corpus grows, buckets become unbalanced and ANN quality degrades. To fix this, the ingestion pipeline (`ingestion/pipeline.py`) automatically runs `REINDEX INDEX chunks_embedding_ivfflat_idx` after every ingestion run that produces at least one chunk. This reruns k-means clustering from scratch over all current vectors and redistributes everything into fresh balanced buckets.

```sql
-- 2. Sparse / BM25 index
CREATE INDEX chunks_fts_gin_idx ON chunks USING gin (fts);
```
GIN (Generalised Inverted iNdex) maps every word stem to the list of rows containing it — exactly like a book's index. Without it, full-text queries would scan every row's `fts` value.

```sql
-- 3. Tenant filter
CREATE INDEX chunks_tenant_idx ON chunks (tenant);
```
Every query has `WHERE tenant = ?`. This index lets Postgres jump directly to the right tenant's rows before touching the vector or FTS indexes.

```sql
-- 4. Tenant + source composite
CREATE INDEX chunks_tenant_source_idx ON chunks (tenant, source_id);
```
Used for two things:
- **Corpus-gap diagnostics**: "Is `faq-rds` even in the corpus for `aws-saa`?" — one index scan.
- **Per-document re-ingestion**: delete all chunks for `(aws-saa, faq-ec2)` and re-insert when the source is re-fetched.

---

## How a query uses this table

A user on the AWS SAA tenant asks: *"What EC2 instance types are optimised for memory?"*

```
Query: "What EC2 instance types are optimised for memory?"
Tenant: aws-saa

Step 1 — Dense search (embedding <=>):
  finds chunks about EC2 families by meaning
  → returns top 20 by cosine similarity

Step 2 — Sparse search (fts @@):
  finds chunks containing "EC2", "instance", "memory", "optimis*"
  → returns top 20 by BM25 rank

Step 3 — RRF fusion:
  formula: score = 1/(60 + rank_in_dense) + 1/(60 + rank_in_sparse)
  a chunk appearing in both lists scores ~2× one that appears in only one
  example:
    chunk A: dense rank 1, sparse rank 2 → 1/61 + 1/62 = 0.0325
    chunk B: dense rank 1, sparse rank — → 1/61 + 0     = 0.0164
  no score calibration needed — only rank position matters
  → single ranked list of ~20 unique chunks

Step 4 — [Optional] Cohere rerank (hybrid mode only, ~2400ms):
  re-scores all 20 against the original question
  → top 5 chunks
  Production default (hybrid_no_rerank) skips this step;
  RRF score is normalised to 0–1 for confidence (~10ms)

Step 5 — Expand to context:
  for each of the 5 chunks, fetch prev_chunk_id and next_chunk_id
  → 5 triplets of chunks

Step 6 — Generate answer:
  LLM reads the 5 triplets + original question
  → grounded answer with citations from the `citation` column
```

Both Step 1 and Step 2 filter by `tenant = 'aws-saa'` before doing any similarity work. That is the isolation guarantee.

---

## Idempotency — why you can run the migration twice safely

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, ...);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM schema_migrations WHERE version = '001') THEN
        RAISE NOTICE 'Migration 001 already applied, skipping.';
        RETURN;  -- exits the block immediately
    END IF;

    CREATE TABLE chunks (...);
    -- ... indexes ...
    INSERT INTO schema_migrations (version) VALUES ('001');
END $$;
```

The `DO` block is a PL/pgSQL anonymous function. On first run: `schema_migrations` has no row for `'001'`, so it falls through to `CREATE TABLE chunks` and inserts the tracking row at the end. On second run: it finds `version = '001'` and returns immediately — no error, no duplicate table.

This matters because `apply_migrations()` in `backend/db.py` re-runs all `.sql` files on every startup to ensure the schema is current. Safe to call in Docker entrypoints, CI, or after a container restart.
