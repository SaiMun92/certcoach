# Contextual Retrieval — Design Spec

**Date:** 2026-08-10
**Status:** Approved

## Background

Anthropic's Contextual Retrieval technique prepends a short LLM-generated blurb to each chunk
before embedding it, situating the chunk within its source document. Anthropic measured ~35%
retrieval-failure reduction with this alone, ~49% combined with BM25. This spec implements it
as a measurable, A/B-comparable enhancement to CertCoach's ingestion and retrieval pipeline.

---

## Goal

Run contextual retrieval alongside the existing pipeline so eval numbers can be compared
directly. The portfolio story is: "contextual retrieval reduced retrieval failures by X% on
my AWS/GCP corpus."

---

## Section 1 — Schema & Migration

**File:** `migrations/002_contextual_embeddings.sql`

Adds two nullable columns to the existing `chunks` table:

| Column | Type | Purpose |
|---|---|---|
| `context_blurb` | `TEXT` | LLM-generated 1-2 sentence situating description |
| `contextual_embedding` | `vector(1536)` | Embedding of `blurb + "\n\n" + content` |

An IVFFlat index is created on `contextual_embedding` with the same config as the existing
`embedding` index (`lists=100`, `vector_cosine_ops`).

Migration is idempotent via the existing `schema_migrations` guard pattern (version `'002'`).
Existing rows are unaffected — both columns remain NULL until `--contextual` is run.

---

## Section 2 — Blurb Generation

**File:** `ingestion/contextualize.py`

Single public function:

```python
def generate_blurb(chunk_content: str, doc_title: str, tenant: str) -> str
```

- Calls `claude-haiku-4-5` via the existing `generate()` in `backend/generate.py`
- Prompt instructs the model to write 1-2 sentences naming the specific service, concept, or
  section covered by the chunk — optimised for retrieval, not human reading
- Returns the blurb string
- No retry logic — failures propagate to the pipeline's existing per-document try/except,
  which skips and rolls back the document

**Prompt template:**

```
Given this chunk from "{doc_title}" ({tenant} certification material),
write 1-2 sentences situating it for a retrieval system.
Be specific: name the service, concept, or section this chunk covers.
Chunk: {chunk_content}
```

---

## Section 3 — Pipeline Integration

**File:** `ingestion/pipeline.py`

`run_pipeline()` gains `contextual: bool = False`. CLI gains `--contextual` flag.

When `contextual=True`:
1. After chunking a document, call `generate_blurb()` for each chunk sequentially
2. Embed `blurb + "\n\n" + content` (instead of `content` alone)
3. Pass `context_blurb` and `contextual_embedding` into `ChunkRow`
4. `store_chunks` writes them to the new columns

When `contextual=False` (default):
- Pipeline is completely unchanged — no blurb calls, content-only embedding, NULL columns

`ChunkRow` in `ingestion/store.py` gains two optional fields:
- `context_blurb: str | None = None`
- `contextual_embedding: list[float] | None = None`

`store_chunks` writes them when present, skips when None.

---

## Section 4 — Retrieval Integration

**File:** `backend/retrieval.py`

`dense_search()` gains `use_contextual: bool = False`:
- When `True`: queries `contextual_embedding` column, adds `WHERE contextual_embedding IS NOT NULL`
- When `False`: unchanged, queries `embedding` column

`retrieve()` gains a new `retrieval_mode` value: `"contextual"`:
- Dense search uses `contextual_embedding`
- Sparse BM25 search runs unchanged on `content`
- RRF fusion, rerank, and expand-chunks are all unchanged
- `context_blurb` is ingestion-only and is never sent to the LLM at generation time
- `ChunkResult.content` remains the raw chunk text — the answer prompt is unaffected

---

## Section 5 — Eval Harness

**File:** `backend/eval.py`

Add `"contextual"` as a valid `retrieval_mode` value. No other eval changes.

This allows direct A/B comparison:
```
retrieval_mode=hybrid_no_rerank   # baseline
retrieval_mode=contextual          # contextual variant
```

Same judge, same questions, same scoring — only the retrieval path differs.

---

## Out of Scope

- Streaming blurb generation (not needed at ingestion time)
- Caching blurbs across re-ingestion runs (idempotent delete+reinsert handles this)
- Using the blurb in the answer prompt (ingestion-only by design)
- Parallel blurb generation (sequential avoids rate limits; Haiku is fast enough)

---

## Cost Estimate

~1 Haiku call per chunk. Typical corpus: ~500–1000 chunks across both tenants.
At Haiku pricing (~$0.25/1M input tokens, ~350 tokens/chunk prompt), full re-ingestion
with `--contextual` costs < $0.50.
