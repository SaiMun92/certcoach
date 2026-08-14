# CertCoach — Architecture Deep-Dive

> Interview reference document. Covers every layer of the system with pointers to the
> exact source files and functions that implement each concern.

---

## 1. System Overview

CertCoach is a **multi-tenant RAG (Retrieval-Augmented Generation) coaching assistant** for
cloud certification exams. It exposes a FastAPI backend, a Streamlit chat UI, and a
full evaluation harness with LLM-as-judge scoring.

**Tenants:** `aws-saa` (AWS Solutions Architect Associate), `gcp-ace` (GCP Associate Cloud
Engineer), `terraform-assoc` (HashiCorp Terraform Associate). All share one Postgres schema;
every row is tagged with a `tenant` column.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User / Streamlit UI  (frontend/streamlit_app.py)                           │
│         │  HTTP POST /ask  +  X-Tenant header                               │
│         ▼                                                                   │
│  FastAPI  (backend/api.py)                                                  │
│         │                                                                   │
│         ├─► retrieve()  (backend/retrieval.py)                              │
│         │       ├─► embed()  →  dense_search()  (pgvector cosine)          │
│         │       ├─► sparse_search()  (Postgres full-text, tsvector)        │
│         │       ├─► rrf_fuse()  (Reciprocal Rank Fusion)                   │
│         │       ├─► [rerank()  Cohere Rerank Pro — hybrid mode only]       │
│         │       └─► expand_chunks()  (prev/next neighbor fetch)            │
│         │                                                                   │
│         └─► answer_question()  (backend/generate.py)                       │
│                 └─► generate()  →  claude-sonnet-4-6 via AI Core           │
│                                                                             │
│  Postgres + pgvector  (Docker)                                              │
│  Ingestion pipeline  (ingestion/)                                           │
│  Eval harness  (backend/eval.py)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥3.11 (tested on 3.13) |
| API framework | FastAPI + Uvicorn |
| Vector store | PostgreSQL 16 + pgvector extension |
| Embeddings | `text-embedding-3-large` (1536-dim, truncated from 3072) |
| Reranker | Cohere Rerank Pro (`cohere-rerank-pro`) |
| Generator | `claude-sonnet-4-6` (default), swappable via `model=` arg |
| Eval judge | `gpt-5` (different family from generator — avoids self-grading bias) |
| Model gateway | SAP AI Core (OpenAI-compatible proxy for embeddings/generation; direct OAuth2 for rerank) |
| UI | Streamlit |
| Chunker tokenizer | `cl100k_base` (tiktoken) |
| Tests | pytest, 105 tests |
| Containers | Docker Compose — `db` (pgvector), `phoenix`, `api`, `frontend` |

---

## 3. Ingestion Pipeline

**Entry point:** `ingestion/pipeline.py` → `run_pipeline()`

### 3.1 Manifest

`ingestion/sources.yaml` declares every document per tenant: URL, format (html/pdf),
corpus flag, and citation string. The manifest loader (`ingestion/sources.py`) produces
a typed `Manifest` object. Adding a new tenant or document is a YAML edit only.

### 3.2 Document cleaning

`ingestion/clean.py` → `clean_doc(path, format)`

- **HTML:** `markdownify` converts to Markdown, then regex strips nav/footer boilerplate.
- **PDF:** `pdfminer.six` extracts text; whitespace normalisation removes hyphenation artefacts.

Output is plain Unicode text ready for chunking.

### 3.3 Chunking strategy

**File:** `ingestion/chunk.py` → `chunk_text()`

**Algorithm: sliding-window with paragraph-aware breaking**

| Parameter | Value |
|---|---|
| Target window | 650 tokens |
| Overlap | 100 tokens |
| Tokenizer | `cl100k_base` (same as OpenAI models) |
| Boundary preference | Last paragraph break (`\n{2,}` — two or more blank lines) within the window |

**How it works:**

1. Encode the whole document to tokens.
2. If total ≤ 650 tokens, return one chunk.
3. Otherwise, slide a 650-token window forward.
4. Within that window, find the **last blank line** (`\n{2,}`) using `_PARA_RE.finditer()`.
   Measure how many tokens fall between the window start and that blank line.
   - If that count is **≥ 50 tokens** → snap the cut to the blank line. This ensures the
     chunk ends at a natural paragraph boundary rather than mid-sentence.
   - If that count is **< 50 tokens** → ignore the break and cut at the full 650-token
     boundary. This prevents a paragraph break very near the start of the window from
     producing a tiny near-empty chunk.
   - If **no blank line exists** in the window → cut at the full 650-token boundary.
5. Advance the window by `window_size − overlap` (≈ 550 tokens), preserving a 100-token
   tail in the next chunk for continuity.

**Why this matters in interviews:** The paragraph-snap means chunks tend to end at a
natural prose boundary rather than mid-sentence. The 100-token overlap means context
near a boundary appears in both adjacent chunks, reducing the chance a retrieval
miss is caused purely by the cut point.

### 3.4 Embedding at ingest time

`ingestion/pipeline.py` calls `embed(texts)` in batches of 32 (`batch_size=32`).
Each chunk's 1536-dim vector is stored in the `embedding` column (`vector(1536)` via pgvector).

The `embed()` call targets `text-embedding-3-large` with `dimensions=1536` — this uses
OpenAI's Matryoshka representation learning to truncate from 3072 to 1536 dimensions,
halving storage and compute cost with minimal quality loss.

### 3.5 FTS index at ingest time

`fts` is a **SQL generated column** defined in `migrations/001_chunks.sql`:

```sql
fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

Postgres computes and stores it automatically on every insert or update — no trigger,
no application-level call. `ingestion/store.py` never writes to `fts`; it only inserts
the `content` column and Postgres handles the rest. This enables BM25-style keyword
search without a separate search engine.

### 3.6 Database schema

Key table: `chunks`

| Column | Type | Purpose |
|---|---|---|
| `id` | serial PK | |
| `tenant` | text | Multi-tenancy partition key |
| `source_id` | text | Document identifier (e.g. `faq-ec2`) |
| `chunk_index` | int | Position within document |
| `content` | text | Raw chunk text |
| `token_count` | int | Pre-computed token count |
| `embedding` | vector(1536) | Dense vector for ANN search |
| `fts` | tsvector | Sparse index for keyword search |
| `citation` | text | Human-readable source label for citations |
| `prev_chunk_id` / `next_chunk_id` | int FK | Linked list for context expansion |

---

## 4. Retrieval Pipeline

**File:** `backend/retrieval.py` → `retrieve()`

The retrieval pipeline has three configurable modes, selected via `retrieval_mode`:

```
query
  │
  ├─[all modes]──► embed(query) → 1536-dim vector
  │
  ├─[all modes]──► dense_search()   → top-20 by cosine similarity  (pgvector)
  │
  ├─[hybrid*]────► sparse_search()  → top-20 by BM25/tsvector      (Postgres FTS)
  │
  ├─[hybrid*]────► rrf_fuse()       → merged ranked list
  │
  ├─[hybrid]─────► rerank()         → Cohere Rerank Pro, top-5 from top-20 candidates
  │
  └─[all]────────► expand_chunks()  → fetch prev/next neighbors
                        │
                        ▼
                   RetrievalResult(chunks, confidence, below_threshold)
```

### 4.1 Dense search — `dense_search()`

```python
SELECT ... , 1 - (embedding <=> %s::vector) AS score
FROM chunks WHERE tenant = %s
ORDER BY embedding <=> %s::vector LIMIT 20
```

Uses the **cosine distance operator** (`<=>`) from pgvector. Scores are converted to
cosine similarity (1 − distance). IVFFlat index with `ivfflat.probes = 10` — the
default of 1 probe is too low for small corpora and causes recall degradation;
10 probes recovers full recall.

### 4.2 Sparse search — `sparse_search()`

```python
SELECT ... , ts_rank_cd(fts, plainto_tsquery('english', %s)) AS score
FROM chunks WHERE tenant = %s AND fts @@ plainto_tsquery('english', %s)
ORDER BY score DESC LIMIT 20
```

`plainto_tsquery` tokenises and stems the query; `ts_rank_cd` weights by cover density
(term frequency weighted by proximity). This captures exact keyword matches that semantic
embeddings may miss — acronyms, version numbers, CLI flags.

### 4.3 Reciprocal Rank Fusion — `rrf_fuse()`

RRF combines the two ranked lists without needing calibrated scores:

```
RRF_score(chunk) = Σ  1 / (k + rank_in_list)
                   lists
```

`k = 60` (standard default). Chunks appearing in both lists get additive contributions.
The result is a single merged ranked list with RRF scores (~0.016 at top-1 for this corpus size).

**Key property:** RRF is robust to score scale differences between dense (cosine
similarity, 0–1) and sparse (ts_rank_cd, unbounded) search. It only uses rank position,
not the raw scores.

### 4.4 Reranking — `rerank()`

**File:** `backend/ai_client.py` → `rerank()`

- Sends the query + top-20 candidate chunks to **Cohere Rerank Pro** via the SAP AI Core
  deployment endpoint (`/v2/inference/deployments/{id}/rerank`).
- The reranker is a cross-encoder: it jointly encodes query and each document, producing
  calibrated relevance scores (0–1 range) that are more accurate than bi-encoder cosine
  similarity.
- Returns top-5 by rerank score.
- Authentication: OAuth2 client-credentials flow (separate from the OpenAI-compatible proxy).
  Token is cached with a 60-second safety margin before expiry.

**Confidence gate:** `confidence_threshold = 0.1`. If the top rerank score < 0.1, the
result is flagged `below_threshold=True` and the generator returns a canned
"insufficient information" response instead of hallucinating.

### 4.5 Context expansion — `expand_chunks()`

After ranking, each selected chunk's `prev_chunk_id` and `next_chunk_id` are fetched in
a single batch query. The `ExpandedChunk` wraps the core chunk with its neighbors'
content. When the LLM prompt is assembled (`backend/prompts.py` → `format_context()`),
the context string is:

```
[Citation]
<prev_content>

<core_content>

<next_content>
```

This gives the model ≈ 3× the context of a single chunk without tripling the retrieval
scope, and ensures sentences at chunk boundaries are not truncated.

### 4.6 Retrieval modes summary

| Mode | Dense | Sparse | RRF | Rerank | Confidence gate | Use case |
|---|---|---|---|---|---|---|
| `dense_only` | ✓ | ✗ | ✗ | ✗ | ✗ | Baseline |
| `hybrid_no_rerank` | ✓ | ✓ | ✓ | ✗ | ✗ (bypassed) | **Production default** |
| `hybrid` (full) | ✓ | ✓ | ✓ | ✓ | ✓ | Available, slower |

The confidence gate is bypassed in `hybrid_no_rerank` mode because RRF scores
(`1/(60+rank)`) are not on the same 0–1 scale as rerank scores. Instead, the raw RRF
score is **normalised to 0–1** against the theoretical maximum (`2/(60+1) ≈ 0.033` —
top-1 in both lists), so the confidence field is human-interpretable.

---

## 5. Generation

**File:** `backend/generate.py` → `answer_question()`

1. If `below_threshold=True`: return a canned response — no LLM call.
2. Otherwise, call `format_context()` to assemble the context string from expanded chunks.
3. Build the prompt with `build_answer_messages()` from `backend/prompts.py`.
4. Call `generate()` → `client.chat.completions.create()` via the AI Core proxy.

### 5.1 Prompt v1 (default)

```
System: CRITICAL RULE: Your very first word must be a content word. NEVER start with
        "Based on", "According to", "From the context", or any variant.

        You are CertCoach, an expert tutor for cloud certification exams.
        Answer using ONLY the context provided. Cite each claim with the
        source name in parentheses. Be concise: 3-5 sentences.

# Few-shot turn (mid-conversation example to enforce answer style)
User:   Question: What is Amazon S3?
        Context: [Amazon S3 FAQs] Amazon S3 is object storage...
Assistant: Amazon S3 (Simple Storage Service) is object storage... (Amazon S3 FAQs).

User:   <question>{query}</question>

        Context:
        {formatted_chunks}

        Answer directly — start with the first content word, not "Based on the context" or similar.
```

The query is wrapped in `<question>…</question>` XML tags to clearly delimit it from the context block.
The few-shot turn is injected as a mid-conversation `user`/`assistant` pair before the real query.

### 5.2 Prompt v2 (experimental — regressed)

Added chain-of-thought structure with enumeration rules and a "Reasoning:" prefix.
**Eval result: v2 regressed on accuracy** (aws-saa: 0.790 vs v1 0.901 in the final A/B
run). The gpt-5 judge penalises verbose structured answers against concise reference
answers. v1 remains the default.

### 5.3 Model gateway

All generation/embedding calls go through `backend/ai_client.py` → `_get_client()`:
an `openai.OpenAI` instance pointed at `AICORE_BASE_URL`. Model IDs are SAP AI Core
deployment aliases (e.g. `claude-sonnet-4-6`, `gpt-5`, `text-embedding-3-large`) —
not standard Anthropic or OpenAI IDs. The client is `functools.lru_cache`'d to avoid
reconstructing it per request.

---

## 6. Evaluation Harness

**File:** `backend/eval.py` → `run_eval()` and `run_ab_eval()`

### 6.1 Gold sets

| File | Items | Coverage |
|---|---|---|
| `eval/gold_retrieval.jsonl` | 25 | 15 aws-saa, 5 gcp-ace, 5 terraform-assoc |
| `eval/gold_qa.jsonl` | 25 | 10 aws-saa, 10 gcp-ace, 5 terraform-assoc |

Each retrieval item: `{question, tenant, relevant_source_ids: [str]}`.
Each QA item: `{question, tenant, reference_answer: str}`.

### 6.2 Retrieval metrics

Computed in `run_eval()`:

| Metric | Definition |
|---|---|
| `recall@5` | 1 if any relevant source_id appears in top-5 retrieved, else 0 |
| `hit@1` | 1 if the very first retrieved result is relevant, else 0 |
| `mrr@10` | 1/rank of the first relevant result in top-10 |
| `ndcg@5` | Normalised Discounted Cumulative Gain at k=5 |

### 6.3 Answer accuracy (LLM-as-judge)

`judge_answer()` sends the question, reference answer, and candidate answer to `gpt-5`
with a system prompt asking for a `{"score": float, "rationale": str}` JSON response.
Score is 0.0–1.0. The judge is a **different model family** from the generator
(GPT-5 judges Claude's output) to avoid self-grading bias.

### 6.4 Performance optimisations in eval

**Batch embedding:** All questions across the gold set are embedded in **2 API
round-trips** (one for retrieval gold, one for QA gold) using `embed(list_of_questions)`.
The resulting vectors are passed as `query_embedding=` to `retrieve()`, which skips the
per-question embed call.

**ThreadPoolExecutor parallelism:** Both the retrieval loop and the QA loop use
`ThreadPoolExecutor(max_workers=8)`. Each worker acquires its own `get_conn()` DB
connection (psycopg2 connections are not thread-safe). Results are collected via
`as_completed()`.

### 6.5 A/B eval results (final run)

`run_ab_eval()` runs four configs and prints a comparison table.

| Config | recall@5 | hit@1 | mrr@10 | ndcg@5 | accuracy | latency P50 |
|---|---|---|---|---|---|---|
| dense_only | 1.000 | 0.967 | 0.983 | 0.988 | 0.714 | ~3500ms |
| **hybrid_no_rerank** | **1.000** | **0.933** | **0.967** | **0.975** | **0.828** | **~10ms** |
| hybrid+rerank | 1.000 | 0.967 | 0.978 | 0.983 | 0.901 | ~2400ms |
| hybrid+rerank+v2 | 1.000 | 0.933 | 0.956 | 0.967 | 0.790 | ~2400ms |

**Observations:**
- `hybrid+rerank` achieves the best answer accuracy (0.901) but at 2400ms latency.
- `hybrid_no_rerank` is 200× faster (10ms) with competitive accuracy (0.828) — chosen
  as the production default given the corpus size.
- Rerank adds value at large corpus scale (more candidates to re-sort); at small scale
  (≈200 chunks) the RRF ordering is already strong enough.
- Prompt v2 regresses accuracy — verbose chain-of-thought answers score lower against
  concise reference answers under the judge's scoring rubric.

---

## 7. API

**File:** `backend/api.py`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{status, db}` — checks DB connectivity |
| `/ask` | POST | Main Q&A endpoint. Body: `{question}`. Header: `X-Tenant`. |
| `/quiz` | POST | Generate practice questions. Body: `{topic, n}`. Header: `X-Tenant`. |
| `/grade` | POST | Grade a learner answer. Body: `{question, learner_answer, reference}`. Header: `X-Tenant`. |

Multi-tenancy is enforced via the `X-Tenant` header. Valid values:
`aws-saa`, `gcp-ace`, `terraform-assoc`. Unknown tenants return 400.

---

## 8. One-command deploy

`docker compose up --build` starts all four services:

| Service | Container | Port |
|---|---|---|
| Postgres + pgvector | `certcoach-db` | 5433 |
| Phoenix tracing UI | `certcoach-phoenix` | 6006 (UI), 4317 (OTLP) |
| FastAPI backend | `certcoach-api` | 8000 |
| Streamlit UI | `certcoach-frontend` | 8501 |

Teardown: `docker compose down` (keeps volumes) or `docker compose down -v` (wipes data).

---

## 9. Design Decisions Worth Defending

**Why pgvector instead of a dedicated vector DB (Pinecone, Weaviate)?**
The corpus is small (hundreds of chunks per tenant). pgvector eliminates operational
complexity — one fewer service to deploy, and transactional consistency between the
relational metadata and the vectors is free.

**Why RRF instead of score normalisation for fusion?**
Dense scores (cosine similarity) and sparse scores (ts_rank_cd) live on completely
different scales. Normalising requires empirical calibration. RRF only uses rank
position, making it calibration-free and robust.

**Why batch-embed + ThreadPoolExecutor in eval?**
Without batching, N eval questions = N sequential embedding round-trips (~2–3s each =
hours for a large gold set). Batching reduces to 2 round-trips. Parallelising the
retrieval+generation loop with 8 workers cuts wall time by ~8× on top of that.

**Why is the confidence gate bypassed in `hybrid_no_rerank`, and what is the confidence score?**
The gate compares a score against `confidence_threshold=0.1`. Rerank scores are
calibrated 0–1 floats. RRF scores are `1/(60+rank)` — the highest possible is
`2/61 ≈ 0.033` (top-1 in both lists). Applying a 0.1 threshold to raw RRF scores
would flag everything as below-threshold. Instead, the RRF score is **normalised to 0–1**
against that theoretical max (`min(rrf_score / (2/61), 1.0)`), making the confidence
field meaningful: a result at rank 1 in both dense and sparse search reports ~1.0.

**Why `gpt-5` as the eval judge when `claude-sonnet-4-6` is the generator?**
Using the same model to judge its own outputs introduces self-grading bias — the model
tends to favour answers with its own stylistic patterns. Cross-family judging (GPT judges
Claude) gives a more honest signal.

---

## 10. Interview Q&A

### "Why does rerank *hurt* your accuracy (0.828 vs 0.901 without rerank)?"

Lead with the result, not a defence:

> "That's real — I'm not going to explain it away. But I know why it happens at this scale, and I know what would flip it."

**Root cause 1 — the confidence gate is triggering false abstentions (most likely driver)**

With rerank active there is a confidence gate: if the top Cohere score is below 0.1, the
system returns a canned "insufficient information" response instead of generating an answer.
The judge scores that as zero. On a ~300-chunk corpus where `recall@5 = 1.000` — meaning
the right chunk is *always* in the top 5 — some questions are almost certainly triggering
that gate unnecessarily. The reranker is being conservative on a corpus it wasn't calibrated
against, and the gate converts that conservatism into hard zeros.

Verifiable: log every request where `below_threshold=True` during eval, count them, and
check whether their source IDs are actually in the corpus. At 25 eval items, one false
abstention is a 4-point accuracy swing — enough to explain a significant part of the gap.

**Root cause 2 — the reranker is re-sorting an already-strong list**

At ~300 chunks per tenant, RRF fusion already produces strong ordering. The tell: `hit@1`
*drops* from 0.933 to 0.967 with rerank on `dense_only`, and the full `hybrid+rerank` path
only recovers to match `dense_only` on `hit@1`. The reranker is designed to discriminate
between 20 genuinely ambiguous candidates — at this corpus size they are not ambiguous.

**When rerank *would* help**

Two-stage retrieval pays off when the candidate pool contains genuinely ambiguous entries —
multiple chunks with similar cosine/BM25 scores where only one actually answers the
question. That happens at larger corpus scale (~5,000+ chunks). The crossover point is
roughly 10× this corpus size. Adding the full AWS documentation suite (not just FAQs)
would likely flip the result; the architecture is right, the data scale makes rerank
look redundant right now.

**What to fix first**

Tune or remove the confidence gate in rerank mode — or calibrate the threshold against the
actual Cohere score distribution for this corpus rather than using a hardcoded 0.1. That
alone would likely recover most of the accuracy gap without touching the retrieval pipeline.

---

### "What would you do next to improve this system?"

In priority order:

1. **Calibrate the confidence gate** — score-distribution analysis on the rerank output to
   set a data-driven threshold instead of 0.1.
2. **Contextual Retrieval** (Anthropic) — prepend a 1–2 sentence LLM-generated context
   blurb to each chunk at ingest time before embedding. Anthropic measured ~35–49%
   retrieval-failure reduction. Cost: one `claude-haiku-4-5` call per chunk at ingest, negligible at this corpus size.
3. **Larger corpus** — add more AWS/GCP source documents. Most architectural choices
   (rerank, IVFFlat) don't fully demonstrate their value at ~300 chunks.
4. **Larger gold set** — 25 items gives wide confidence intervals (one wrong answer = 4-point swing). 100 items would make the A/B comparisons statistically meaningful.
5. **Tenant-aware few-shot examples** — the current single hardcoded S3 example is shown to GCP and Terraform tenants too. One example per tenant would be more representative.
