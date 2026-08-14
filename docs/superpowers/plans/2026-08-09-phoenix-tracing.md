# Phoenix Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Arize Phoenix tracing to CertCoach so every LLM call, retrieval pipeline step, and HTTP request produces a persistent, queryable trace in the Phoenix UI.

**Architecture:** Phoenix runs as a Docker container alongside Postgres; the app sends OTLP gRPC spans to it. The OpenAI client is auto-instrumented via `openinference-instrumentation-openai`; retrieval and eval get manual spans. Tracing is a no-op when `PHOENIX_COLLECTOR_ENDPOINT` is unset so tests and CI are unaffected.

**Tech Stack:** `arize-phoenix-otel`, `openinference-instrumentation-openai`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-sdk`, Docker (`arizephoenix/phoenix` image).

## Global Constraints

- Python 3.11+
- All new deps added to `pyproject.toml` `[project.dependencies]` (not dev-only)
- Tracing must be a no-op when `PHOENIX_COLLECTOR_ENDPOINT` env var is unset
- Do not modify existing test mocking patterns — tests patch `certcoach.retrieval.embed` and `certcoach.retrieval.rerank` by name; keep those import paths intact
- Phoenix UI: `http://localhost:6006`, OTLP gRPC collector: `http://localhost:4317`
- All span attribute names use snake_case strings (e.g. `"certcoach.tenant"`, `"certcoach.retrieval_mode"`)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `certcoach/tracing.py` | Create | `configure_tracing()`, `retrieval_span()` context manager, `rerank_span()` context manager |
| `certcoach/api.py` | Modify | Call `configure_tracing()` at module level; add `OpenTelemetryMiddleware` |
| `certcoach/retrieval.py` | Modify | Wrap `retrieve()` body with `retrieval_span()`; wrap `rerank()` call with `rerank_span()` |
| `certcoach/eval.py` | Modify | Wrap `run_eval()` with a top-level `eval_run` OTel span |
| `docker-compose.yml` | Modify | Add `phoenix` service with persistent SQLite volume |
| `pyproject.toml` | Modify | Add tracing dependencies |
| `.env.example` | Modify | Add `PHOENIX_COLLECTOR_ENDPOINT` |
| `tests/test_tracing.py` | Create | Unit tests for `configure_tracing()` and `retrieval_span()` |

---

### Task 1: Dependencies + Docker

**Files:**
- Modify: `pyproject.toml`
- Modify: `docker-compose.yml`
- Modify: `.env.example`

**Interfaces:**
- Produces: `PHOENIX_COLLECTOR_ENDPOINT` env var convention used by all later tasks

- [ ] **Step 1: Add tracing deps to pyproject.toml**

In `pyproject.toml`, add to `dependencies`:

```toml
dependencies = [
    # ... existing deps ...
    "arize-phoenix-otel>=0.6",
    "openinference-instrumentation-openai>=0.1",
    "opentelemetry-instrumentation-fastapi>=0.45b0",
    "opentelemetry-sdk>=1.24",
    "opentelemetry-exporter-otlp-proto-grpc>=1.24",
]
```

- [ ] **Step 2: Add Phoenix service to docker-compose.yml**

```yaml
services:
  db:
    # ... existing db service unchanged ...

  phoenix:
    image: arizephoenix/phoenix:latest
    container_name: certcoach-phoenix
    ports:
      - "6006:6006"   # UI
      - "4317:4317"   # OTLP gRPC
    volumes:
      - phoenixdata:/data
    environment:
      PHOENIX_WORKING_DIR: /data

volumes:
  pgdata:
  phoenixdata:
```

- [ ] **Step 3: Add env var to .env.example**

Append to `.env.example`:

```
# Arize Phoenix tracing (optional — omit to disable tracing)
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
```

- [ ] **Step 4: Install updated deps**

```bash
pip install -e ".[dev]"
```

Expected: installs without errors. `python -c "import openinference.instrumentation.openai"` succeeds.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example
git commit -m "feat: add Phoenix tracing deps and Docker service"
```

---

### Task 2: `certcoach/tracing.py` — core tracing module

**Files:**
- Create: `certcoach/tracing.py`
- Create: `tests/test_tracing.py`

**Interfaces:**
- Produces:
  - `configure_tracing() -> None` — call once at app/eval startup; no-op if `PHOENIX_COLLECTOR_ENDPOINT` unset
  - `retrieval_span(query: str, tenant: str, retrieval_mode: str) -> contextmanager` — yields an OTel span; caller sets `confidence`, `chunk_count`, `below_threshold` attributes on it after retrieval completes
  - `rerank_span(query: str, n_candidates: int) -> contextmanager` — yields an OTel span for the rerank call

- [ ] **Step 1: Write failing tests**

Create `tests/test_tracing.py`:

```python
import os
import pytest
from unittest.mock import patch, MagicMock


def test_configure_tracing_noop_when_no_endpoint(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    # Should not raise, should not attempt any network connection
    from certcoach.tracing import configure_tracing
    configure_tracing()  # no-op


def test_configure_tracing_registers_instrumentor_when_endpoint_set(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    mock_instrumentor = MagicMock()
    mock_instrumentor_class = MagicMock(return_value=mock_instrumentor)

    with patch("certcoach.tracing.OpenAIInstrumentor", mock_instrumentor_class), \
         patch("certcoach.tracing._setup_otel_provider"):
        from certcoach import tracing
        # Force re-execution by resetting the cached state
        tracing._tracing_configured = False
        tracing.configure_tracing()

    mock_instrumentor.instrument.assert_called_once()


def test_retrieval_span_returns_context_manager():
    from certcoach.tracing import retrieval_span
    with retrieval_span("what is EC2?", "aws-saa", "hybrid") as span:
        assert span is not None


def test_rerank_span_returns_context_manager():
    from certcoach.tracing import rerank_span
    with rerank_span("what is EC2?", n_candidates=20) as span:
        assert span is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tracing.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `certcoach.tracing` doesn't exist yet.

- [ ] **Step 3: Implement `certcoach/tracing.py`**

```python
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracing_configured = False
_tracer_name = "certcoach"


def _setup_otel_provider(endpoint: str) -> None:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def configure_tracing() -> None:
    global _tracing_configured
    if _tracing_configured:
        return
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return
    from openinference.instrumentation.openai import OpenAIInstrumentor
    _setup_otel_provider(endpoint)
    OpenAIInstrumentor().instrument()
    _tracing_configured = True


def _tracer() -> trace.Tracer:
    return trace.get_tracer(_tracer_name)


@contextmanager
def retrieval_span(
    query: str,
    tenant: str,
    retrieval_mode: str,
) -> Generator[trace.Span, None, None]:
    with _tracer().start_as_current_span("certcoach.retrieve") as span:
        span.set_attribute("certcoach.query", query)
        span.set_attribute("certcoach.tenant", tenant)
        span.set_attribute("certcoach.retrieval_mode", retrieval_mode)
        yield span


@contextmanager
def rerank_span(
    query: str,
    n_candidates: int,
) -> Generator[trace.Span, None, None]:
    with _tracer().start_as_current_span("certcoach.rerank") as span:
        span.set_attribute("certcoach.query", query)
        span.set_attribute("certcoach.rerank.n_candidates", n_candidates)
        yield span
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tracing.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add certcoach/tracing.py tests/test_tracing.py
git commit -m "feat: add certcoach/tracing.py with configure_tracing and span helpers"
```

---

### Task 3: Wire tracing into `certcoach/retrieval.py`

**Files:**
- Modify: `certcoach/retrieval.py`

**Interfaces:**
- Consumes: `retrieval_span(query, tenant, retrieval_mode)` and `rerank_span(query, n_candidates)` from `certcoach.tracing`
- Produces: `retrieve()` now emits a `certcoach.retrieve` parent span with `certcoach.rerank` child span when tracing is active

- [ ] **Step 1: Add span wrapping to `retrieve()`**

At the top of `certcoach/retrieval.py`, add the import:

```python
from certcoach.tracing import retrieval_span, rerank_span
```

Then in `retrieve()`, wrap the function body. The current body starts after the docstring. Replace:

```python
    # 1. Embed query (skip if pre-computed embedding is provided)
    if query_embedding is None:
        query_embedding = embed([query])[0]
    ...
    return RetrievalResult(
        chunks=expanded,
        confidence=confidence,
        below_threshold=below_threshold,
        query=query,
        tenant=tenant,
    )
```

With:

```python
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

        if retrieval_mode == "hybrid_no_rerank":
            top_chunks = fused[:rerank_top_n]
            confidence = top_chunks[0].score if top_chunks else 0.0
            expanded = expand_chunks(conn, top_chunks)
            span.set_attribute("certcoach.chunk_count", len(expanded))
            span.set_attribute("certcoach.confidence", confidence)
            span.set_attribute("certcoach.below_threshold", False)
            return RetrievalResult(
                chunks=expanded, confidence=confidence,
                below_threshold=False, query=query, tenant=tenant,
            )

        # 5. Rerank
        candidates = fused[: rerank_top_n * 4]
        with rerank_span(query, n_candidates=len(candidates)):
            rerank_results = rerank(query, [c.content for c in candidates], top_n=rerank_top_n)
        rerank_results_sorted = sorted(rerank_results, key=lambda r: r.score, reverse=True)

        confidence = rerank_results_sorted[0].score if rerank_results_sorted else 0.0
        below_threshold = confidence < confidence_threshold

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
```

- [ ] **Step 2: Run existing retrieval tests to verify nothing broke**

```bash
pytest tests/test_retrieval.py -v
```

Expected: all tests pass (tracing is no-op since `PHOENIX_COLLECTOR_ENDPOINT` not set in test env).

- [ ] **Step 3: Commit**

```bash
git add certcoach/retrieval.py
git commit -m "feat: instrument retrieval pipeline with Phoenix spans"
```

---

### Task 4: Wire tracing into `certcoach/api.py`

**Files:**
- Modify: `certcoach/api.py`

**Interfaces:**
- Consumes: `configure_tracing()` from `certcoach.tracing`
- Produces: FastAPI requests produce a root span that parents retrieval + LLM child spans

- [ ] **Step 1: Add `configure_tracing()` call and FastAPI middleware**

At the top of `certcoach/api.py`, add:

```python
from certcoach.tracing import configure_tracing

configure_tracing()
```

Then, after `app = FastAPI(title="CertCoach", version="0.1.0")`, add the middleware:

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)
```

The full top of the file should look like:

```python
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from certcoach.db import get_conn, apply_migrations
from certcoach.retrieval import retrieve
from certcoach.generate import answer_question, generate_quiz, grade_answer
from certcoach.tracing import configure_tracing

configure_tracing()

app = FastAPI(title="CertCoach", version="0.1.0")

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)
```

- [ ] **Step 2: Run existing API tests to verify nothing broke**

```bash
pytest tests/test_api.py -v
```

Expected: all tests pass. The `configure_tracing()` call is a no-op in tests since `PHOENIX_COLLECTOR_ENDPOINT` is not set.

- [ ] **Step 3: Commit**

```bash
git add certcoach/api.py
git commit -m "feat: wire Phoenix tracing into FastAPI app"
```

---

### Task 5: Wire eval run span into `certcoach/eval.py`

**Files:**
- Modify: `certcoach/eval.py`

**Interfaces:**
- Consumes: `configure_tracing()` from `certcoach.tracing`; standard `opentelemetry.trace` for the eval span
- Produces: `run_eval()` emits a top-level `certcoach.eval_run` span tagged with `prompt_version` and `judge_model`

- [ ] **Step 1: Add configure_tracing call and eval_run span to `run_eval()`**

At the top of `certcoach/eval.py`, add imports:

```python
from opentelemetry import trace as otel_trace
from certcoach.tracing import configure_tracing
```

At the start of `run_eval()`, add:

```python
def run_eval(...) -> dict:
    configure_tracing()
    tracer = otel_trace.get_tracer("certcoach")
    with tracer.start_as_current_span("certcoach.eval_run") as span:
        span.set_attribute("certcoach.prompt_version", prompt_version)
        span.set_attribute("certcoach.judge_model", judge_model)

        # ... rest of existing run_eval body indented inside this with block ...

        output_path_obj.write_text(json.dumps(output, indent=2))
        return output
```

The entire body of `run_eval()` (from `gold_ret = _load_jsonl(...)` through `return output`) must be indented one level inside the `with tracer.start_as_current_span(...)` block.

- [ ] **Step 2: Run existing eval tests to verify nothing broke**

```bash
pytest tests/test_eval.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add certcoach/eval.py
git commit -m "feat: wrap run_eval with eval_run trace span"
```

---

### Task 6: Smoke test with live Phoenix

**Files:** none (verification only)

- [ ] **Step 1: Start the stack**

```bash
docker compose up -d
```

Expected: both `certcoach-db` and `certcoach-phoenix` containers start. Check with `docker compose ps`.

- [ ] **Step 2: Set env var and start the API**

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
uvicorn certcoach.api:app --reload
```

- [ ] **Step 3: Send a test request**

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Tenant: aws-saa" \
  -d '{"question": "What is EC2?"}'
```

- [ ] **Step 4: Verify trace in Phoenix UI**

Open `http://localhost:6006` in a browser. You should see:
- A root span for the `POST /ask` request
- A child `certcoach.retrieve` span with attributes `certcoach.tenant=aws-saa`, `certcoach.retrieval_mode=hybrid`
- A child `certcoach.rerank` span
- An auto-instrumented `openai.chat` span with token counts and the prompt/response payload

- [ ] **Step 5: Commit**

```bash
git add .  # no file changes expected; this step is verification only
git commit --allow-empty -m "chore: verify Phoenix tracing smoke test passes"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Phoenix Docker container, persistent SQLite | Task 1 |
| `configure_tracing()` no-op when env unset | Task 2 |
| Auto-instrument OpenAI client | Task 2 (`OpenAIInstrumentor`) |
| `retrieval_span()` with query/tenant/mode/confidence/chunk_count/below_threshold | Task 3 |
| `rerank_span()` (manual, since rerank bypasses OpenAI client) | Task 3 |
| FastAPI root span | Task 4 |
| `eval_run` span with prompt_version + judge_model | Task 5 |
| `PHOENIX_COLLECTOR_ENDPOINT` env var in `.env.example` | Task 1 |

All spec requirements covered. No gaps found.
