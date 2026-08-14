# Phoenix Tracing Design

**Date:** 2026-08-09
**Status:** Approved

## Goal

Add Arize Phoenix tracing to CertCoach so that every LLM call, retrieval pipeline, and HTTP request is observable in a persistent, queryable UI — useful for both development debugging and correlating eval results with trace data.

## Architecture

Phoenix runs as a Docker container alongside the existing Postgres container. `docker compose up` starts both. The app sends OpenTelemetry spans via OTLP gRPC to Phoenix's collector endpoint.

```
HTTP Request
  └── FastAPI middleware (root span)
        ├── retrieve() span
        │     ├── embed() [auto — openinference]
        │     ├── dense_search()
        │     ├── sparse_search()
        │     └── rerank()
        └── chat.completions.create() [auto — openinference]

eval_run span (prompt_version=N)
  ├── retrieve() span × 20
  └── chat.completions.create() × 20 (judge + answer)
```

Phoenix endpoints:
- **UI:** `http://localhost:6006`
- **OTLP gRPC collector:** `http://localhost:4317`

## Instrumentation strategy

- **Auto-instrumentation via `openinference-instrumentation-openai`:** patches the `openai.OpenAI` client at startup. Captures all `chat.completions.create` and `embeddings.create` calls automatically — model, token counts, full prompt/response payloads. Zero changes to `generate()` or `embed()` call sites.
- **Manual `retrieve` span** in `retrieval.py`: wraps the `retrieve()` function body, capturing `query`, `tenant`, `retrieval_mode`, `confidence`, `chunk_count`, `below_threshold` as span attributes. Also adds a manual child span for `rerank()` since it calls the internal model gateway directly (not via the OpenAI client) and is therefore not auto-instrumented.
- **FastAPI OTLP middleware** in `api.py`: makes each HTTP request the root span, so child spans (retrieval, LLM) are nested under it in the trace view.
- **Eval run span** in `eval.py`: wraps `run_eval()` with a top-level span tagged with `prompt_version` and `judge_model`, so eval runs are filterable in Phoenix.

## Graceful degradation

`configure_tracing()` (in `certcoach/tracing.py`) is a no-op if `PHOENIX_COLLECTOR_ENDPOINT` is not set. The app and tests work normally without Phoenix running. CI does not require Phoenix.

## Components

| File | Change |
|------|--------|
| `docker-compose.yml` | Add `phoenix` service (image: `arizephoenix/phoenix`, port 6006/4317, named volume for SQLite persistence) |
| `certcoach/tracing.py` | New: `configure_tracing()`, `retrieval_span()` context manager |
| `certcoach/api.py` | Call `configure_tracing()` at app startup; add `OpenTelemetryMiddleware` |
| `certcoach/retrieval.py` | Wrap `retrieve()` body with `retrieval_span()` |
| `certcoach/eval.py` | Wrap `run_eval()` with an `eval_run` OTel span |
| `pyproject.toml` | Add deps: `arize-phoenix-otel`, `openinference-instrumentation-openai`, `opentelemetry-instrumentation-fastapi` |
| `.env.example` | Add `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317` |

## What you see in Phoenix

- **Per-request trace tree:** query text → chunks retrieved → rerank scores → LLM prompt/response → final answer, all in one waterfall view.
- **Eval run grouping:** filter by `eval_run` span + `prompt_version` attribute to compare v1 vs v2 prompt traces side-by-side.
- **Persistent store:** Phoenix uses SQLite by default, mounted as a Docker volume — traces survive container restarts.

## Out of scope

- Streaming responses (not used in this project)
- Authentication for the Phoenix UI (local dev only)
- Sending traces to a remote Phoenix Cloud instance
