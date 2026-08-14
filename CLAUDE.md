# CertCoach

Multi-tenant **RAG coaching assistant for cloud certifications** with a first-class
**evaluation harness**. Portfolio project to demonstrate production-grade LLM/RAG
engineering. **Read `docs/spec.md` first — it is the full, approved design.**

## Context you need

- **Owner:** Sai Mun Lee — senior software/ML-platform engineer. Production depth is tabular
  foundation-model serving + recommender systems; LLM/RAG is currently prototyping-level. This
  project exists to make LLM/RAG depth real for AI-architect roles (e.g. BCG U).
- **Honesty constraint:** everything must be genuinely built and defensible in a technical
  deep-dive. Never overstate capability. The résumé line gets written only once it runs and the
  eval numbers are real.

## Scope (see spec for detail)

- Two tenants: **AWS SA-Associate** + **GCP ACE** (Terraform = stretch). Multi-tenant from day one.
- ~2-week budget. Non-goals: auth/accounts, fancy UI, fine-tuning, streaming, multi-node.

## Stack

- Python 3.11, FastAPI, Pydantic. Postgres + **pgvector** (Docker). pytest; TDD where it fits.
- Embeddings: `text-embedding-3-large` (primary), `gemini-embedding` (A/B). Rerank: `cohere-rerank-pro`.
- Generation default `claude-sonnet-4-6`, swappable to `gpt-5` / `gemini-2.5-pro` via a
  **model-agnostic client**. Eval judge = a different model family than the generator.

## Model gateway

- All models are served by **`ai-core`** (OpenAI-compatible, the owner's SAP Generative AI Hub).
  Wrap it behind our own client; model IDs are gateway aliases, not standard Anthropic IDs.
  Consult the `claude-api` skill when building the client. Full model list + pricing in `docs/spec.md`.
- **No Azure access** — do not use or claim Azure.

## How to start

Invoke `superpowers:writing-plans` to turn `docs/spec.md` into a phased implementation plan,
then build Week-1 Day-1 (scaffold + Docker + model client + smoke test) test-first.
