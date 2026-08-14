# CertCoach — Design Spec

> Multi-tenant RAG coaching assistant for cloud certifications, with a first-class
> evaluation harness. Built as a portfolio project to demonstrate production-grade
> LLM/RAG engineering (retrieval, eval, multi-tenant isolation, model-agnostic design).

**Status:** design approved, ready for implementation planning.
**Time budget:** ~2 weeks.
**Author/owner:** Sai Mun Lee (senior SWE, ML platform background).

> **Revision 2026-07-15:** retrieval upgraded from pure-dense to **hybrid (dense + sparse) with
> RRF fusion** plus a **retrieval confidence gate**, and eval axes/metrics extended (NDCG,
> hybrid A/B, no-answer rate, diagnostic separation). Changed lines are marked `[rev 2026-07-15]`.
> Rationale in the retrieval-review notes; core scope and 2-week budget unchanged.

> **Revision 2026-07-23:** added **per-stage latency metrics** to the eval harness (§6), a
> logged decision to **defer query rewriting / HyDE** (§8), and a **data-sourcing & licensing
> decision** (§4, §8) backed by a checked-in `ingestion/sources.yaml` provenance manifest — after
> reviewing a RAG design case study against this design. Changed lines are marked `[rev 2026-07-23]`.
> Scope and budget unchanged.

---

## 1. Why this project exists

The owner is a strong senior/ML-platform engineer whose *production* depth is tabular
foundation-model serving + recommender systems. LLM/RAG is currently prototyping-level.
Target roles (e.g. BCG U AI-skilling architect) treat **RAG, prompt engineering, vector
stores, and LLM evaluation as the core of the job**. This project converts that gap into a
shipped, measured artifact — and leans on the owner's genuine strength (multi-tenant
platform engineering) so it's defensible in a technical deep-dive.

**Honesty constraint (important):** everything here must be genuinely built and defensible.
Do not overstate. The résumé line only gets written once the thing actually runs and the
eval numbers are real. See the résumé repo's memory note "honest-optimized" for the owner's
standing preference.

## 2. Product in one paragraph

A learner picks a certification (tenant) — **AWS Solutions Architect – Associate** or
**GCP Associate Cloud Engineer** — and can: (a) **ask** a question and get an answer
**grounded in the official cert corpus with citations**; (b) request a **quiz** (generated
practice questions); (c) submit an answer and get it **graded with an explanation of the
gap**. Every retrieval and answer decision is measured by an eval harness.

## 3. Scope

- **Two tenants:** AWS SA-Associate + GCP ACE. Terraform Associate is a stretch third tenant.
- **Multi-tenant from day one:** each cert is an isolated corpus/namespace; retrieval is
  filtered by tenant and this isolation is tested.
- **Non-goals (YAGNI):** user accounts/auth (tenant passed via header), fancy UI, fine-tuning,
  response streaming, multi-node infra.

## 4. Architecture (isolated components, one job each)

1. **Ingestion (offline script)** — fetch official sources (AWS SAA-C03 exam guide + service
   FAQs + Well-Architected Framework; GCP ACE exam guide + core service docs + Architecture
   Framework) → clean (**normalize to Markdown** [rev 2026-07-23]) → structure-aware chunking
   (~500–800 tokens, overlap) → embed → store in pgvector with `tenant` + `source` (citation)
   metadata. **[rev 2026-07-15]** Also (a) build a Postgres full-text (`tsvector`) index over
   chunk text so sparse/BM25 retrieval lives in the *same* store, and (b) store `parent_doc_id`,
   neighbor-chunk ids, and `source_version`/`source_date` per chunk (enables "retrieve-small,
   expand-to-context" and version-aware citations). One-off boundary/table-fidelity spot check
   validates chunk quality before eval. **[rev 2026-07-23]** Sources are **first-party vendor
   docs only**, enumerated in a checked-in `ingestion/sources.yaml` provenance manifest
   (url · type · citation · version); raw docs are fetched into gitignored `data/`, never
   committed. No third-party exam dumps — see §8.
2. **Retrieval service** **[rev 2026-07-15 — was: pure-dense top-20 → rerank]** —
   `(query, tenant)` → **hybrid candidate generation**: dense vector search (pgvector) **and**
   sparse full-text search (Postgres `tsvector`/BM25) run in parallel, both filtered by
   `tenant` (top ~20 each) → **fuse with Reciprocal Rank Fusion (RRF)** (rank-only, no training
   data) → **production default: `hybrid_no_rerank`** (RRF score normalised to 0–1 confidence,
   ~10ms latency) → top ~5. **Optional `hybrid` mode adds `cohere-rerank-pro`** (cross-encoder
   reranking, ~2400ms) — available and measured in the eval A/B table, but not the serving
   default at this corpus scale. **Confidence gate (rerank mode only):** if the top reranker
   score is below a threshold, return a grounded "not enough context" answer instead of
   generating — a hallucination guardrail that also yields a no-answer-rate metric. Tenant
   isolation enforced at the query boundary.
   *Why hybrid: cloud-cert content is keyword-dense (service names, CIDR blocks, IAM actions like
   `s3:GetObject`); dense embeddings systematically miss exact-token matches, which sparse covers.
   No second system needed — both indexes live in Postgres.*
3. **Model-agnostic LLM client** — a single `generate(messages, model)` interface wrapping
   the `ai-core` gateway (OpenAI-compatible). Claude / GPT-5 / Gemini swappable by config.
   → satisfies "model-agnostic abstraction layer."
4. **Coaching layer** — versioned prompt templates for three modes: `answer` (grounded +
   citations), `quiz` (generate practice Qs), `grade` (score learner answer + explain gap).
5. **API (FastAPI)** — `/ask`, `/quiz`, `/grade`, `/health`; tenant via header.
6. **⭐ Eval harness (centerpiece)** — see §6.

## 5. Stack & model roles

| Concern | Choice | Notes |
|---|---|---|
| Language / API | Python >=3.11, FastAPI, Pydantic | owner's stack |
| Vector store | **Postgres + pgvector** (Docker) | runs local, deployable, credible; **[rev 2026-07-15]** also provides `tsvector` full-text search → hybrid (dense+sparse) retrieval in one store |
| Embeddings | `text-embedding-3-large` (primary), `gemini-embedding` (A/B) | both ~\$0.0001 |
| Rerank | `cohere-rerank-pro` | two-stage retrieval; an eval axis |
| Generation | default `claude-sonnet-4-6`; swappable `gpt-5` / `gemini-2.5-pro`; `claude-haiku-4-5` for bulk quiz-gen | cost/quality balance |
| Eval judge | cross-family (e.g. `gpt-5` or `o3` judging Claude answers) | reduces self-preference bias |
| Tests | pytest, TDD where it fits | back up the "TDD" résumé claim |

**Model gateway:** `ai-core` (OpenAI-compatible, the owner's SAP Generative AI Hub tenant).
Wrap it behind our own client rather than the stock Anthropic SDK. When building the client,
consult the `claude-api` skill for params/limits, but note model IDs here are gateway aliases,
not standard Anthropic IDs. Available models (with per-1K pricing where known):

- Chat: `claude-haiku-4-5-20251001` (0.00079/0.00367), `claude-sonnet-4-6` (0.00223/0.01087),
  `claude-opus-4-6` (0.00367/0.01806), `claude-opus-4-8` (pricing null — may not be GA on tenant),
  `gpt-5` (0.00091/0.00677), `gpt-5-mini` (0.00019/0.00136), `gpt-5-nano` (0.00004/0.00028),
  `gpt-5.5` (0.00342/0.02015), `gemini-2.5-pro` (0.00087/0.00647), `o3` (0.0061/0.02436),
  `qwen3.6-plus` (null), `sonar-pro` (0.00243/0.01185).
- Embeddings: `text-embedding-3-large` (0.00009), `gemini-embedding` (0.00011).
- Rerank: `cohere-rerank-pro`.

## 6. Eval harness (the differentiator)

**Retrieval eval** — gold set of `question → relevant chunk/doc` **[rev 2026-07-15]** *(50–100
hand-verified pairs; provenance = public practice Qs + official exam-guide objectives — state the
size and provenance honestly in the README rather than implying a large set)*:
- Metrics: recall@k, MRR, hit-rate, **NDCG@10** (retrieval) and **NDCG@5** (rerank, before/after).
- A/B axes: `text-embedding-3-large` vs `gemini-embedding`; **dense-only vs sparse-only vs
  hybrid (RRF)** **[rev 2026-07-15]**; rerank on vs off; chunk size.
- **Diagnostic separation [rev 2026-07-15]:** distinguish *corpus gaps* (answer isn't in the
  corpus) from *retrieval failures* (it is, but wasn't retrieved) so a regression is actionable.

**Answer eval** — gold Q&A drawn from public practice questions (cloud certs have near-objective
answers — this is why the domain was chosen):
- **Accuracy** via LLM-as-judge, judge from a *different* model family than the generator.
- **Groundedness / citation-faithfulness** — do the answer's claims trace to retrieved chunks?
- **No-answer rate & hallucination rate [rev 2026-07-15]** — with the confidence gate on, how
  often do we correctly abstain vs. answer ungrounded?

**Latency eval [rev 2026-07-23]** — alongside quality, record **per-stage latency** (dense, sparse,
RRF, rerank, generate) and **end-to-end P50/P95** over the gold set, so every quality A/B (e.g.
rerank on/off) also shows its latency cost. This is a small-sample portfolio measurement — report
medians over the gold set and label it as such, not a load test.

**Output:** a `make eval` (or `python -m certcoach.eval`) command that emits a metrics table,
diffable across configs. Week 2 should show a **before/after prompt-iteration improvement** in
these numbers — that story is the headline.

## 7. Milestones (~2 weeks)

**Week 1**
- Days 1–2: repo scaffold, Docker (Postgres+pgvector), model-agnostic client against `ai-core`,
  `/health` + smoke "ask" end-to-end on one doc.
- Days 3–4: AWS ingestion (chunk/embed/store + citations + full-text index), **hybrid retrieval
  (dense+sparse → RRF) + rerank + confidence gate** [rev 2026-07-15], grounded `/ask`.
- Day 5: GCP tenant ingested; tenant isolation enforced + tested.

**Week 2**
- Days 6–7: eval harness + gold sets (retrieval + Q&A); first metrics report.
- Days 8–9: `quiz` + `grade` modes; prompt iteration shown as before/after eval gains.
- Day 10: README with architecture diagram + eval results; `docker compose up`; optional
  Streamlit demo + screenshot.
- Buffer / stretch: multimodal (ElevenLabs TTS on answers); Terraform tenant; Cloud Foundry deploy.

## 8. Decisions taken (revisit freely)

- **Demo UI:** minimal Streamlit chat, Week-2 nicety only (after eval is done). Fallback: API + CLI.
- **Deploy:** `docker compose up` + documented run is sufficient for the portfolio; Cloud Foundry
  push is a stretch. (No Azure — the owner has no access; do **not** claim Azure anywhere.)
- **Data sourcing & licensing [rev 2026-07-23]:** corpus = **first-party, freely-published vendor
  docs only** (official exam guides, service FAQs, Well-Architected / Architecture Framework, core
  service docs), enumerated in `ingestion/sources.yaml`. **No exam dumps / brain-dumps / third-party
  crammer content** (copyright + certification NDA + indefensible in a deep-dive). Vendor docs are
  copyrighted but freely accessible → non-commercial portfolio use is fine, but the raw corpus is
  **fetched at ingestion into gitignored `data/`, never committed or redistributed**; the checked-in
  manifest plus per-chunk `source_version`/`source_date` carry provenance. Eval gold set = official
  sample questions + **hand-authored** pairs from exam-guide objectives (first-party only, same rule).
- **Multimodal:** clearly-labeled stretch, not core.
- **Query rewriting / HyDE:** considered, **deferred [rev 2026-07-23].** CertCoach is single-turn
  (no conversation history to resolve) and its keyword-dense exact-match gap is already covered by
  the hybrid **sparse/BM25** leg — the same gap HyDE is usually reached for. HyDE would add a
  hot-path LLM call and risks injecting hallucinated service names into the query embedding, which
  hurts exact-match retrieval. Revisit only if eval **diagnostic separation** shows a
  retrieval-failure cluster on free-form `/ask` questions — then query rewriting (not HyDE) is the
  first lever to try.

## 9. Résumé payoff (only once real)

- A live, measured, multi-tenant RAG system → a legitimate **Selected Projects** entry.
- Upgrades the Skills "LLM/RAG (prototyping)" line to shipped.
- Interview story spanning retrieval, evaluation, multi-tenant isolation, and model-agnostic design.

## 10. Suggested first step in the new session

Invoke the `superpowers:writing-plans` skill to turn this spec into a phased implementation
plan, then build Week-1 Day-1 (scaffold + Docker + model client + smoke test) test-first.
