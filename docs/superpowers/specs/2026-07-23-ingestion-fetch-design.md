# CertCoach Ingestion — Raw Source Fetcher (Design)

**Date:** 2026-07-23
**Status:** approved by owner (design + written spec); implementation plan next.
**Scope:** the **fetch stage only** of the ingestion pipeline (spec.md §4 item 1, first arrow).

---

## 1. Context

CertCoach ingestion (`docs/spec.md` §4 item 1, `docs/architecture.md` §2) is a multi-stage
offline pipeline: **fetch → clean → chunk → embed → store** (pgvector + `tsvector` FTS +
metadata). This design covers **only the first stage**: pulling raw source documents to local
disk with provenance, for both tenants (`aws-saa`, `gcp-ace`), driven by the checked-in
`ingestion/sources.yaml` manifest.

The repo is otherwise docs-only, so this increment also introduces a **minimal** Python
scaffold — just enough to run and test the fetcher, not the full project foundation (no Docker,
no DB, no model client yet).

## 2. Goal / non-goals

- **Goal:** given `ingestion/sources.yaml`, download every document's raw content to
  `data/<tenant>/<id>.<ext>` (gitignored) and write a `data/fetch_index.json` provenance record.
  Fetch **everything** in the manifest — both `corpus: true` (RAG corpus) and `corpus: false`
  (official sample questions) — preserving the flag in the index.
- **Non-goals (deferred to later stages):** parsing/cleaning, chunking, embeddings, pgvector,
  FTS, the API, auth, and **Playwright/JS rendering** (deferred; see §6).

> **Clean-stage target format (recorded for continuity — not built in this increment):** the later
> clean stage normalizes every fetched source — HTML **and** PDF — to **Markdown** before
> structure-aware chunking (structure-preserving, low-noise, LLM-native; PDF and plain text are worse
> to chunk from). This fetch increment still stores raw bytes as-fetched; the decision is logged here
> so it isn't re-litigated when the clean stage is built.

## 3. Data layout & provenance

```
data/                          # gitignored — raw copyrighted docs are never committed (spec §8)
  aws-saa/
    saa-exam-guide.pdf         # raw bytes, exactly as fetched
    faq-ec2.html
    ...
  gcp-ace/
    ...
  fetch_index.json             # provenance record for every document
```

Each `fetch_index.json` entry:
`tenant · id · url · type · format · corpus · citation · source_version · fetched_date (UTC ISO) ·
http_status · content_type · bytes · sha256 · ok · note`.

This gives reproducibility and the provenance the citation design wants, and **honestly records
anything that failed** (`ok: false` + `note`) instead of hiding it.

## 4. Components (testable core, thin I/O shell)

- **`ingestion/sources.py`** — Pydantic models `Manifest`, `TenantSpec`, `DocSpec`, and
  `load_manifest(path) -> Manifest` (parse + validate `sources.yaml`; apply `defaults`).
- **`ingestion/fetch.py`**
  - `target_path(data_dir, tenant, doc) -> Path` — derives `<tenant>/<id>.<ext>` from `format`.
  - `classify_response(doc, status, body) -> (ok, note)` — non-200 and thin-content detection.
  - `fetch_all(manifest, data_dir, fetch_fn, *, delay=...) -> list[FetchRecord]` — **pure core**;
    the HTTP call is an injected `fetch_fn(url) -> (status, headers, body)`.
  - `write_index(data_dir, records)` — serialize records to `fetch_index.json`.
  - `main()` — wires the real `httpx` client, parses CLI args, runs `fetch_all`, writes index.
- **`FetchRecord`** — Pydantic model serialized into the index.

Keeping `fetch_all` pure with an injected `fetch_fn` is what makes it testable without network.

## 5. Fetch strategy

Plain `httpx`: descriptive User-Agent, follow redirects, sane timeout, a small politeness delay
between requests. Save raw bytes as-is.

- Non-200 → `ok: false`, note = status.
- Format `html` whose text content (after a crude tag strip) is below a small threshold →
  `ok: false`, note = `"thin content — possible JS shell"`. This surfaces JS-rendered pages
  rather than silently storing an empty shell.
- No Playwright / headless browser (deferred — §6).

## 6. URL verification (in-scope task, owner-requested)

Several manifest URLs are `# TODO: verify` placeholders (notably the AWS SAA-C03 exam-guide PDF,
AWS sample-questions PDF, GCP ACE exam guide, GCP sample questions — filenames were stubbed).
Before the fetch is considered done, resolve each via `WebSearch`/`WebFetch` (Playwright MCP if a
page needs JS), update `ingestion/sources.yaml`, and **record honestly** any URL for which no
stable, first-party public source exists (rather than substituting a third-party one).

## 7. Testing (TDD)

Drive `fetch_all` with a **fake `fetch_fn`** returning canned responses; no network in tests:
- 200 HTML, 200 PDF bytes, 404, and thin-HTML cases.
- Assert: files written under a tmp data dir with correct names/extensions; `FetchRecord` fields
  (`sha256`, `bytes`, `content_type`, `ok`, `note`); `fetch_index.json` shape; `--tenant`
  scoping; and manifest validation errors from `load_manifest`.

## 8. Scaffold introduced

- `pyproject.toml` — project metadata; deps `httpx`, `pyyaml`, `pydantic`; dev `pytest`. Python 3.11+.
- `ingestion/__init__.py`, `ingestion/sources.py`, `ingestion/fetch.py`, `tests/test_fetch.py`.

## 9. Run

```
python -m ingestion.fetch [--tenant aws-saa] [--manifest ingestion/sources.yaml] [--data-dir data]
```
Idempotent: re-running overwrites artifacts and refreshes provenance. A `--tenant`-scoped run
updates only that tenant's entries in `fetch_index.json` (merge by `tenant` + `id`), leaving other
tenants' records intact — a scoped run must not wipe the full index.

## 10. Risks / honest notes

- **JS-rendered pages** (some GCP docs) may return thin shells over `httpx`; we **flag, not hide**.
  Playwright fallback is a follow-up only if the index shows real thin-content failures.
- **Stable public URLs** for exam-guide / sample-question PDFs can move or not exist publicly;
  recorded honestly in the index rather than faked.
- **Copyright:** fetched content is first-party vendor docs, kept in gitignored `data/`, never
  committed or redistributed (spec §8). No exam dumps.
