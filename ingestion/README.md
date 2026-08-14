# Ingestion — fetch stage

The **fetch stage** of the CertCoach ingestion pipeline. It reads the source
manifest (`sources.yaml`), downloads each raw document to
`data/<tenant>/<id>.<ext>`, and writes a provenance index to
`data/fetch_index.json`. Cleaning, Markdown normalization, chunking, and
embedding are **later stages, not built here**.

## Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) for the virtualenv and
dependency install:

```bash
uv venv                       # create .venv (Python >=3.11)
uv pip install -e ".[dev]"    # install certcoach + dev deps (pytest)
```

## Run

Invoke the fetcher through the venv interpreter directly (keeps output pristine —
`uv run` prints a benign VIRTUAL_ENV mismatch warning under a pyenv shell):

```bash
.venv/bin/python -m ingestion.fetch                    # all tenants
.venv/bin/python -m ingestion.fetch --tenant aws-saa   # one tenant
.venv/bin/python -m ingestion.fetch --delay 1.0        # override the default 0.5s politeness delay
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--manifest` | `ingestion/sources.yaml` | Source manifest to read |
| `--data-dir` | `data` | Output root for fetched docs + index |
| `--tenant` | *(all)* | Fetch only this tenant; repeatable |
| `--delay` | `0.5` | Seconds to sleep between requests |

Exit code is `0` only if **every** document fetched OK; non-zero if any record
failed (so it composes in CI / `make` pipelines).

## Provenance index

`data/fetch_index.json` is the authoritative record of what was fetched. Each
entry carries `tenant`, `id`, `url`, `type`, `format`, `corpus`, `citation`,
`source_version`, `fetched_date`, `http_status`, `content_type`, `bytes`,
`sha256`, `ok`, and a `note`. The index is merged by tenant across runs, so
re-fetching one tenant does not drop the others.

## Notes

- **First-party sources only** — official AWS / Google Cloud docs. No exam
  dumps, brain-dumps, or third-party crammer content (see `docs/spec.md` §8).
- `data/` is **gitignored**; fetched copyrighted vendor docs are never committed
  or redistributed. Only this README, the code, and `sources.yaml` are tracked.
- **Only successful fetches write bytes to disk** (`ok: true`). A failed or
  thin/JS-shell response is still recorded in `fetch_index.json` with
  `ok: false` and a note — failures are surfaced, never hidden.
- **Known expected failure:** the GCP *Associate Cloud Engineer Sample
  Questions* source is a Google Form whose questions are JavaScript-rendered, so
  a plain HTTP fetch returns only a thin shell and is recorded `ok: false`
  ("thin content - possible JS shell"). It is an **eval-only** source
  (`corpus: false`), so this does not affect the retrieval corpus.
- **Last verified run (2026-07-24):** 11/12 OK — all corpus documents fetched;
  the single failure was the eval-only Google Form noted above.
- **Next stage (not built yet):** clean → normalize to **Markdown** → chunk →
  embed → store (pgvector + FTS).
