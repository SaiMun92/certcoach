# Docker + pgvector Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Postgres + pgvector Docker environment with the full CertCoach chunks schema, migration runner, and a Python `db` module that later ingestion and retrieval stages import.

**Architecture:** A `docker-compose.yml` brings up one Postgres 16 + pgvector container with a named volume. A plain SQL migration file creates the `chunks` table and all indexes (dense vector, tsvector FTS, tenant, source). A thin Python `certcoach/db.py` module wraps `psycopg2` connection management and exposes `get_conn()`. Tests use a live DB (no mocks — mocks masked a real prod bug in this project; always hit the real DB for integration tests). An `.env.example` documents the required env vars; a gitignored `.env` holds local values.

**Tech Stack:** Postgres 16 + pgvector 0.7, Docker Compose v2, psycopg2-binary 2.9, pgvector Python client 0.3, pytest 8, python-dotenv 1.0.

---

## Global Constraints

- Python ≥ 3.11.
- No mocks for DB integration tests — always use the real running Postgres container.
- `data/` and `pgdata/` are gitignored — never commit them.
- No Azure. No third-party exam-dump sources.
- `.env` is gitignored. `.env.example` is committed.
- All model IDs are `ai-core` gateway aliases, not standard Anthropic IDs — irrelevant to this plan but noted for later tasks.
- Embedding dimension for `text-embedding-3-large` is **3072**.
- Tenant values are exactly `"aws-saa"` and `"gcp-ace"` (from `sources.yaml`).

---

## File map

| File | Create / Modify | Responsibility |
|---|---|---|
| `docker-compose.yml` | Create | Postgres 16 + pgvector container + named volume |
| `.env.example` | Create | Documents `DATABASE_URL` and `POSTGRES_*` vars |
| `.env` | Create (gitignored) | Local values — never committed |
| `migrations/001_chunks.sql` | Create | DDL: `chunks` table + all indexes |
| `certcoach/__init__.py` | Create | Marks `certcoach` as a package |
| `certcoach/db.py` | Create | `get_conn()`, `apply_migrations()` |
| `tests/test_db.py` | Create | Integration tests against live container |
| `pyproject.toml` | Modify | Add `certcoach` package + new deps |

---

### Task 1: Docker Compose + env files

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.env` (gitignored, local only)

**Interfaces:**
- Consumes: nothing.
- Produces: a running `certcoach-db` Postgres container on port 5432 that later tasks connect to.

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: certcoach-db
    environment:
      POSTGRES_DB: certcoach
      POSTGRES_USER: certcoach
      POSTGRES_PASSWORD: certcoach
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U certcoach -d certcoach"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```

- [ ] **Step 2: Create `.env.example`**

```
# Copy to .env and fill in values for local development.
# DATABASE_URL is the only var certcoach code reads at runtime.
DATABASE_URL=postgresql://certcoach:certcoach@localhost:5432/certcoach

# These match docker-compose.yml defaults — change only if you override them there.
POSTGRES_DB=certcoach
POSTGRES_USER=certcoach
POSTGRES_PASSWORD=certcoach
```

- [ ] **Step 3: Create `.env`** (same content as `.env.example` — local defaults work as-is)

```
DATABASE_URL=postgresql://certcoach:certcoach@localhost:5432/certcoach
POSTGRES_DB=certcoach
POSTGRES_USER=certcoach
POSTGRES_PASSWORD=certcoach
```

- [ ] **Step 4: Start the container and verify it is healthy**

Run:
```bash
docker compose up -d
docker compose ps
```
Expected: `certcoach-db` shows `healthy` (may take ~10 s; re-run `docker compose ps` if still starting).

- [ ] **Step 5: Verify psql connects**

Run:
```bash
docker compose exec db psql -U certcoach -d certcoach -c "SELECT version();"
```
Expected: prints the Postgres version string.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "chore: docker-compose + pgvector container + env template"
```

---

### Task 2: Chunks DDL migration

**Files:**
- Create: `migrations/001_chunks.sql`

**Interfaces:**
- Consumes: running `certcoach-db` container from Task 1.
- Produces:
  - `chunks` table with columns: `id BIGSERIAL PRIMARY KEY`, `tenant TEXT NOT NULL`, `source_id TEXT NOT NULL`, `source_version TEXT`, `source_date DATE`, `citation TEXT NOT NULL`, `doc_type TEXT NOT NULL`, `parent_doc_id TEXT`, `prev_chunk_id BIGINT`, `next_chunk_id BIGINT`, `chunk_index INT NOT NULL`, `content TEXT NOT NULL`, `token_count INT`, `embedding vector(3072)`, `fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`, `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`.
  - Indexes: `ivfflat` on `embedding`, `GIN` on `fts`, B-tree on `(tenant, source_id)`, B-tree on `tenant`.
  - `schema_migrations` table for idempotent migration tracking.

- [ ] **Step 1: Create `migrations/001_chunks.sql`**

```sql
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
        embedding       vector(3072),                  -- text-embedding-3-large output
        fts             tsvector GENERATED ALWAYS AS   -- sparse BM25 index
                            (to_tsvector('english', content)) STORED,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- Dense vector index (IVFFlat; switch to HNSW once corpus > ~10k chunks)
    -- lists=100 is a reasonable default for corpora up to ~1M vectors
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
```

- [ ] **Step 2: Apply the migration manually to verify it runs cleanly**

Run:
```bash
docker compose exec -T db psql -U certcoach -d certcoach < migrations/001_chunks.sql
```
Expected: output ends with `NOTICE:  Migration 001 applied.` and no ERROR lines.

- [ ] **Step 3: Verify the table and indexes exist**

Run:
```bash
docker compose exec db psql -U certcoach -d certcoach -c "\d chunks"
docker compose exec db psql -U certcoach -d certcoach -c "\di chunks*"
```
Expected: `\d chunks` shows all columns including `fts` as a generated column and `embedding vector(3072)`; `\di` lists the four indexes.

- [ ] **Step 4: Verify the migration is idempotent (run it twice)**

Run:
```bash
docker compose exec -T db psql -U certcoach -d certcoach < migrations/001_chunks.sql
```
Expected: output says `NOTICE:  Migration 001 already applied, skipping.` — no error, no duplicate table.

- [ ] **Step 5: Commit**

```bash
git add migrations/001_chunks.sql
git commit -m "feat: chunks table DDL + vector/FTS/tenant indexes"
```

---

### Task 3: Python `certcoach` package + `db` module

**Files:**
- Create: `certcoach/__init__.py`
- Create: `certcoach/db.py`
- Modify: `pyproject.toml` (add `certcoach` package + new deps, add `python-dotenv`, `psycopg2-binary`, `pgvector`)

**Interfaces:**
- Consumes: `DATABASE_URL` env var (loaded from `.env` via `python-dotenv`); running `certcoach-db` container from Task 1.
- Produces:
  - `get_conn() -> psycopg2.extensions.connection` — returns an open connection using `DATABASE_URL`. Caller is responsible for closing (use as context manager or call `.close()`).
  - `apply_migrations(conn, migrations_dir: Path = Path("migrations")) -> None` — reads all `*.sql` files in `migrations_dir` in sorted order and executes each via `psycopg2` (the SQL already handles idempotency internally). Commits after each file.

- [ ] **Step 1: Extend `pyproject.toml`** — add new dependencies and the `certcoach` package

Replace the `[project]` dependencies list and `[tool.setuptools]` packages section:

```toml
[project]
name = "certcoach"
version = "0.1.0"
description = "Multi-tenant RAG coaching assistant for cloud certifications"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pyyaml>=6.0",
    "pydantic>=2.6",
    "psycopg2-binary>=2.9",
    "pgvector>=0.3",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["ingestion", "certcoach"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install updated dependencies**

Run:
```bash
.venv/bin/pip install -e ".[dev]"
```
Expected: `psycopg2-binary`, `pgvector`, and `python-dotenv` install without errors.

- [ ] **Step 3: Create `certcoach/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `certcoach/db.py`**

```python
from __future__ import annotations

from pathlib import Path

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv
import os

load_dotenv()


def get_conn() -> psycopg2.extensions.connection:
    url = os.environ["DATABASE_URL"]
    return psycopg2.connect(url)


def apply_migrations(
    conn: psycopg2.extensions.connection,
    migrations_dir: Path = Path("migrations"),
) -> None:
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
```

- [ ] **Step 5: Verify the module imports cleanly**

Run:
```bash
.venv/bin/python -c "from certcoach.db import get_conn, apply_migrations; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add certcoach/__init__.py certcoach/db.py pyproject.toml
git commit -m "feat: certcoach package + db connection + migration runner"
```

---

### Task 4: Integration tests against the live DB

**Files:**
- Create: `tests/test_db.py`

**Interfaces:**
- Consumes: `get_conn()` and `apply_migrations()` from `certcoach.db`; running `certcoach-db` container.
- Produces: a test suite that verifies the schema is correct and the migration is idempotent, against the real DB — no mocks.

> **Why no mocks:** mocking the DB masked a real migration failure in this project. Integration tests always hit the real container. See project history.

- [ ] **Step 1: Write the failing tests** — create `tests/test_db.py`

```python
import os
import pytest
import psycopg2
from pathlib import Path

from certcoach.db import apply_migrations, get_conn

# Skip all tests in this file if the DB is unreachable (e.g. CI without Docker).
def _db_reachable() -> bool:
    try:
        conn = get_conn()
        conn.close()
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason="certcoach-db container not running",
)


@pytest.fixture
def conn():
    c = get_conn()
    yield c
    c.close()


def test_get_conn_returns_open_connection(conn):
    assert conn.closed == 0


def test_apply_migrations_creates_chunks_table(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'chunks' AND table_schema = 'public'"
        )
        assert cur.fetchone()[0] == 1


def test_chunks_table_has_required_columns(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND table_schema = 'public'"
        )
        cols = {row[0] for row in cur.fetchall()}
    required = {
        "id", "tenant", "source_id", "source_version", "source_date",
        "citation", "doc_type", "parent_doc_id", "prev_chunk_id",
        "next_chunk_id", "chunk_index", "content", "token_count",
        "embedding", "fts", "created_at",
    }
    assert required <= cols, f"missing columns: {required - cols}"


def test_embedding_column_is_vector_3072(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "vector"
    # Verify dimension via pg_attribute
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "JOIN pg_class ON pg_class.oid = pg_attribute.attrelid "
            "WHERE pg_class.relname = 'chunks' AND pg_attribute.attname = 'embedding'"
        )
        # atttypmod for vector stores dimension directly
        dim = cur.fetchone()[0]
    assert dim == 3072, f"expected vector(3072), got vector({dim})"


def test_apply_migrations_is_idempotent(conn):
    apply_migrations(conn, Path("migrations"))
    # Running a second time must not raise
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = '001'")
        assert cur.fetchone()[0] == 1


def test_insert_and_retrieve_chunk(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks
                (tenant, source_id, citation, doc_type, chunk_index, content)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            ("aws-saa", "faq-ec2", "Amazon EC2 FAQs", "faq", 0, "EC2 is a virtual machine service."),
        )
        inserted_id = cur.fetchone()[0]
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT tenant, content FROM chunks WHERE id = %s", (inserted_id,))
        row = cur.fetchone()
    assert row == ("aws-saa", "EC2 is a virtual machine service.")
    # Clean up
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE id = %s", (inserted_id,))
    conn.commit()


def test_fts_generated_column_is_populated(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks
                (tenant, source_id, citation, doc_type, chunk_index, content)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            ("aws-saa", "faq-s3", "Amazon S3 FAQs", "faq", 0, "S3 stores objects in buckets."),
        )
        inserted_id = cur.fetchone()[0]
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fts IS NOT NULL FROM chunks WHERE id = %s", (inserted_id,)
        )
        assert cur.fetchone()[0] is True
    # Clean up
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE id = %s", (inserted_id,))
    conn.commit()


def test_tenant_isolation_index_is_used(conn):
    apply_migrations(conn, Path("migrations"))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname LIKE '%tenant%'"
        )
        indexes = {row[0] for row in cur.fetchall()}
    assert "chunks_tenant_idx" in indexes
    assert "chunks_tenant_source_idx" in indexes
```

- [ ] **Step 2: Run tests to verify they fail** (container must be running but migration not yet applied via Python)

Run:
```bash
.venv/bin/python -m pytest tests/test_db.py -v
```
Expected: tests are collected; if the container is running they should pass already (since we applied the migration manually in Task 2). If the table was dropped between tasks, they will show the schema assertion failures. The key check is that the tests *run* without import errors.

- [ ] **Step 3: Run the full suite to confirm no regressions**

Run:
```bash
.venv/bin/python -m pytest -v
```
Expected: all 22 existing tests + the new DB tests pass. DB tests are skipped if the container is not running.

- [ ] **Step 4: Commit**

```bash
git add tests/test_db.py
git commit -m "test: live-DB integration tests for schema + migration runner"
```

---

### Task 5: Smoke-test the full stack end-to-end

**Files:**
- No new files — this task exercises everything wired together.

**Interfaces:**
- Consumes: `get_conn()`, `apply_migrations()`, running `certcoach-db` container, `chunks` table.
- Produces: verified confidence that `docker compose up` + `apply_migrations` + insert/query works in one flow; final green suite.

- [ ] **Step 1: Drop and re-create the schema from scratch via Python**

Run:
```bash
docker compose exec db psql -U certcoach -d certcoach -c "DROP TABLE IF EXISTS chunks CASCADE; DROP TABLE IF EXISTS schema_migrations CASCADE;"
.venv/bin/python -c "
from certcoach.db import get_conn, apply_migrations
from pathlib import Path
conn = get_conn()
apply_migrations(conn, Path('migrations'))
conn.close()
print('migrations applied ok')
"
```
Expected: prints `NOTICE:  Migration 001 applied.` then `migrations applied ok`.

- [ ] **Step 2: Run the full test suite one final time**

Run:
```bash
.venv/bin/python -m pytest -v
```
Expected: all tests pass (DB tests run; none skipped unless container is down).

- [ ] **Step 3: Commit (if any last-minute fixes were made)**

```bash
git add -p   # stage only intentional changes
git commit -m "chore: verify docker+schema smoke test end-to-end"
```

---

## Self-review against spec

| Spec requirement | Covered by |
|---|---|
| Postgres + pgvector (Docker) | Task 1 |
| `chunks` table with `tenant`, `source`, `parent_doc_id`, neighbor ids, `source_version`, `source_date` | Task 2 |
| `tsvector` FTS index (same table, same rows — no second system) | Task 2 `fts` generated column + GIN index |
| `vector(3072)` for `text-embedding-3-large` | Task 2 |
| IVFFlat index for dense search | Task 2 |
| Tenant index for `tenant`-filtered queries | Task 2 |
| `pgdata/` gitignored (no corpus committed) | Pre-existing `.gitignore` |
| No mocks — real DB tests | Task 4 |
| `docker compose up` is the deliverable | Task 1 |
| Python `db` module for later ingestion/retrieval stages | Task 3 |
