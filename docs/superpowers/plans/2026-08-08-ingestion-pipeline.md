# Ingestion Pipeline (clean → chunk → embed → store) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn raw fetched documents in `data/` into embedded chunks stored in Postgres, producing a queryable corpus for both AWS SAA and GCP ACE tenants.

**Architecture:** Four stages in a single `ingestion/pipeline.py` module: (1) **clean** — parse HTML/PDF in `data/` to Markdown using `markdownify` for HTML and `pdfminer.six` for PDF; (2) **chunk** — structure-aware sliding window (~500–800 tokens, ~100-token overlap) using `tiktoken`; (3) **embed** — call `text-embedding-3-large` via `certcoach/ai_client.py` (new, wraps ai-core OpenAI-compatible gateway) with `dimensions=1536`; (4) **store** — bulk-insert rows into `chunks` table via `certcoach/db.py` and link `prev_chunk_id`/`next_chunk_id` neighbors. A CLI script `python -m ingestion.pipeline` drives the full pipeline. Integration tests hit the live DB (no mocks).

**Tech Stack:** Python ≥3.11, `markdownify`, `pdfminer.six`, `tiktoken`, `openai` (for ai-core OpenAI-compatible calls), `pgvector` Python client, `psycopg2-binary`, `pytest`.

## Global Constraints

- Python ≥ 3.11.
- **No mocks for DB or embedding integration tests** — always hit the real running container / real API.
- `data/` is gitignored — never committed. Raw docs stay local.
- No Azure. No exam-dump sources.
- Embedding model: `text-embedding-3-large` with `dimensions=1536` (Matryoshka truncation — IVFFlat requires ≤2000 dims).
- Chunk target: **500–800 tokens**, overlap **~100 tokens**, measured with `tiktoken` model `cl100k_base`.
- Tenant values: exactly `"aws-saa"` and `"gcp-ace"` (from `sources.yaml`).
- All model IDs are ai-core gateway aliases — use exactly as listed; do NOT substitute standard Anthropic IDs.
- `corpus: false` documents (sample questions) must be **skipped** by the pipeline (not cleaned, chunked, embedded, or stored).
- The `chunks` table already exists from migration `001` — do not re-create it; only INSERT.
- Neighbor links (`prev_chunk_id`, `next_chunk_id`) must be set after insert (IDs are assigned by Postgres BIGSERIAL).
- AI client env vars: `AICORE_BASE_URL` (gateway base URL), `AICORE_API_KEY` (token). Add both to `.env.example`.
- Per-doc upsert: before inserting a document's chunks, DELETE existing rows for `(tenant, source_id)` first so re-runs are idempotent.

---

## File map

| File | Create / Modify | Responsibility |
|---|---|---|
| `certcoach/ai_client.py` | Create | Model-agnostic client: `embed(texts, model) → list[list[float]]`; wraps ai-core OpenAI-compat gateway |
| `ingestion/clean.py` | Create | `clean_doc(path, format) → str` — HTML→Markdown via markdownify; PDF→text via pdfminer |
| `ingestion/chunk.py` | Create | `chunk_text(text, …) → list[Chunk]` — sliding window by token count using tiktoken |
| `ingestion/store.py` | Create | `store_chunks(conn, chunks) → int` — delete-then-insert with neighbor linking |
| `ingestion/pipeline.py` | Create | Orchestrates clean→chunk→embed→store; CLI entry point |
| `tests/test_clean.py` | Create | Unit tests for clean stage |
| `tests/test_chunk.py` | Create | Unit tests for chunker |
| `tests/test_store.py` | Create | Integration tests: store → DB, neighbor links, idempotency |
| `tests/test_pipeline.py` | Create | Integration smoke: full pipeline on one small fixture doc |
| `pyproject.toml` | Modify | Add `markdownify`, `pdfminer.six`, `tiktoken`, `openai` deps |
| `.env.example` | Modify | Add `AICORE_BASE_URL`, `AICORE_API_KEY` |

---

### Task 1: AI client — embed() wrapper

**Files:**
- Create: `certcoach/ai_client.py`
- Create: `tests/test_ai_client.py`
- Modify: `pyproject.toml` (add `openai>=1.30`)
- Modify: `.env.example` (add `AICORE_BASE_URL`, `AICORE_API_KEY`)

**Interfaces:**
- Consumes: env vars `AICORE_BASE_URL`, `AICORE_API_KEY` (loaded via `python-dotenv`).
- Produces:
  - `embed(texts: list[str], *, model: str = "text-embedding-3-large", dimensions: int = 1536) -> list[list[float]]` — returns one float list per input text, length == `dimensions`.
  - The function raises `ValueError` if `texts` is empty.
  - Module-level `_get_client() -> openai.OpenAI` (private, cached via `functools.lru_cache`) — builds `openai.OpenAI(base_url=AICORE_BASE_URL, api_key=AICORE_API_KEY)`. Test code can monkeypatch this.

- [ ] **Step 1: Add openai dependency**

Edit `pyproject.toml` dependencies list to append `"openai>=1.30"`. Also add `AICORE_BASE_URL` and `AICORE_API_KEY` to `.env.example`:

```
# ai-core gateway (internal model gateway, OpenAI-compatible)
AICORE_BASE_URL=https://your-aicore-endpoint/v1
AICORE_API_KEY=your-token-here
```

- [ ] **Step 2: Write the failing test (unit — monkeypatched client)**

Create `tests/test_ai_client.py`:

```python
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_client(vectors: list[list[float]]) -> MagicMock:
    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_client.embeddings = mock_embeddings
    items = [MagicMock(embedding=v) for v in vectors]
    mock_embeddings.create.return_value = MagicMock(data=items)
    return mock_client


def test_embed_returns_vectors():
    from certcoach.ai_client import embed
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock = _make_mock_client(vectors)
    with patch("certcoach.ai_client._get_client", return_value=mock):
        result = embed(["hello", "world"], dimensions=3)
    assert result == vectors
    mock.embeddings.create.assert_called_once_with(
        input=["hello", "world"],
        model="text-embedding-3-large",
        dimensions=3,
    )


def test_embed_empty_raises():
    from certcoach.ai_client import embed
    with pytest.raises(ValueError, match="empty"):
        embed([])


def test_embed_default_dimensions():
    from certcoach.ai_client import embed
    mock = _make_mock_client([[0.0] * 1536])
    with patch("certcoach.ai_client._get_client", return_value=mock):
        result = embed(["text"])
    assert len(result[0]) == 1536
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_ai_client.py -v
```
Expected: ImportError or AttributeError (module doesn't exist yet).

- [ ] **Step 4: Implement `certcoach/ai_client.py`**

```python
from __future__ import annotations
import functools
import os
from dotenv import load_dotenv

load_dotenv()


@functools.lru_cache(maxsize=1)
def _get_client():
    import openai
    return openai.OpenAI(
        base_url=os.environ["AICORE_BASE_URL"],
        api_key=os.environ["AICORE_API_KEY"],
    )


def embed(
    texts: list[str],
    *,
    model: str = "text-embedding-3-large",
    dimensions: int = 1536,
) -> list[list[float]]:
    if not texts:
        raise ValueError("texts must not be empty")
    client = _get_client()
    response = client.embeddings.create(input=texts, model=model, dimensions=dimensions)
    return [item.embedding for item in response.data]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_ai_client.py -v
```
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add certcoach/ai_client.py tests/test_ai_client.py pyproject.toml .env.example
git commit -m "feat: ai_client embed() wrapper for ai-core gateway"
```

---

### Task 2: Clean stage — HTML and PDF to Markdown

**Files:**
- Create: `ingestion/clean.py`
- Create: `tests/test_clean.py`
- Modify: `pyproject.toml` (add `markdownify>=0.12`, `pdfminer.six>=20221105`)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `clean_doc(path: Path, fmt: str) -> str` — `fmt` is `"html"` or `"pdf"`. Returns clean Markdown string. Raises `ValueError` for unsupported `fmt`. Raises `FileNotFoundError` if `path` doesn't exist.
  - For HTML: strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags first, then runs `markdownify.markdownify(html, heading_style="ATX")`. Collapses 3+ consecutive blank lines to 2.
  - For PDF: uses `pdfminer.high_level.extract_text(path)`. Returns raw extracted text (no Markdown conversion — PDF structure is too unpredictable; the chunker works on plain text too).

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml` to append `"markdownify>=0.12"` and `"pdfminer.six>=20221105"` to the dependencies list.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_clean.py`:

```python
from __future__ import annotations
import re
import pytest
from pathlib import Path


HTML_SIMPLE = """\
<html><body>
<nav>skip nav</nav>
<h1>EC2 FAQ</h1>
<script>alert(1)</script>
<p>What is <strong>EC2</strong>? It is a compute service.</p>
<footer>skip footer</footer>
</body></html>
"""

HTML_BLANK_LINES = "<html><body><p>a</p>\n\n\n\n\n<p>b</p></body></html>"


def test_clean_html_produces_markdown(tmp_path):
    from ingestion.clean import clean_doc
    f = tmp_path / "test.html"
    f.write_text(HTML_SIMPLE, encoding="utf-8")
    md = clean_doc(f, "html")
    assert "# EC2 FAQ" in md
    assert "**EC2**" in md
    assert "skip nav" not in md
    assert "skip footer" not in md
    assert "alert" not in md


def test_clean_html_collapses_blank_lines(tmp_path):
    from ingestion.clean import clean_doc
    f = tmp_path / "test.html"
    f.write_text(HTML_BLANK_LINES, encoding="utf-8")
    md = clean_doc(f, "html")
    assert "\n\n\n" not in md


def test_clean_pdf_returns_text(tmp_path):
    # Use the real fixture PDF from data/ if available, otherwise skip
    pdf = Path("data/aws-saa/saa-exam-guide.pdf")
    if not pdf.exists():
        pytest.skip("data/aws-saa/saa-exam-guide.pdf not fetched")
    from ingestion.clean import clean_doc
    text = clean_doc(pdf, "pdf")
    assert len(text) > 500
    assert isinstance(text, str)


def test_clean_unsupported_format_raises(tmp_path):
    from ingestion.clean import clean_doc
    f = tmp_path / "x.docx"
    f.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="unsupported format"):
        clean_doc(f, "docx")


def test_clean_missing_file_raises(tmp_path):
    from ingestion.clean import clean_doc
    with pytest.raises(FileNotFoundError):
        clean_doc(tmp_path / "nonexistent.html", "html")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_clean.py -v
```
Expected: ImportError (module doesn't exist yet).

- [ ] **Step 4: Implement `ingestion/clean.py`**

```python
from __future__ import annotations
import re
from pathlib import Path

_BLANK_RE = re.compile(r"\n{3,}")
_STRIP_TAGS = ["script", "style", "nav", "footer", "header"]


def _strip_html_tags(html: str, tags: list[str]) -> str:
    for tag in tags:
        html = re.sub(rf"(?is)<{tag}[^>]*>.*?</{tag}>", "", html)
    return html


def clean_doc(path: Path, fmt: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if fmt == "html":
        import markdownify
        html = path.read_text(encoding="utf-8", errors="replace")
        html = _strip_html_tags(html, _STRIP_TAGS)
        md = markdownify.markdownify(html, heading_style="ATX")
        md = _BLANK_RE.sub("\n\n", md)
        return md.strip()
    if fmt == "pdf":
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    raise ValueError(f"unsupported format: {fmt!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_clean.py -v
```
Expected: 4 tests pass (PDF test passes or skips depending on data/ presence).

- [ ] **Step 6: Commit**

```bash
git add ingestion/clean.py tests/test_clean.py pyproject.toml
git commit -m "feat: clean stage — HTML/PDF to Markdown"
```

---

### Task 3: Chunk stage — sliding window by token count

**Files:**
- Create: `ingestion/chunk.py`
- Create: `tests/test_chunk.py`
- Modify: `pyproject.toml` (add `tiktoken>=0.7`)

**Interfaces:**
- Consumes: clean text string from Task 2's `clean_doc`.
- Produces:
  - `Chunk` dataclass: `content: str`, `chunk_index: int`, `token_count: int`.
  - `chunk_text(text: str, *, target_tokens: int = 650, overlap_tokens: int = 100, encoding_name: str = "cl100k_base") -> list[Chunk]` — splits on sentence/paragraph boundaries where possible, falls back to hard token cuts. Returns at least one `Chunk` for any non-empty text. Empty text returns `[]`.
  - Guarantees: every chunk has `50 ≤ token_count ≤ target_tokens + 50` (except possibly the last chunk which may be smaller). Overlap means the last `overlap_tokens` worth of the previous chunk's tokens start the next chunk.

- [ ] **Step 1: Add tiktoken dependency**

Edit `pyproject.toml` to append `"tiktoken>=0.7"` to the dependencies list.

- [ ] **Step 2: Write failing tests**

Create `tests/test_chunk.py`:

```python
from __future__ import annotations
import pytest


SHORT = "Hello world."
MEDIUM = ("AWS provides a suite of services. " * 50).strip()   # ~300 tokens
LONG = ("Amazon EC2 is a web service that provides resizable compute capacity. " * 100).strip()  # ~1400 tokens


def test_empty_returns_empty():
    from ingestion.chunk import chunk_text
    assert chunk_text("") == []


def test_short_text_is_one_chunk():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(SHORT)
    assert len(chunks) == 1
    assert chunks[0].content == SHORT
    assert chunks[0].chunk_index == 0


def test_long_text_splits_into_multiple_chunks():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(LONG, target_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 3
    # Every chunk within bounds (last chunk may be small)
    for i, c in enumerate(chunks[:-1]):
        assert 20 <= c.token_count <= 260, f"chunk {i} token_count={c.token_count}"


def test_chunk_indices_are_sequential():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(LONG, target_tokens=200, overlap_tokens=20)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_overlap_content_appears_in_next_chunk():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(LONG, target_tokens=200, overlap_tokens=50)
    if len(chunks) < 2:
        pytest.skip("text too short to produce overlap")
    # Last sentence of chunk 0 should appear somewhere in chunk 1
    tail = chunks[0].content.split()[-5:]
    tail_str = " ".join(tail)
    assert tail_str in chunks[1].content


def test_token_count_matches_encoding():
    import tiktoken
    from ingestion.chunk import chunk_text
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_text(MEDIUM, target_tokens=100, overlap_tokens=10)
    for c in chunks:
        actual = len(enc.encode(c.content))
        assert abs(actual - c.token_count) <= 2, (
            f"reported {c.token_count} but encoded {actual}"
        )
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_chunk.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `ingestion/chunk.py`**

```python
from __future__ import annotations
import re
from dataclasses import dataclass

import tiktoken


@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int


_PARA_RE = re.compile(r"\n{2,}")


def chunk_text(
    text: str,
    *,
    target_tokens: int = 650,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    if not text.strip():
        return []
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    total = len(tokens)

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < total:
        end = min(start + target_tokens, total)
        # Extend to a paragraph boundary if within 50 tokens
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        # Try to break at last paragraph boundary within the chunk
        para_match = list(_PARA_RE.finditer(chunk_text_str))
        if para_match and end < total:
            last_para = para_match[-1]
            # Only break at paragraph if at least 50 tokens remain after start
            para_end_tokens = len(enc.encode(chunk_text_str[: last_para.end()]))
            if para_end_tokens >= 50:
                chunk_tokens = enc.encode(chunk_text_str[: last_para.end()])
                chunk_text_str = enc.decode(chunk_tokens)
        content = chunk_text_str.strip()
        if content:
            chunks.append(Chunk(
                content=content,
                chunk_index=idx,
                token_count=len(enc.encode(content)),
            ))
            idx += 1
        # Advance by (chunk_size - overlap)
        advance = max(len(chunk_tokens) - overlap_tokens, 1)
        start += advance
    return chunks
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_chunk.py -v
```
Expected: all 6 pass.

- [ ] **Step 6: Commit**

```bash
git add ingestion/chunk.py tests/test_chunk.py pyproject.toml
git commit -m "feat: chunk stage — sliding window by token count"
```

---

### Task 4: Store stage — bulk insert with neighbor links

**Files:**
- Create: `ingestion/store.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Consumes:
  - `certcoach.db.get_conn()` — psycopg2 connection
  - `ingestion.chunk.Chunk` dataclass
  - `ingestion.sources.DocSpec` for metadata
- Produces:
  - `ChunkRow` dataclass: `content: str`, `chunk_index: int`, `token_count: int`, `embedding: list[float]`
  - `store_chunks(conn, *, tenant: str, doc: DocSpec, source_date: str | None, chunks: list[ChunkRow]) -> int` — deletes existing rows for `(tenant, doc.id)`, bulk-inserts new rows, then updates `prev_chunk_id`/`next_chunk_id` for neighbor linking. Returns row count inserted.
  - Delete-then-insert makes each call idempotent per `(tenant, source_id)`.
  - Neighbor update: after INSERT, issue one UPDATE per chunk to set `prev_chunk_id = id of chunk_index-1` and `next_chunk_id = id of chunk_index+1`. The first chunk has `prev_chunk_id = NULL`, the last has `next_chunk_id = NULL`.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_store.py`:

```python
from __future__ import annotations
import os
import pytest
import psycopg2


def _db_reachable() -> bool:
    try:
        from certcoach.db import get_conn
        conn = get_conn()
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(), reason="Postgres container not running"
)


@pytest.fixture
def conn():
    from certcoach.db import get_conn, apply_migrations
    from pathlib import Path
    c = get_conn()
    apply_migrations(c, Path("migrations"))
    yield c
    c.rollback()
    c.close()


def _make_doc():
    from ingestion.sources import DocSpec
    return DocSpec(
        id="test-source",
        title="Test Source",
        url="https://example.com",
        type="faq",
        format="html",
        citation="Test Citation",
        corpus=True,
        source_version=None,
    )


def _make_chunk_rows(n: int = 3):
    from ingestion.store import ChunkRow
    return [
        ChunkRow(
            content=f"chunk content {i} " + "word " * 20,
            chunk_index=i,
            token_count=25,
            embedding=[0.1] * 1536,
        )
        for i in range(n)
    ]


def test_store_inserts_rows(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(3)
    count = store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    assert count == 3
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 3


def test_store_is_idempotent(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(3)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 3


def test_neighbor_links_are_set(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(3)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_index, prev_chunk_id, next_chunk_id "
            "FROM chunks WHERE tenant='aws-saa' AND source_id='test-source' "
            "ORDER BY chunk_index"
        )
        rows = cur.fetchall()
    assert rows[0][1] is None        # first: no prev
    assert rows[0][2] == rows[1][0] or rows[0][2] is not None  # first: has next
    assert rows[1][1] is not None    # middle: has prev
    assert rows[1][2] is not None    # middle: has next
    assert rows[2][2] is None        # last: no next


def test_tenant_isolation(conn):
    from ingestion.store import store_chunks
    doc = _make_doc()
    chunks = _make_chunk_rows(2)
    store_chunks(conn, tenant="aws-saa", doc=doc, source_date="2026-08-08", chunks=chunks)
    store_chunks(conn, tenant="gcp-ace", doc=doc, source_date="2026-08-08", chunks=chunks)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='aws-saa' AND source_id='test-source'")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM chunks WHERE tenant='gcp-ace' AND source_id='test-source'")
        assert cur.fetchone()[0] == 2


def test_cleanup(conn):
    # Clean up test rows so they don't pollute other tests
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id='test-source'")
    conn.commit()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_store.py -v
```
Expected: ImportError (module doesn't exist). If DB not running, all tests skip — that's correct.

- [ ] **Step 3: Implement `ingestion/store.py`**

```python
from __future__ import annotations
from dataclasses import dataclass

import psycopg2.extras

from ingestion.sources import DocSpec


@dataclass
class ChunkRow:
    content: str
    chunk_index: int
    token_count: int
    embedding: list[float]


def store_chunks(
    conn,
    *,
    tenant: str,
    doc: DocSpec,
    source_date: str | None,
    chunks: list[ChunkRow],
) -> int:
    if not chunks:
        return 0
    with conn.cursor() as cur:
        # Idempotent: delete existing rows for this (tenant, source_id)
        cur.execute(
            "DELETE FROM chunks WHERE tenant = %s AND source_id = %s",
            (tenant, doc.id),
        )
        # Bulk insert — embedding as list (pgvector accepts Python list)
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks
                (tenant, source_id, source_version, source_date, citation,
                 doc_type, parent_doc_id, chunk_index, content, token_count, embedding)
            VALUES %s
            RETURNING id, chunk_index
            """,
            [
                (
                    tenant,
                    doc.id,
                    doc.source_version,
                    source_date,
                    doc.citation,
                    doc.type,
                    doc.id,
                    c.chunk_index,
                    c.content,
                    c.token_count,
                    c.embedding,
                )
                for c in chunks
            ],
            fetch=True,
        )
        rows = cur.fetchall()  # [(id, chunk_index), ...]
        id_by_idx = {ci: rid for rid, ci in rows}

        # Update neighbor links
        for rid, ci in rows:
            prev_id = id_by_idx.get(ci - 1)
            next_id = id_by_idx.get(ci + 1)
            cur.execute(
                "UPDATE chunks SET prev_chunk_id = %s, next_chunk_id = %s WHERE id = %s",
                (prev_id, next_id, rid),
            )
    return len(chunks)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_store.py -v
```
Expected: 5 pass (or skip if DB not running).

- [ ] **Step 5: Commit**

```bash
git add ingestion/store.py tests/test_store.py
git commit -m "feat: store stage — bulk insert with neighbor links"
```

---

### Task 5: Pipeline orchestrator + CLI

**Files:**
- Create: `ingestion/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - `ingestion.sources.load_manifest`, `DocSpec`
  - `ingestion.clean.clean_doc`
  - `ingestion.chunk.chunk_text` → `list[Chunk]`
  - `certcoach.ai_client.embed`
  - `ingestion.store.store_chunks`, `ChunkRow`
  - `certcoach.db.get_conn`, `apply_migrations`
- Produces:
  - `run_pipeline(manifest_path, data_dir, *, tenants=None, batch_size=32) -> dict[str, int]` — returns `{source_id: chunk_count}` for all docs processed. Skips `corpus=False` docs. Calls `embed()` in batches of `batch_size` texts.
  - `main(argv=None) -> int` — CLI entry point (argparse). Args: `--manifest`, `--data-dir`, `--tenant` (repeatable), `--batch-size`.
  - Per-doc progress printed to stdout: `"[aws-saa] faq-ec2 -> 87 chunks"`.
  - Any doc that fails (file missing, clean error) is logged with `print(f"SKIP {doc.id}: {e}")` and the pipeline continues.

- [ ] **Step 1: Write the failing integration smoke test**

Create `tests/test_pipeline.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
import pytest


def _db_reachable() -> bool:
    try:
        from certcoach.db import get_conn
        c = get_conn(); c.close(); return True
    except Exception:
        return False


def _aicore_configured() -> bool:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.environ.get("AICORE_BASE_URL") and os.environ.get("AICORE_API_KEY"))


pytestmark = pytest.mark.skipif(
    not _db_reachable() or not _aicore_configured(),
    reason="Postgres container or ai-core credentials not available"
)


def test_pipeline_single_doc(tmp_path):
    """Run pipeline on a tiny synthetic HTML doc — no real API call, use a stub embed."""
    from unittest.mock import patch
    from ingestion.pipeline import run_pipeline
    from ingestion.sources import Manifest, TenantSpec, DocSpec

    # Write a tiny HTML fixture
    html = "<html><body><h1>Test</h1><p>" + ("word " * 200) + "</p></body></html>"
    doc_path = tmp_path / "aws-saa" / "test-fixture.html"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(html, encoding="utf-8")

    stub_embed = lambda texts, **kw: [[0.0] * 1536 for _ in texts]

    with patch("ingestion.pipeline.embed", side_effect=stub_embed):
        result = run_pipeline(
            manifest_path="ingestion/sources.yaml",
            data_dir=tmp_path,
            tenants=["aws-saa"],
            _manifest_override=Manifest(
                version=1,
                tenants={
                    "aws-saa": TenantSpec(
                        display_name="AWS SAA",
                        license="test",
                        documents=[
                            DocSpec(
                                id="test-fixture",
                                title="Test Fixture",
                                url="https://example.com",
                                type="faq",
                                format="html",
                                citation="Test Fixture",
                                corpus=True,
                            )
                        ],
                    )
                },
            ),
        )
    assert "test-fixture" in result
    assert result["test-fixture"] > 0

    # Verify rows in DB
    from certcoach.db import get_conn
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE source_id = 'test-fixture'")
        assert cur.fetchone()[0] == result["test-fixture"]
    # Cleanup
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id = 'test-fixture'")
    conn.commit()
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `ingestion/pipeline.py`**

```python
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

from certcoach.ai_client import embed
from certcoach.db import apply_migrations, get_conn
from ingestion.chunk import chunk_text
from ingestion.clean import clean_doc
from ingestion.sources import Manifest, load_manifest
from ingestion.store import ChunkRow, store_chunks


def run_pipeline(
    manifest_path: str | Path = "ingestion/sources.yaml",
    data_dir: str | Path = "data",
    *,
    tenants: list[str] | None = None,
    batch_size: int = 32,
    _manifest_override: Manifest | None = None,
) -> dict[str, int]:
    manifest = _manifest_override or load_manifest(manifest_path)
    data_dir = Path(data_dir)
    selected = tenants or list(manifest.tenants.keys())

    conn = get_conn()
    apply_migrations(conn)

    results: dict[str, int] = {}
    for tenant_name in selected:
        tenant = manifest.tenants[tenant_name]
        for doc in tenant.documents:
            if not doc.corpus:
                continue
            raw_path = data_dir / tenant_name / f"{doc.id}.{doc.format}"
            try:
                text = clean_doc(raw_path, doc.format)
                chunks = chunk_text(text)
                if not chunks:
                    print(f"[{tenant_name}] {doc.id} -> 0 chunks (empty after clean)")
                    results[doc.id] = 0
                    continue

                # Embed in batches
                texts = [c.content for c in chunks]
                embeddings: list[list[float]] = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    embeddings.extend(embed(batch))

                chunk_rows = [
                    ChunkRow(
                        content=c.content,
                        chunk_index=c.chunk_index,
                        token_count=c.token_count,
                        embedding=embeddings[j],
                    )
                    for j, c in enumerate(chunks)
                ]
                source_date = datetime.now(timezone.utc).date().isoformat()
                n = store_chunks(
                    conn,
                    tenant=tenant_name,
                    doc=doc,
                    source_date=source_date,
                    chunks=chunk_rows,
                )
                conn.commit()
                print(f"[{tenant_name}] {doc.id} -> {n} chunks")
                results[doc.id] = n
            except Exception as e:
                print(f"SKIP {doc.id}: {e}")
                conn.rollback()
    conn.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CertCoach ingestion pipeline.")
    parser.add_argument("--manifest", default="ingestion/sources.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tenant", action="append", dest="tenants")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    results = run_pipeline(
        args.manifest, args.data_dir,
        tenants=args.tenants, batch_size=args.batch_size,
    )
    total = sum(results.values())
    print(f"\nTotal: {total} chunks across {len(results)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```
Expected: passes (or skips if DB/ai-core not available).

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```
Expected: all existing tests still pass; new tests pass or skip gracefully.

- [ ] **Step 6: Commit**

```bash
git add ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingestion pipeline orchestrator + CLI"
```

---

### Task 6: Spot-check and install dependencies

**Files:**
- Modify: `pyproject.toml` (confirm all new deps present)

This task runs the actual ingestion end-to-end on the real fetched data and does a manual spot-check of chunk quality.

**Interfaces:**
- Consumes: all of Tasks 1–5.
- Produces: verified corpus in Postgres. No new source files.

- [ ] **Step 1: Install all new dependencies**

```bash
pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

Verify imports work:
```bash
python -c "import markdownify, pdfminer, tiktoken, openai; print('ok')"
```

- [ ] **Step 2: Confirm Docker DB is running**

```bash
docker compose ps
```
Expected: `certcoach-db` Up and healthy. If not: `docker compose up -d`.

- [ ] **Step 3: Run full test suite**

```bash
pytest -v
```
Expected: all tests pass or skip. Zero failures.

- [ ] **Step 4: Run pipeline on AWS SAA tenant**

```bash
python -m ingestion.pipeline --tenant aws-saa
```
Expected output: lines like `[aws-saa] faq-ec2 -> 87 chunks`. No SKIP lines for corpus docs.

- [ ] **Step 5: Spot-check chunk quality in Postgres**

```bash
python - <<'EOF'
from certcoach.db import get_conn
conn = get_conn()
with conn.cursor() as cur:
    cur.execute("SELECT count(*), source_id FROM chunks WHERE tenant='aws-saa' GROUP BY source_id ORDER BY source_id")
    for row in cur.fetchall():
        print(row)
EOF
```
Verify: each `corpus=True` source has > 0 chunks. Check one chunk manually:
```bash
python - <<'EOF'
from certcoach.db import get_conn
conn = get_conn()
with conn.cursor() as cur:
    cur.execute("SELECT content, token_count, prev_chunk_id, next_chunk_id FROM chunks WHERE tenant='aws-saa' AND source_id='faq-ec2' ORDER BY chunk_index LIMIT 3")
    for row in cur.fetchall():
        print(f"tokens={row[1]} prev={row[2]} next={row[3]}")
        print(row[0][:200])
        print("---")
EOF
```
Verify: content looks like clean Markdown (not raw HTML tags or garbage). token_count 50–800. Neighbor IDs set (except first/last).

- [ ] **Step 6: Run pipeline on GCP ACE tenant**

```bash
python -m ingestion.pipeline --tenant gcp-ace
```
Verify: GCP docs produce chunks. `ace-sample-questions` is skipped (corpus=False).

- [ ] **Step 7: Final test suite**

```bash
pytest -v
```
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml
git commit -m "chore: verify ingestion pipeline end-to-end on both tenants"
```
