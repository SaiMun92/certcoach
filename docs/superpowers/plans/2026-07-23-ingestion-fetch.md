# Ingestion Fetch Stage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fetch stage of CertCoach ingestion — read `ingestion/sources.yaml`, download every source's raw bytes to gitignored `data/<tenant>/<id>.<ext>`, and write a provenance index `data/fetch_index.json`.

**Architecture:** A pure, testable core (`fetch_all` with an injected `fetch_fn`) plus a thin I/O shell (`main` wiring a real `httpx` client). Manifest and provenance are Pydantic models. No cleaning, chunking, embedding, DB, or API — those are later stages.

**Tech Stack:** Python ≥3.11, `httpx`, `pyyaml`, `pydantic` v2; `pytest` for tests.

**Design doc:** `docs/superpowers/specs/2026-07-23-ingestion-fetch-design.md`

## Global Constraints

- **Python ≥3.11.**
- **Fetch stage only** — no clean/parse, chunking, embeddings, pgvector, FTS, or API in this plan.
- **First-party vendor sources only; no exam dumps** (copyright + NDA + indefensible). Raw fetched docs live in **gitignored `data/`, never committed** (spec §8).
- **Failures are recorded, not hidden** — non-200 or thin-content responses produce `ok: false` + a note in the index; they do not write a content file.
- **TDD**: write the failing test first; **commit after each green task**.
- Clean-stage output format is Markdown, but that is a *later* stage — out of scope here.

---

### Task 1: Project scaffold + smoke test

**Files:**
- Create: `pyproject.toml`
- Create: `ingestion/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed, importable `ingestion` package; a working `pytest` setup via a `.venv`.

- [ ] **Step 1: Create `pyproject.toml`**

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
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["ingestion"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `ingestion/__init__.py`** (empty file marking the package)

```python
```

- [ ] **Step 3: Write the smoke test** — `tests/test_smoke.py`

```python
def test_ingestion_package_imports():
    import ingestion  # noqa: F401
```

- [ ] **Step 4: Create the venv and install (editable + dev)**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```
Expected: install completes; `httpx`, `pyyaml`, `pydantic`, `pytest` present.

- [ ] **Step 5: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml ingestion/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold certcoach package + pytest"
```

---

### Task 2: Manifest models + loader (`ingestion/sources.py`)

**Files:**
- Create: `ingestion/sources.py`
- Create: `tests/test_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DocSpec` — fields: `id: str`, `title: str`, `url: str`, `type: Literal["exam_guide","faq","whitepaper","doc","sample_questions"]`, `format: Literal["pdf","html"]`, `citation: str`, `corpus: bool = True`, `source_version: str | None = None`.
  - `TenantSpec` — `display_name: str`, `license: str`, `documents: list[DocSpec]`.
  - `Manifest` — `version: int`, `tenants: dict[str, TenantSpec]`.
  - `load_manifest(path: str | Path) -> Manifest` — parses YAML, applies `defaults.corpus` to docs that omit `corpus`.

- [ ] **Step 1: Write the failing tests** — `tests/test_sources.py`

```python
import pytest
from pydantic import ValidationError

from ingestion.sources import Manifest, load_manifest

MANIFEST_YAML = """
version: 1
defaults:
  fetched_date: null
  corpus: true
tenants:
  aws-saa:
    display_name: "AWS SAA"
    license: "test license"
    documents:
      - id: faq-ec2
        title: "EC2 FAQs"
        url: "https://aws.amazon.com/ec2/faqs/"
        type: faq
        format: html
        citation: "Amazon EC2 FAQs"
      - id: saa-sample-questions
        title: "Sample Questions"
        url: "https://example.com/sample.pdf"
        type: sample_questions
        format: pdf
        corpus: false
        citation: "AWS Sample Questions"
"""


def _write(tmp_path, text):
    p = tmp_path / "sources.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_manifest_parses_tenants_and_docs(tmp_path):
    m = load_manifest(_write(tmp_path, MANIFEST_YAML))
    assert isinstance(m, Manifest)
    assert m.version == 1
    aws = m.tenants["aws-saa"]
    assert aws.display_name == "AWS SAA"
    assert len(aws.documents) == 2


def test_corpus_defaults_true_and_override(tmp_path):
    docs = {d.id: d for d in load_manifest(_write(tmp_path, MANIFEST_YAML)).tenants["aws-saa"].documents}
    assert docs["faq-ec2"].corpus is True            # inherited from defaults
    assert docs["saa-sample-questions"].corpus is False  # explicit override


def test_invalid_format_raises(tmp_path):
    bad = MANIFEST_YAML.replace("format: html", "format: docx")
    with pytest.raises(ValidationError):
        load_manifest(_write(tmp_path, bad))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.sources'`.

- [ ] **Step 3: Write the implementation** — `ingestion/sources.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

DocType = Literal["exam_guide", "faq", "whitepaper", "doc", "sample_questions"]
DocFormat = Literal["pdf", "html"]


class DocSpec(BaseModel):
    id: str
    title: str
    url: str
    type: DocType
    format: DocFormat
    citation: str
    corpus: bool = True
    source_version: str | None = None


class TenantSpec(BaseModel):
    display_name: str
    license: str
    documents: list[DocSpec]


class Manifest(BaseModel):
    version: int
    tenants: dict[str, TenantSpec]


def load_manifest(path: str | Path) -> Manifest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    default_corpus = defaults.get("corpus", True)
    for tenant in (raw.get("tenants") or {}).values():
        for doc in tenant.get("documents", []):
            doc.setdefault("corpus", default_corpus)
    return Manifest(version=raw["version"], tenants=raw.get("tenants") or {})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sources.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add ingestion/sources.py tests/test_sources.py
git commit -m "feat: manifest models + loader for ingestion sources"
```

---

### Task 3: Provenance record + pure helpers (`ingestion/fetch.py`)

**Files:**
- Create: `ingestion/fetch.py`
- Create: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `DocSpec` from `ingestion.sources`.
- Produces:
  - `FetchFn = Callable[[str], tuple[int, dict[str, str], bytes]]` — url → (status, lowercase-keyed headers, body).
  - `FetchRecord` (Pydantic) — fields: `tenant, id, url, type, format, corpus, citation, source_version, fetched_date, http_status, content_type, bytes, sha256, ok, note`.
  - `target_path(data_dir: Path, tenant: str, doc: DocSpec) -> Path` → `data_dir/tenant/<id>.<format>`.
  - `classify_response(doc: DocSpec, status: int, body: bytes) -> tuple[bool, str]`.
  - Constants: `USER_AGENT: str`, `THIN_HTML_MIN_WORDS: int = 200`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_fetch.py`

```python
from pathlib import Path

from ingestion.sources import DocSpec
from ingestion.fetch import classify_response, target_path


def _doc(**kw):
    base = dict(
        id="faq-ec2", title="EC2 FAQs", url="https://aws.amazon.com/ec2/faqs/",
        type="faq", format="html", citation="Amazon EC2 FAQs",
        corpus=True, source_version=None,
    )
    base.update(kw)
    return DocSpec(**base)


def test_target_path_uses_tenant_and_format():
    doc = _doc(id="faq-s3", format="html")
    assert target_path(Path("/data"), "aws-saa", doc) == Path("/data/aws-saa/faq-s3.html")


def test_classify_response_ok_html():
    body = ("<html><body>" + "word " * 300 + "</body></html>").encode()
    assert classify_response(_doc(format="html"), 200, body) == (True, "ok")


def test_classify_response_non_200():
    ok, note = classify_response(_doc(), 404, b"")
    assert ok is False and "404" in note


def test_classify_response_thin_html():
    ok, note = classify_response(_doc(format="html"), 200, b"<html></html>")
    assert ok is False and "thin" in note


def test_classify_response_pdf_ok():
    assert classify_response(_doc(format="pdf"), 200, b"%PDF-1.7 body bytes")[0] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.fetch'`.

- [ ] **Step 3: Write the implementation** — create `ingestion/fetch.py`

```python
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from ingestion.sources import DocSpec

FetchFn = Callable[[str], "tuple[int, dict[str, str], bytes]"]

USER_AGENT = "CertCoach-ingest/0.1 (portfolio project)"
THIN_HTML_MIN_WORDS = 200

_SCRIPT_RE = re.compile(r"(?is)<script.*?</script>")
_STYLE_RE = re.compile(r"(?is)<style.*?</style>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")


class FetchRecord(BaseModel):
    tenant: str
    id: str
    url: str
    type: str
    format: str
    corpus: bool
    citation: str
    source_version: str | None
    fetched_date: str
    http_status: int
    content_type: str | None
    bytes: int
    sha256: str
    ok: bool
    note: str


def target_path(data_dir: Path, tenant: str, doc: DocSpec) -> Path:
    return data_dir / tenant / f"{doc.id}.{doc.format}"


def _visible_word_count(body: bytes) -> int:
    text = body.decode("utf-8", errors="ignore")
    text = _SCRIPT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return len(text.split())


def classify_response(doc: DocSpec, status: int, body: bytes) -> tuple[bool, str]:
    if status != 200:
        return False, f"http {status}"
    if doc.format == "html" and _visible_word_count(body) < THIN_HTML_MIN_WORDS:
        return False, "thin content - possible JS shell"
    return True, "ok"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add ingestion/fetch.py tests/test_fetch.py
git commit -m "feat: fetch provenance record + path/response helpers"
```

---

### Task 4: Fetch orchestration + provenance index (`fetch_all`, `write_index`)

**Files:**
- Modify: `ingestion/fetch.py` (add `now_iso`, `fetch_all`, `write_index`)
- Modify: `tests/test_fetch.py` (add orchestration tests)

**Interfaces:**
- Consumes: `Manifest`, `TenantSpec`, `DocSpec` from `ingestion.sources`; `FetchRecord`, `target_path`, `classify_response`, `FetchFn`.
- Produces:
  - `now_iso() -> str` (UTC ISO-8601).
  - `fetch_all(manifest: Manifest, data_dir: Path, fetch_fn: FetchFn, *, tenants: list[str] | None = None, delay: float = 0.0, now_fn: Callable[[], str] = now_iso, sleep_fn: Callable[[float], None] = time.sleep) -> list[FetchRecord]` — writes a content file for each `ok` doc; always appends a record.
  - `write_index(data_dir: Path, records: list[FetchRecord]) -> Path` — writes/merges `data_dir/fetch_index.json`; replaces entries for tenants present in `records`, preserves other tenants, sorts by `(tenant, id)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_fetch.py`

```python
import json

from ingestion.sources import Manifest, TenantSpec
from ingestion.fetch import FetchRecord, fetch_all, write_index


def _manifest(**tenants):
    return Manifest(version=1, tenants=tenants)


def _tenant(*docs):
    return TenantSpec(display_name="T", license="L", documents=list(docs))


def _fake_fetch(responses):
    def fetch(url):
        return responses[url]
    return fetch


_HTML_OK = ("<html><body>" + "word " * 300 + "</body></html>").encode()


def test_fetch_all_writes_ok_file_and_record(tmp_path):
    doc = _doc(url="https://x/ec2")
    m = _manifest(**{"aws-saa": _tenant(doc)})
    fetch = _fake_fetch({"https://x/ec2": (200, {"content-type": "text/html"}, _HTML_OK)})
    records = fetch_all(m, tmp_path, fetch, now_fn=lambda: "2026-07-23T00:00:00+00:00")
    assert (tmp_path / "aws-saa" / "faq-ec2.html").read_bytes() == _HTML_OK
    assert len(records) == 1
    r = records[0]
    assert r.ok and r.bytes == len(_HTML_OK) and len(r.sha256) == 64
    assert r.content_type == "text/html"
    assert r.fetched_date == "2026-07-23T00:00:00+00:00"


def test_fetch_all_skips_file_on_failure(tmp_path):
    doc = _doc(url="https://x/ec2")
    m = _manifest(**{"aws-saa": _tenant(doc)})
    fetch = _fake_fetch({"https://x/ec2": (404, {}, b"")})
    records = fetch_all(m, tmp_path, fetch)
    assert not (tmp_path / "aws-saa" / "faq-ec2.html").exists()
    assert records[0].ok is False and "404" in records[0].note


def test_fetch_all_tenant_scoping(tmp_path):
    a = _doc(id="a", url="https://x/a")
    b = _doc(id="b", url="https://x/b")
    m = _manifest(**{"aws-saa": _tenant(a), "gcp-ace": _tenant(b)})
    fetch = _fake_fetch({
        "https://x/a": (200, {"content-type": "text/html"}, _HTML_OK),
        "https://x/b": (200, {"content-type": "text/html"}, _HTML_OK),
    })
    records = fetch_all(m, tmp_path, fetch, tenants=["aws-saa"])
    assert {r.tenant for r in records} == {"aws-saa"}


def _record(tenant, id):
    return FetchRecord(
        tenant=tenant, id=id, url="u", type="faq", format="html", corpus=True,
        citation="c", source_version=None, fetched_date="t", http_status=200,
        content_type="text/html", bytes=1, sha256="0" * 64, ok=True, note="ok",
    )


def test_write_index_merges_preserving_other_tenants(tmp_path):
    write_index(tmp_path, [_record("gcp-ace", "b")])
    write_index(tmp_path, [_record("aws-saa", "a")])  # scoped re-run
    data = json.loads((tmp_path / "fetch_index.json").read_text())
    assert {e["tenant"] for e in data} == {"aws-saa", "gcp-ace"}


def test_write_index_replaces_same_tenant(tmp_path):
    write_index(tmp_path, [_record("aws-saa", "a")])
    write_index(tmp_path, [_record("aws-saa", "a")])
    data = json.loads((tmp_path / "fetch_index.json").read_text())
    assert len(data) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_all'`.

- [ ] **Step 3: Write the implementation** — add to `ingestion/fetch.py`

Add these imports at the top of the file, and extend the existing sources import:
```python
import hashlib
import json
import time
from datetime import datetime, timezone
# change the existing line "from ingestion.sources import DocSpec" to:
from ingestion.sources import DocSpec, Manifest
```

Append these functions to the end of the file:
```python
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_all(
    manifest: Manifest,
    data_dir: Path,
    fetch_fn: FetchFn,
    *,
    tenants: list[str] | None = None,
    delay: float = 0.0,
    now_fn: Callable[[], str] = now_iso,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[FetchRecord]:
    records: list[FetchRecord] = []
    selected = tenants or list(manifest.tenants.keys())
    for tenant_name in selected:
        tenant = manifest.tenants[tenant_name]
        for doc in tenant.documents:
            status, headers, body = fetch_fn(doc.url)
            ok, note = classify_response(doc, status, body)
            if ok:
                path = target_path(data_dir, tenant_name, doc)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            records.append(
                FetchRecord(
                    tenant=tenant_name,
                    id=doc.id,
                    url=doc.url,
                    type=doc.type,
                    format=doc.format,
                    corpus=doc.corpus,
                    citation=doc.citation,
                    source_version=doc.source_version,
                    fetched_date=now_fn(),
                    http_status=status,
                    content_type=headers.get("content-type"),
                    bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                    ok=ok,
                    note=note,
                )
            )
            if delay:
                sleep_fn(delay)
    return records


def write_index(data_dir: Path, records: list[FetchRecord]) -> Path:
    index_path = data_dir / "fetch_index.json"
    new_tenants = {r.tenant for r in records}
    existing: list[dict] = []
    if index_path.exists():
        existing = [
            e for e in json.loads(index_path.read_text(encoding="utf-8"))
            if e["tenant"] not in new_tenants
        ]
    combined = existing + [r.model_dump() for r in records]
    combined.sort(key=lambda e: (e["tenant"], e["id"]))
    data_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return index_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -v`
Expected: PASS (all fetch tests green).

- [ ] **Step 5: Commit**

```bash
git add ingestion/fetch.py tests/test_fetch.py
git commit -m "feat: fetch_all orchestration + merged provenance index"
```

---

### Task 5: CLI + real httpx client (`build_httpx_fetch_fn`, `main`)

**Files:**
- Modify: `ingestion/fetch.py` (add `build_httpx_fetch_fn`, `main`, `__main__` guard)
- Modify: `tests/test_fetch.py` (add a `main` integration test using an injected fake)

**Interfaces:**
- Consumes: `load_manifest`, `fetch_all`, `write_index`, `FetchFn`, `USER_AGENT`.
- Produces:
  - `build_httpx_fetch_fn(*, timeout: float = 30.0) -> FetchFn` — real client, follows redirects, sets User-Agent, returns lowercase-keyed headers.
  - `main(argv: list[str] | None = None, fetch_fn: FetchFn | None = None) -> int` — CLI (`--manifest`, `--data-dir`, `--tenant` repeatable, `--delay`); returns 0 iff every record is `ok`.
- Run entrypoint: `python -m ingestion.fetch`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_fetch.py`

```python
from ingestion.fetch import main

_MIN_MANIFEST = """
version: 1
defaults:
  corpus: true
tenants:
  aws-saa:
    display_name: "AWS SAA"
    license: "L"
    documents:
      - id: faq-ec2
        title: "EC2 FAQs"
        url: "https://x/ec2"
        type: faq
        format: html
        citation: "Amazon EC2 FAQs"
"""


def test_main_runs_with_injected_fetch(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(_MIN_MANIFEST, encoding="utf-8")
    fetch = _fake_fetch({"https://x/ec2": (200, {"content-type": "text/html"}, _HTML_OK)})
    rc = main(
        ["--manifest", str(manifest), "--data-dir", str(tmp_path / "data"), "--delay", "0"],
        fetch_fn=fetch,
    )
    assert rc == 0
    assert (tmp_path / "data" / "fetch_index.json").exists()
    assert (tmp_path / "data" / "aws-saa" / "faq-ec2.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fetch.py::test_main_runs_with_injected_fetch -v`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Write the implementation** — add to `ingestion/fetch.py`

Add `import argparse` to the imports, and change the sources import line to `from ingestion.sources import DocSpec, Manifest, load_manifest`. Append:
```python
def build_httpx_fetch_fn(*, timeout: float = 30.0) -> FetchFn:
    import httpx

    client = httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )

    def fetch(url: str) -> tuple[int, dict[str, str], bytes]:
        resp = client.get(url)
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status_code, headers, resp.content

    return fetch


def main(argv: list[str] | None = None, fetch_fn: FetchFn | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch CertCoach raw sources to disk.")
    parser.add_argument("--manifest", default="ingestion/sources.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tenant", action="append", dest="tenants",
                        help="restrict to a tenant (repeatable)")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    data_dir = Path(args.data_dir)
    records = fetch_all(
        manifest, data_dir, fetch_fn or build_httpx_fetch_fn(),
        tenants=args.tenants, delay=args.delay,
    )
    index_path = write_index(data_dir, records)

    ok = sum(1 for r in records if r.ok)
    print(f"fetched {ok}/{len(records)} ok -> {index_path}")
    for r in records:
        if not r.ok:
            print(f"  FAIL {r.tenant}/{r.id}: {r.note} ({r.url})")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full test suite to verify green**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS (all tests across all files).

- [ ] **Step 5: Commit**

```bash
git add ingestion/fetch.py tests/test_fetch.py
git commit -m "feat: fetch CLI + httpx client for ingestion"
```

---

### Task 6: Verify & fix the TODO URLs in `sources.yaml`

**Files:**
- Modify: `ingestion/sources.yaml`

**Interfaces:**
- Consumes: nothing (data-correctness task).
- Produces: a manifest whose URLs resolve to real first-party sources, or an honest note where none exists.

This task uses research tools (`WebSearch`, `WebFetch`; the connected Playwright MCP if a page needs JS) — **not** TDD. Every URL currently marked `# TODO: verify` must be confirmed against the live first-party vendor site.

- [ ] **Step 1: Enumerate the unverified URLs**

Run: `grep -n "TODO: verify" ingestion/sources.yaml`
Expected: the AWS SAA-C03 exam-guide PDF, AWS sample-questions PDF, GCP ACE exam guide, and GCP sample-questions entries.

- [ ] **Step 2: Resolve each URL from first-party sources only**

For each: use `WebSearch`/`WebFetch` to find the canonical AWS (`aws.amazon.com`, `d1.awsstatic.com`, `docs.aws.amazon.com`) or Google Cloud (`cloud.google.com`) URL. Confirm it returns the document (HTTP 200, correct content type). **Do not** substitute a third-party/dump source — if no stable first-party public URL exists, leave the entry and mark it `# UNRESOLVED: no stable first-party URL` and set `corpus: false` if it can't be fetched.

- [ ] **Step 3: Update the manifest** — replace each verified URL and delete its `# TODO: verify` marker. Add real IDs for the remaining `# TODO: add ...` FAQ/doc entries only if quick; otherwise leave the add-lists as-is (out of scope to exhaustively enumerate here).

- [ ] **Step 4: Sanity-check the manifest still parses**

Run: `.venv/bin/python -c "from ingestion.sources import load_manifest; m=load_manifest('ingestion/sources.yaml'); print(sum(len(t.documents) for t in m.tenants.values()), 'docs')"`
Expected: prints the document count with no validation error.

- [ ] **Step 5: Commit**

```bash
git add ingestion/sources.yaml
git commit -m "chore: verify and fix source URLs in ingestion manifest"
```

---

### Task 7: End-to-end fetch run + ingestion README

**Files:**
- Create: `ingestion/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a real populated `data/` (gitignored) + `data/fetch_index.json`; a committed README documenting how to run and an honest note on any sources that failed.

- [ ] **Step 1: Run the fetcher for real**

Run: `.venv/bin/python -m ingestion.fetch --delay 0.5`
Expected: prints `fetched N/M ok -> data/fetch_index.json`; any failures listed with their note.

- [ ] **Step 2: Inspect the provenance index**

Run: `.venv/bin/python -c "import json; d=json.load(open('data/fetch_index.json')); print(len(d),'records;', sum(e['ok'] for e in d),'ok'); [print('FAIL',e['tenant'],e['id'],e['note']) for e in d if not e['ok']]"`
Expected: record count; any `ok:false` rows printed with their note. Confirm a few content files exist under `data/<tenant>/`.

- [ ] **Step 3: Confirm `data/` is not tracked by git**

Run: `git status --porcelain data/ ; git check-ignore data/fetch_index.json`
Expected: no `data/` entries in status; `git check-ignore` prints `data/fetch_index.json` (i.e. ignored). Raw docs must never be committed (spec §8).

- [ ] **Step 4: Write `ingestion/README.md`**

```markdown
# Ingestion

The **fetch stage** of the CertCoach ingestion pipeline. Downloads raw source
documents listed in `sources.yaml` to `data/<tenant>/<id>.<ext>` and writes a
provenance index to `data/fetch_index.json`.

## Run

```bash
python -m ingestion.fetch                 # all tenants
python -m ingestion.fetch --tenant aws-saa  # one tenant
```

Options: `--manifest` (default `ingestion/sources.yaml`), `--data-dir`
(default `data`), `--tenant` (repeatable), `--delay` (seconds between requests).

## Notes

- **First-party sources only** — official AWS/GCP docs. No exam dumps (spec §8).
- `data/` is **gitignored**; fetched copyrighted docs are never committed.
- Failures (non-200, thin/JS-shell HTML) are recorded in `fetch_index.json`
  with `ok: false` and a note — not hidden.
- Next stage (not built yet): clean → normalize to **Markdown** → chunk → embed.
```

- [ ] **Step 5: Commit**

```bash
git add ingestion/README.md
git commit -m "docs: ingestion fetch README + verified end-to-end run"
```

---

## Notes for the executor

- All test/run commands assume the `.venv` from Task 1. If a fresh shell is used, either activate it (`source .venv/bin/activate`) or keep the explicit `.venv/bin/...` prefixes shown.
- `fetch_fn` returns headers with **lowercase keys**; the real `httpx` client normalizes them, and tests pass them lowercase.
- Task 6 requires network + research tools; Tasks 1–5 are fully offline and deterministic (no network in tests).
