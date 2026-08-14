# Coaching Layer, FastAPI, and Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `retrieve()` into a `generate()` LLM client and three prompt-driven coaching modes (`answer`, `quiz`, `grade`), expose them over FastAPI (`/ask`, `/quiz`, `/grade`, `/health`), and build an eval harness that measures retrieval quality and answer quality with a single `python -m certcoach.eval` command.

**Architecture:** `certcoach/generate.py` wraps the ai-core OpenAI-compatible gateway with a `generate(messages, model)` function and a `generate_answer / generate_quiz / generate_grade` convenience layer; `certcoach/prompts.py` holds versioned Jinja2 prompt templates; `certcoach/api.py` is the FastAPI app; `certcoach/eval.py` runs retrieval + answer evaluation and prints a metrics table to stdout (and optionally writes `eval_results.json`).

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Jinja2, psycopg2, existing `certcoach.retrieval`, `certcoach.ai_client`, `certcoach.db`.

## Global Constraints

- All models served by `ai-core` gateway (OpenAI-compatible). Default generation model: `claude-sonnet-4-6`. Eval judge: `gpt-5` (different family). Model IDs are gateway aliases — never standard Anthropic IDs.
- `AICORE_BASE_URL` and `AICORE_API_KEY` env vars. No Azure.
- Tenant passed via `X-Tenant` HTTP header. Valid values: `aws-saa`, `gcp-ace`.
- `retrieve()` signature: `retrieve(conn, query, tenant, *, dense_k=20, sparse_k=20, rerank_top_n=5, confidence_threshold=0.1) -> RetrievalResult`
- `RetrievalResult` fields: `chunks: list[ExpandedChunk]`, `confidence: float`, `below_threshold: bool`, `query: str`, `tenant: str`
- `ExpandedChunk` fields: `chunk: ChunkResult`, `prev_content: str | None`, `next_content: str | None`
- `ChunkResult.citation: str` — used in answer citations
- Confidence gate: if `result.below_threshold`, return a grounded "I don't have enough information" response — do NOT call the LLM generator.
- `generate()` must be patchable at `certcoach.generate.generate` for tests.
- Eval gold set files live at `eval/gold_retrieval.jsonl` and `eval/gold_qa.jsonl` (created in Task 5).
- `python -m certcoach.eval` emits a metrics table to stdout; exits 0 always (failures are metrics, not exceptions).
- No auth/accounts — YAGNI. No streaming. No fine-tuning.
- pytest, TDD where practical. All DB-dependent tests guarded with `pytest.mark.skipif(not _db_reachable(), ...)`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `certcoach/generate.py` | Create | `generate(messages, model)` → LLM via ai-core; `AnswerRequest`, `QuizRequest`, `GradeRequest` → coaching responses |
| `certcoach/prompts.py` | Create | Jinja2 prompt templates for `answer`, `quiz`, `grade` modes |
| `certcoach/api.py` | Create | FastAPI app: `/health`, `/ask`, `/quiz`, `/grade` |
| `certcoach/eval.py` | Create | Eval harness: retrieval metrics + answer eval + latency |
| `eval/gold_retrieval.jsonl` | Create | 20 hand-authored retrieval gold pairs (question → source_id) |
| `eval/gold_qa.jsonl` | Create | 20 hand-authored Q&A gold pairs (question + answer) |
| `tests/test_generate.py` | Create | Unit tests for generate layer (patched LLM) |
| `tests/test_api.py` | Create | FastAPI integration tests (patched retrieve + generate) |
| `tests/test_eval.py` | Create | Eval harness unit tests (patched retrieve + judge) |

---

### Task 1: `generate()` LLM client and coaching response layer

**Files:**
- Create: `certcoach/generate.py`
- Create: `certcoach/prompts.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `certcoach.ai_client._get_client()` (existing OpenAI client)
- Produces:
  - `generate(messages: list[dict], *, model: str = "claude-sonnet-4-6") -> str` — calls ai-core chat completions, returns assistant content string
  - `CoachingResponse` dataclass: `answer: str`, `citations: list[str]`, `below_threshold: bool`, `confidence: float`
  - `answer_question(result: RetrievalResult, *, model: str = "claude-sonnet-4-6") -> CoachingResponse`
  - `generate_quiz(topic: str, tenant: str, *, n: int = 5, model: str = "claude-sonnet-4-6") -> str` — returns raw LLM string (JSON quiz)
  - `grade_answer(question: str, learner_answer: str, reference: str, *, model: str = "claude-sonnet-4-6") -> str` — returns raw LLM string (JSON grade)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py
import pytest
from unittest.mock import patch, MagicMock


def _make_retrieval_result(below_threshold=False, confidence=0.85):
    from certcoach.retrieval import RetrievalResult, ExpandedChunk, ChunkResult
    chunk = ChunkResult(
        id=1, tenant="aws-saa", source_id="faq-ec2", chunk_index=0,
        content="EC2 instances provide resizable compute capacity.",
        token_count=8, citation="Amazon EC2 FAQs", doc_type="faq",
        source_date="2026-08-08", prev_chunk_id=None, next_chunk_id=None, score=0.9,
    )
    return RetrievalResult(
        chunks=[ExpandedChunk(chunk=chunk, prev_content=None, next_content=None)],
        confidence=confidence,
        below_threshold=below_threshold,
        query="What is EC2?",
        tenant="aws-saa",
    )


def test_generate_calls_llm_and_returns_string():
    from certcoach.generate import generate
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="test response"))]
    )
    with patch("certcoach.generate._get_client", return_value=mock_client):
        result = generate([{"role": "user", "content": "hello"}])
    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_respects_model_override():
    from certcoach.generate import generate
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch("certcoach.generate._get_client", return_value=mock_client):
        generate([{"role": "user", "content": "hi"}], model="gpt-5")
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-5"


def test_answer_question_returns_coaching_response():
    from certcoach.generate import answer_question, CoachingResponse
    result = _make_retrieval_result()
    with patch("certcoach.generate.generate", return_value="EC2 is a compute service."):
        response = answer_question(result)
    assert isinstance(response, CoachingResponse)
    assert "EC2" in response.answer
    assert response.citations == ["Amazon EC2 FAQs"]
    assert not response.below_threshold
    assert response.confidence == pytest.approx(0.85)


def test_answer_question_below_threshold_skips_generate():
    from certcoach.generate import answer_question, CoachingResponse
    result = _make_retrieval_result(below_threshold=True, confidence=0.04)
    with patch("certcoach.generate.generate") as mock_gen:
        response = answer_question(result)
    mock_gen.assert_not_called()
    assert response.below_threshold
    assert "not enough information" in response.answer.lower()
    assert response.citations == []


def test_generate_quiz_returns_string():
    from certcoach.generate import generate_quiz
    with patch("certcoach.generate.generate", return_value='[{"q": "Q1"}]') as mock_gen:
        result = generate_quiz("EC2 instance types", "aws-saa")
    assert result == '[{"q": "Q1"}]'
    mock_gen.assert_called_once()


def test_grade_answer_returns_string():
    from certcoach.generate import grade_answer
    with patch("certcoach.generate.generate", return_value='{"score": 0.8}') as mock_gen:
        result = grade_answer("What is EC2?", "A compute service", "EC2 is compute.")
    assert result == '{"score": 0.8}'
    mock_gen.assert_called_once()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_generate.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` for `certcoach.generate`.

- [ ] **Step 3: Create `certcoach/prompts.py`**

```python
# certcoach/prompts.py
from __future__ import annotations

ANSWER_SYSTEM = """\
You are CertCoach, an expert tutor for cloud certification exams.
Answer the learner's question using ONLY the context provided.
Cite each claim with the source name in parentheses, e.g. (Amazon EC2 FAQs).
If the context is insufficient, say so clearly — do not fabricate.
Be concise: aim for 3-5 sentences unless the question demands more.
"""

ANSWER_USER = """\
Question: {query}

Context:
{context}
"""

QUIZ_SYSTEM = """\
You are CertCoach. Generate {n} multiple-choice practice questions for the {tenant} certification exam.
Topic: {topic}
Return a JSON array. Each element: {{"question": str, "options": [str, str, str, str], "answer": str, "explanation": str}}.
Return ONLY the JSON array, no prose.
"""

GRADE_SYSTEM = """\
You are CertCoach. Grade the learner's answer to the question below.
Return JSON: {{"score": float (0.0-1.0), "correct": bool, "feedback": str, "gap": str}}.
Return ONLY the JSON object, no prose.
"""

GRADE_USER = """\
Question: {question}

Reference answer: {reference}

Learner's answer: {learner_answer}
"""


def build_answer_messages(query: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": ANSWER_USER.format(query=query, context=context)},
    ]


def build_quiz_messages(topic: str, tenant: str, n: int) -> list[dict]:
    return [
        {"role": "system", "content": QUIZ_SYSTEM.format(n=n, tenant=tenant, topic=topic)},
        {"role": "user", "content": f"Generate {n} questions about: {topic}"},
    ]


def build_grade_messages(question: str, learner_answer: str, reference: str) -> list[dict]:
    return [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user", "content": GRADE_USER.format(
            question=question, reference=reference, learner_answer=learner_answer
        )},
    ]


def format_context(chunks) -> str:
    """Format ExpandedChunk list into a context string for the LLM prompt."""
    parts = []
    for ec in chunks:
        text = ""
        if ec.prev_content:
            text += ec.prev_content + "\n\n"
        text += ec.chunk.content
        if ec.next_content:
            text += "\n\n" + ec.next_content
        parts.append(f"[{ec.chunk.citation}]\n{text.strip()}")
    return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Create `certcoach/generate.py`**

```python
# certcoach/generate.py
from __future__ import annotations
from dataclasses import dataclass, field

from certcoach.ai_client import _get_client
from certcoach.prompts import build_answer_messages, build_quiz_messages, build_grade_messages, format_context


@dataclass
class CoachingResponse:
    answer: str
    citations: list[str]
    below_threshold: bool
    confidence: float


def generate(messages: list[dict], *, model: str = "claude-sonnet-4-6") -> str:
    client = _get_client()
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def answer_question(result, *, model: str = "claude-sonnet-4-6") -> CoachingResponse:
    if result.below_threshold:
        return CoachingResponse(
            answer="I don't have enough information in the corpus to answer this confidently.",
            citations=[],
            below_threshold=True,
            confidence=result.confidence,
        )
    context = format_context(result.chunks)
    messages = build_answer_messages(result.query, context)
    answer_text = generate(messages, model=model)
    citations = list(dict.fromkeys(ec.chunk.citation for ec in result.chunks))
    return CoachingResponse(
        answer=answer_text,
        citations=citations,
        below_threshold=False,
        confidence=result.confidence,
    )


def generate_quiz(topic: str, tenant: str, *, n: int = 5, model: str = "claude-sonnet-4-6") -> str:
    messages = build_quiz_messages(topic, tenant, n)
    return generate(messages, model=model)


def grade_answer(
    question: str,
    learner_answer: str,
    reference: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> str:
    messages = build_grade_messages(question, learner_answer, reference)
    return generate(messages, model=model)
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
python -m pytest tests/test_generate.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add certcoach/generate.py certcoach/prompts.py tests/test_generate.py
git commit -m "feat: coaching layer — generate(), answer/quiz/grade, prompt templates"
```

---

### Task 2: FastAPI app — `/health`, `/ask`, `/quiz`, `/grade`

**Files:**
- Create: `certcoach/api.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `certcoach.generate.answer_question`, `certcoach.generate.generate_quiz`, `certcoach.generate.grade_answer`, `certcoach.retrieval.retrieve`, `certcoach.db.get_conn`, `certcoach.db.apply_migrations`
- Produces: FastAPI `app` object importable as `from certcoach.api import app`

**Request/Response schemas (Pydantic v2):**

```python
# /ask
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    below_threshold: bool

# /quiz
class QuizRequest(BaseModel):
    topic: str
    n: int = 5

class QuizResponse(BaseModel):
    questions: str  # raw JSON string from LLM

# /grade
class GradeRequest(BaseModel):
    question: str
    learner_answer: str
    reference: str

class GradeResponse(BaseModel):
    result: str  # raw JSON string from LLM

# /health
class HealthResponse(BaseModel):
    status: str  # "ok"
    db: str      # "ok" | "error"
```

Tenant comes from `X-Tenant` header. Valid: `aws-saa`, `gcp-ace`. Invalid → 400.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _make_retrieval_result(below_threshold=False):
    from certcoach.retrieval import RetrievalResult, ExpandedChunk, ChunkResult
    chunk = ChunkResult(
        id=1, tenant="aws-saa", source_id="faq-ec2", chunk_index=0,
        content="EC2 instances provide resizable compute capacity.",
        token_count=8, citation="Amazon EC2 FAQs", doc_type="faq",
        source_date=None, prev_chunk_id=None, next_chunk_id=None, score=0.9,
    )
    return RetrievalResult(
        chunks=[ExpandedChunk(chunk=chunk, prev_content=None, next_content=None)],
        confidence=0.85, below_threshold=below_threshold,
        query="What is EC2?", tenant="aws-saa",
    )


def _make_coaching_response(below_threshold=False):
    from certcoach.generate import CoachingResponse
    return CoachingResponse(
        answer="EC2 is a compute service. (Amazon EC2 FAQs)",
        citations=["Amazon EC2 FAQs"],
        below_threshold=below_threshold,
        confidence=0.85,
    )


def test_health_ok():
    from certcoach.api import app
    client = TestClient(app)
    with patch("certcoach.api.get_conn") as mock_conn:
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_returns_answer():
    from certcoach.api import app
    client = TestClient(app)
    with patch("certcoach.api.retrieve", return_value=_make_retrieval_result()), \
         patch("certcoach.api.answer_question", return_value=_make_coaching_response()):
        resp = client.post(
            "/ask",
            json={"question": "What is EC2?"},
            headers={"X-Tenant": "aws-saa"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "EC2" in body["answer"]
    assert body["citations"] == ["Amazon EC2 FAQs"]
    assert body["confidence"] == pytest.approx(0.85)
    assert not body["below_threshold"]


def test_ask_missing_tenant_header_returns_400():
    from certcoach.api import app
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "What is EC2?"})
    assert resp.status_code == 400


def test_ask_invalid_tenant_returns_400():
    from certcoach.api import app
    client = TestClient(app)
    resp = client.post(
        "/ask",
        json={"question": "What is EC2?"},
        headers={"X-Tenant": "invalid-tenant"},
    )
    assert resp.status_code == 400


def test_quiz_returns_questions():
    from certcoach.api import app
    client = TestClient(app)
    quiz_json = '[{"question": "Q1", "options": ["A","B","C","D"], "answer": "A", "explanation": "x"}]'
    with patch("certcoach.api.generate_quiz", return_value=quiz_json):
        resp = client.post(
            "/quiz",
            json={"topic": "EC2 instance types", "n": 1},
            headers={"X-Tenant": "aws-saa"},
        )
    assert resp.status_code == 200
    assert resp.json()["questions"] == quiz_json


def test_grade_returns_result():
    from certcoach.api import app
    client = TestClient(app)
    grade_json = '{"score": 0.9, "correct": true, "feedback": "Good", "gap": ""}'
    with patch("certcoach.api.grade_answer", return_value=grade_json):
        resp = client.post(
            "/grade",
            json={"question": "What is EC2?", "learner_answer": "compute", "reference": "EC2 is compute."},
            headers={"X-Tenant": "aws-saa"},
        )
    assert resp.status_code == 200
    assert resp.json()["result"] == grade_json
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_api.py -v
```
Expected: `ImportError` for `certcoach.api`.

- [ ] **Step 3: Implement `certcoach/api.py`**

```python
# certcoach/api.py
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from certcoach.db import get_conn, apply_migrations
from certcoach.retrieval import retrieve
from certcoach.generate import answer_question, generate_quiz, grade_answer

app = FastAPI(title="CertCoach", version="0.1.0")

_VALID_TENANTS = {"aws-saa", "gcp-ace"}


def _require_tenant(x_tenant: str | None) -> str:
    if not x_tenant or x_tenant not in _VALID_TENANTS:
        raise HTTPException(status_code=400, detail=f"X-Tenant must be one of {sorted(_VALID_TENANTS)}")
    return x_tenant


class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    below_threshold: bool

class QuizRequest(BaseModel):
    topic: str
    n: int = 5

class QuizResponse(BaseModel):
    questions: str

class GradeRequest(BaseModel):
    question: str
    learner_answer: str
    reference: str

class GradeResponse(BaseModel):
    result: str

class HealthResponse(BaseModel):
    status: str
    db: str


@app.get("/health", response_model=HealthResponse)
def health():
    db_status = "ok"
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
    except Exception:
        db_status = "error"
    return HealthResponse(status="ok", db=db_status)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_tenant: str | None = Header(default=None)):
    tenant = _require_tenant(x_tenant)
    conn = get_conn()
    try:
        result = retrieve(conn, req.question, tenant)
    finally:
        conn.close()
    response = answer_question(result)
    return AskResponse(
        answer=response.answer,
        citations=response.citations,
        confidence=response.confidence,
        below_threshold=response.below_threshold,
    )


@app.post("/quiz", response_model=QuizResponse)
def quiz(req: QuizRequest, x_tenant: str | None = Header(default=None)):
    tenant = _require_tenant(x_tenant)
    questions = generate_quiz(req.topic, tenant, n=req.n)
    return QuizResponse(questions=questions)


@app.post("/grade", response_model=GradeResponse)
def grade(req: GradeRequest, x_tenant: str | None = Header(default=None)):
    _require_tenant(x_tenant)
    result = grade_answer(req.question, req.learner_answer, req.reference)
    return GradeResponse(result=result)
```

- [ ] **Step 4: Add `fastapi` and `httpx` to `pyproject.toml` dependencies**

`pyproject.toml` `[project.dependencies]` must include:
```
"fastapi>=0.111",
"uvicorn>=0.29",
```
(`httpx` is already pulled in by `fastapi[all]` or via existing ingestion use.)

- [ ] **Step 5: Install and run tests**

```bash
pip install -e ".[dev]"
python -m pytest tests/test_api.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add certcoach/api.py tests/test_api.py pyproject.toml
git commit -m "feat: FastAPI app — /health, /ask, /quiz, /grade"
```

---

### Task 3: Eval gold sets

**Files:**
- Create: `eval/gold_retrieval.jsonl`
- Create: `eval/gold_qa.jsonl`

**Purpose:** Hand-authored pairs used by the eval harness in Task 4. 20 retrieval pairs + 20 Q&A pairs, sourced from official exam-guide objectives and official sample questions (first-party only — see spec §8).

**Format:**

`eval/gold_retrieval.jsonl` — one JSON object per line:
```json
{"question": "What is the maximum size of an S3 object?", "relevant_source_ids": ["faq-s3"], "tenant": "aws-saa"}
```

`eval/gold_qa.jsonl` — one JSON object per line:
```json
{"question": "What is the maximum size of an S3 object?", "reference_answer": "An individual Amazon S3 object can be as large as 5 TB.", "tenant": "aws-saa"}
```

**Provenance rules (enforced in the content below):**
- All questions derived from official AWS SAA-C03 Exam Guide objectives or official AWS documentation; no third-party crammer content.
- GCP ACE questions derived from official GCP ACE Exam Guide objectives or official GCP documentation.

- [ ] **Step 1: Create `eval/` directory and gold sets**

Create `eval/gold_retrieval.jsonl`:

```jsonl
{"question": "What storage classes does Amazon S3 offer for infrequently accessed data?", "relevant_source_ids": ["faq-s3"], "tenant": "aws-saa"}
{"question": "How does Amazon EC2 Auto Scaling work?", "relevant_source_ids": ["faq-ec2"], "tenant": "aws-saa"}
{"question": "What is the purpose of an Amazon VPC Internet Gateway?", "relevant_source_ids": ["faq-vpc"], "tenant": "aws-saa"}
{"question": "What are the five pillars of the AWS Well-Architected Framework?", "relevant_source_ids": ["well-architected-framework"], "tenant": "aws-saa"}
{"question": "What is the difference between EC2 On-Demand and Reserved Instances?", "relevant_source_ids": ["faq-ec2"], "tenant": "aws-saa"}
{"question": "How does S3 versioning protect against accidental deletion?", "relevant_source_ids": ["faq-s3"], "tenant": "aws-saa"}
{"question": "What is a VPC security group and how does it differ from a NACL?", "relevant_source_ids": ["faq-vpc"], "tenant": "aws-saa"}
{"question": "What AWS SAA-C03 exam domains cover resilient architectures?", "relevant_source_ids": ["saa-exam-guide"], "tenant": "aws-saa"}
{"question": "What are EC2 Spot Instances and when should you use them?", "relevant_source_ids": ["faq-ec2"], "tenant": "aws-saa"}
{"question": "How does S3 Transfer Acceleration work?", "relevant_source_ids": ["faq-s3"], "tenant": "aws-saa"}
{"question": "What is the shared responsibility model in AWS?", "relevant_source_ids": ["well-architected-framework"], "tenant": "aws-saa"}
{"question": "How do you enable cross-region replication in S3?", "relevant_source_ids": ["faq-s3"], "tenant": "aws-saa"}
{"question": "What is VPC peering and what are its limitations?", "relevant_source_ids": ["faq-vpc"], "tenant": "aws-saa"}
{"question": "What instance types are memory-optimised in EC2?", "relevant_source_ids": ["faq-ec2"], "tenant": "aws-saa"}
{"question": "What are the AWS Well-Architected security design principles?", "relevant_source_ids": ["well-architected-framework"], "tenant": "aws-saa"}
{"question": "How does Google Cloud IAM control access to resources?", "relevant_source_ids": ["doc-iam"], "tenant": "gcp-ace"}
{"question": "What is the difference between Compute Engine machine families?", "relevant_source_ids": ["doc-compute-engine"], "tenant": "gcp-ace"}
{"question": "What domains does the GCP Associate Cloud Engineer exam cover?", "relevant_source_ids": ["ace-exam-guide"], "tenant": "gcp-ace"}
{"question": "How does Google Cloud Storage organize data into buckets and objects?", "relevant_source_ids": ["doc-cloud-storage"], "tenant": "gcp-ace"}
{"question": "What is the Google Cloud Architecture Framework's reliability pillar?", "relevant_source_ids": ["architecture-framework"], "tenant": "gcp-ace"}
```

Create `eval/gold_qa.jsonl`:

```jsonl
{"question": "What is the maximum size of an individual Amazon S3 object?", "reference_answer": "An individual Amazon S3 object can be as large as 5 terabytes (5 TB). Objects larger than 5 GB should use multipart upload.", "tenant": "aws-saa"}
{"question": "What are the three types of EC2 instance purchasing options?", "reference_answer": "On-Demand Instances (pay per second/hour with no commitment), Reserved Instances (1- or 3-year commitment for a discount), and Spot Instances (bid on unused capacity at lower cost but can be interrupted).", "tenant": "aws-saa"}
{"question": "What is the purpose of an Amazon VPC?", "reference_answer": "An Amazon Virtual Private Cloud (VPC) lets you provision a logically isolated section of the AWS Cloud where you can launch AWS resources in a virtual network you define, with full control over IP ranges, subnets, route tables, and network gateways.", "tenant": "aws-saa"}
{"question": "How does S3 versioning work?", "reference_answer": "S3 versioning preserves, retrieves, and restores every version of every object stored in an S3 bucket. Once enabled, a DELETE operation adds a delete marker rather than permanently removing the object, allowing recovery.", "tenant": "aws-saa"}
{"question": "What is the difference between an S3 security group and a NACL?", "reference_answer": "Security groups are stateful firewalls at the instance/ENI level; return traffic is automatically allowed. Network ACLs (NACLs) are stateless firewalls at the subnet level; both inbound and outbound rules must explicitly allow return traffic.", "tenant": "aws-saa"}
{"question": "What are the five pillars of the AWS Well-Architected Framework?", "reference_answer": "Operational Excellence, Security, Reliability, Performance Efficiency, and Cost Optimization. A sixth pillar, Sustainability, was added later.", "tenant": "aws-saa"}
{"question": "When should you use EC2 Spot Instances?", "reference_answer": "Spot Instances are best for fault-tolerant, flexible workloads that can tolerate interruption: batch processing, data analysis, background jobs, and CI/CD. They offer up to 90% discount but can be reclaimed by AWS with a 2-minute warning.", "tenant": "aws-saa"}
{"question": "What is VPC peering?", "reference_answer": "VPC peering connects two VPCs to route traffic between them using private IPv4 or IPv6 addresses. Peering is non-transitive — traffic does not flow through a third VPC even if it is peered with both.", "tenant": "aws-saa"}
{"question": "What is the AWS shared responsibility model?", "reference_answer": "AWS is responsible for security OF the cloud (hardware, global infrastructure, managed services). Customers are responsible for security IN the cloud (data, IAM, OS patching on EC2, application security, network configuration).", "tenant": "aws-saa"}
{"question": "What is S3 Transfer Acceleration?", "reference_answer": "S3 Transfer Acceleration uses Amazon CloudFront edge locations to speed up uploads to S3. Data is routed over the optimised AWS backbone from the nearest edge to the S3 bucket, improving upload speeds for geographically distant users.", "tenant": "aws-saa"}
{"question": "How does Google Cloud IAM grant access?", "reference_answer": "Cloud IAM grants access by binding a principal (user, service account, or group) to a role on a resource. A role is a collection of permissions. Bindings can be set at org, folder, project, or resource level and are inherited down the hierarchy.", "tenant": "gcp-ace"}
{"question": "What is a Google Cloud service account?", "reference_answer": "A service account is a special identity for non-human principals (applications, VMs, or workloads). It is identified by an email address, and its permissions are controlled via IAM bindings. VMs use service accounts to call GCP APIs without embedding credentials.", "tenant": "gcp-ace"}
{"question": "What domains does the GCP Associate Cloud Engineer exam test?", "reference_answer": "Setting up a cloud solution environment; Planning and configuring a cloud solution; Deploying and implementing a cloud solution; Ensuring successful operation of a cloud solution; Configuring access and security.", "tenant": "gcp-ace"}
{"question": "How does Google Cloud Storage differ from a filesystem?", "reference_answer": "Cloud Storage is an object store, not a filesystem. Objects are stored in buckets and identified by keys. There is no true directory hierarchy (paths simulate it). Objects are immutable — overwriting creates a new version if versioning is enabled.", "tenant": "gcp-ace"}
{"question": "What is the Google Cloud Architecture Framework reliability pillar?", "reference_answer": "The reliability pillar covers designing for failure, fault tolerance, geographic distribution, automatic recovery, and testing resilience. Key practices include multi-region deployments, health checks, graceful degradation, and runbook documentation.", "tenant": "gcp-ace"}
{"question": "What is a Compute Engine preemptible VM?", "reference_answer": "A preemptible VM is a short-lived Compute Engine instance available at a much lower price (up to 80% discount). It can be reclaimed by Google at any time, so it is suitable for fault-tolerant batch and stateless workloads, not for long-running or stateful applications.", "tenant": "gcp-ace"}
{"question": "What is the purpose of a VPC firewall rule in Google Cloud?", "reference_answer": "VPC firewall rules control traffic to and from VM instances. Rules specify direction (ingress or egress), protocol, port, target (tags, service account, or IP range), and action (allow or deny). They are evaluated in priority order.", "tenant": "gcp-ace"}
{"question": "What is Google Cloud Pub/Sub?", "reference_answer": "Cloud Pub/Sub is a fully managed asynchronous messaging service. Publishers send messages to topics; subscribers receive messages from subscriptions. It supports at-least-once delivery, and is used to decouple services and ingest streaming data.", "tenant": "gcp-ace"}
{"question": "What is a Compute Engine managed instance group?", "reference_answer": "A managed instance group (MIG) creates and manages identical VMs from an instance template. MIGs support autoscaling, autohealing (replacing unhealthy instances), and rolling updates, enabling high availability and elastic capacity.", "tenant": "gcp-ace"}
{"question": "What are Cloud Storage object lifecycle management policies?", "reference_answer": "Lifecycle management policies define rules that automatically transition objects between storage classes (e.g. Standard → Nearline after 30 days) or delete objects after a specified age or version count, reducing storage costs without manual intervention.", "tenant": "gcp-ace"}
```

- [ ] **Step 2: Verify files are well-formed**

```bash
python -c "
import json
for f in ['eval/gold_retrieval.jsonl', 'eval/gold_qa.jsonl']:
    lines = open(f).readlines()
    for i, line in enumerate(lines, 1):
        obj = json.loads(line)
        assert 'question' in obj and 'tenant' in obj, f'{f}:{i} missing required keys'
    print(f'{f}: {len(lines)} records OK')
"
```
Expected: both files: 20 records OK.

- [ ] **Step 3: Commit**

```bash
mkdir -p eval
git add eval/gold_retrieval.jsonl eval/gold_qa.jsonl
git commit -m "eval: 20 retrieval + 20 Q&A gold pairs from official vendor docs"
```

---

### Task 4: Eval harness (`certcoach/eval.py`)

**Files:**
- Create: `certcoach/eval.py`
- Create: `tests/test_eval.py`

**Interfaces:**
- Consumes: `certcoach.retrieval.retrieve`, `certcoach.generate.generate`, `certcoach.db.get_conn`
- Produces: `python -m certcoach.eval` command; `eval_results.json` output file

**Metrics to compute:**

*Retrieval eval* (from `eval/gold_retrieval.jsonl`):
- `recall@5`: fraction of gold pairs where at least one `relevant_source_id` appears in top-5 retrieved `source_id`s
- `mrr@10`: mean reciprocal rank — for each query, `1/rank` of the first relevant source in the top-10; 0 if not found
- `hit_rate@1`: fraction where the top-1 result's `source_id` is in `relevant_source_ids`
- `ndcg@5`: normalized discounted cumulative gain at 5 (binary relevance: 1 if source_id in relevant_source_ids, else 0)

*Answer eval* (from `eval/gold_qa.jsonl`, requires LLM judge):
- `accuracy`: LLM-judge score (0–1) averaged across all Q&A pairs; judge model = `gpt-5`
- Judge prompt: system = "You are an objective evaluator. Score the candidate answer against the reference answer for factual accuracy on a 0.0–1.0 scale. Return JSON: {\"score\": float, \"rationale\": str}. Return ONLY JSON."
- User message: "Reference: {reference}\nCandidate: {candidate}"

*Latency* (microsecond-resolution `time.perf_counter`):
- Per-stage: `dense_ms`, `sparse_ms`, `rrf_ms`, `total_retrieval_ms` (median over retrieval gold set)
- Emit alongside quality metrics

**Output format (stdout):**

```
=== CertCoach Eval Results ===
Tenant: aws-saa   n=15
  Retrieval:  recall@5=0.867  hit@1=0.600  mrr@10=0.689  ndcg@5=0.801
  Answer:     accuracy=0.824  (n=10, judge=gpt-5)
  Latency:    retrieval_p50=312ms  retrieval_p95=487ms

Tenant: gcp-ace   n=5
  Retrieval:  recall@5=0.800  hit@1=0.600  mrr@10=0.700  ndcg@5=0.756
  Answer:     accuracy=0.810  (n=10, judge=gpt-5)
  Latency:    retrieval_p50=298ms  retrieval_p95=421ms
```

Also writes `eval_results.json` with full per-query details.

**Key implementation decisions:**
- `retrieve()` is called normally (live DB + live embed + live rerank) during eval; no stubs
- Answer eval calls `generate()` with the judge prompt; patches `certcoach.eval.generate` in tests
- `recall_at_k`, `mrr_at_k`, `hit_rate_at_k`, `ndcg_at_k` are pure functions — unit-testable without DB
- NDCG formula: `DCG = sum(rel_i / log2(i+2) for i in range(k))` where `rel_i ∈ {0,1}`; `IDCG = sum(1/log2(i+2) for i in range(min(k, num_relevant)))`; `NDCG = DCG/IDCG` (0 if IDCG=0)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval.py
import pytest
import math


def test_recall_at_k_hit():
    from certcoach.eval import recall_at_k
    retrieved = ["faq-ec2", "faq-s3", "faq-vpc"]
    relevant = {"faq-s3"}
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_at_k_miss():
    from certcoach.eval import recall_at_k
    retrieved = ["faq-ec2", "faq-ec2", "faq-ec2"]
    relevant = {"faq-s3"}
    assert recall_at_k(retrieved, relevant, k=5) == 0.0


def test_recall_at_k_respects_k():
    from certcoach.eval import recall_at_k
    retrieved = ["faq-ec2", "faq-ec2", "faq-s3"]  # relevant is at index 2
    relevant = {"faq-s3"}
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_mrr_at_k_first_rank():
    from certcoach.eval import mrr_at_k
    retrieved = ["faq-s3", "faq-ec2"]
    relevant = {"faq-s3"}
    assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(1.0)


def test_mrr_at_k_second_rank():
    from certcoach.eval import mrr_at_k
    retrieved = ["faq-ec2", "faq-s3"]
    relevant = {"faq-s3"}
    assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.5)


def test_mrr_at_k_miss():
    from certcoach.eval import mrr_at_k
    retrieved = ["faq-ec2"]
    relevant = {"faq-s3"}
    assert mrr_at_k(retrieved, relevant, k=10) == 0.0


def test_ndcg_at_k_perfect():
    from certcoach.eval import ndcg_at_k
    retrieved = ["faq-s3", "faq-ec2"]
    relevant = {"faq-s3"}
    # Perfect: relevant at rank 1; DCG = 1/log2(2) = 1.0; IDCG = 1.0
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)


def test_ndcg_at_k_second_rank():
    from certcoach.eval import ndcg_at_k
    retrieved = ["faq-ec2", "faq-s3"]
    relevant = {"faq-s3"}
    # DCG = 1/log2(3); IDCG = 1/log2(2)
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(expected)


def test_ndcg_at_k_empty_relevant():
    from certcoach.eval import ndcg_at_k
    retrieved = ["faq-ec2"]
    relevant = set()
    assert ndcg_at_k(retrieved, relevant, k=5) == 0.0


def test_judge_answer_parses_score():
    from certcoach.eval import judge_answer
    from unittest.mock import patch
    with patch("certcoach.eval.generate", return_value='{"score": 0.85, "rationale": "good"}'):
        score = judge_answer("What is EC2?", "EC2 is compute.", "EC2 is a compute service.")
    assert score == pytest.approx(0.85)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_eval.py -v
```
Expected: `ImportError` for `certcoach.eval`.

- [ ] **Step 3: Implement `certcoach/eval.py`**

```python
# certcoach/eval.py
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from certcoach.generate import generate  # module-level for patchability


# ── Pure metric functions ────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if retrieved[:k] and retrieved[0] in relevant else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, r in enumerate(retrieved[:k]) if r in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


# ── LLM judge ───────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an objective evaluator. Score the candidate answer against the reference "
    "answer for factual accuracy on a 0.0–1.0 scale. "
    'Return JSON: {"score": float, "rationale": str}. Return ONLY JSON.'
)


def judge_answer(question: str, candidate: str, reference: str, *, model: str = "gpt-5") -> float:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": f"Reference: {reference}\nCandidate: {candidate}"},
    ]
    raw = generate(messages, model=model)
    # Strip markdown code fences if present
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return float(json.loads(raw)["score"])


# ── Main eval runner ─────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval(
    gold_retrieval_path: Path = Path("eval/gold_retrieval.jsonl"),
    gold_qa_path: Path = Path("eval/gold_qa.jsonl"),
    output_path: Path = Path("eval_results.json"),
    judge_model: str = "gpt-5",
) -> dict:
    from certcoach.db import get_conn
    from certcoach.retrieval import retrieve, dense_search, sparse_search, rrf_fuse

    gold_ret = _load_jsonl(gold_retrieval_path)
    gold_qa = _load_jsonl(gold_qa_path)

    results_by_tenant: dict[str, dict] = {}
    all_ret_latencies: list[float] = []

    # ── Retrieval eval ────────────────────────────────────────────────────
    for item in gold_ret:
        tenant = item["tenant"]
        conn = get_conn()
        try:
            t0 = time.perf_counter()
            result = retrieve(conn, item["question"], tenant)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        finally:
            conn.close()

        all_ret_latencies.append(elapsed_ms)
        retrieved_sources = [ec.chunk.source_id for ec in result.chunks]
        relevant = set(item["relevant_source_ids"])

        bucket = results_by_tenant.setdefault(tenant, {
            "recall5": [], "hit1": [], "mrr10": [], "ndcg5": [],
            "accuracy": [], "latencies_ms": [], "ret_details": [], "qa_details": [],
        })
        bucket["recall5"].append(recall_at_k(retrieved_sources, relevant, k=5))
        bucket["hit1"].append(hit_rate_at_k(retrieved_sources, relevant, k=1))
        bucket["mrr10"].append(mrr_at_k(retrieved_sources, relevant, k=10))
        bucket["ndcg5"].append(ndcg_at_k(retrieved_sources, relevant, k=5))
        bucket["latencies_ms"].append(elapsed_ms)
        bucket["ret_details"].append({
            "question": item["question"],
            "relevant": list(relevant),
            "retrieved": retrieved_sources,
            "elapsed_ms": round(elapsed_ms, 1),
        })

    # ── Answer eval ───────────────────────────────────────────────────────
    for item in gold_qa:
        tenant = item["tenant"]
        conn = get_conn()
        try:
            result = retrieve(conn, item["question"], tenant)
        finally:
            conn.close()

        from certcoach.generate import answer_question
        response = answer_question(result)
        score = judge_answer(
            item["question"], response.answer, item["reference_answer"], model=judge_model
        )
        bucket = results_by_tenant.setdefault(tenant, {
            "recall5": [], "hit1": [], "mrr10": [], "ndcg5": [],
            "accuracy": [], "latencies_ms": [], "ret_details": [], "qa_details": [],
        })
        bucket["accuracy"].append(score)
        bucket["qa_details"].append({
            "question": item["question"],
            "reference": item["reference_answer"],
            "candidate": response.answer,
            "score": score,
        })

    # ── Print results ──────────────────────────────────────────────────────
    print("\n=== CertCoach Eval Results ===")
    output: dict = {}
    for tenant, b in sorted(results_by_tenant.items()):
        n_ret = len(b["recall5"])
        n_qa = len(b["accuracy"])
        lats = sorted(b["latencies_ms"])
        p50 = lats[len(lats) // 2] if lats else 0.0
        p95 = lats[int(len(lats) * 0.95)] if lats else 0.0

        def avg(xs): return sum(xs) / len(xs) if xs else float("nan")

        print(f"Tenant: {tenant}   n_retrieval={n_ret}  n_qa={n_qa}")
        print(f"  Retrieval:  recall@5={avg(b['recall5']):.3f}  hit@1={avg(b['hit1']):.3f}  "
              f"mrr@10={avg(b['mrr10']):.3f}  ndcg@5={avg(b['ndcg5']):.3f}")
        print(f"  Answer:     accuracy={avg(b['accuracy']):.3f}  (n={n_qa}, judge={judge_model})")
        print(f"  Latency:    retrieval_p50={p50:.0f}ms  retrieval_p95={p95:.0f}ms")
        print()
        output[tenant] = {
            "recall_at_5": avg(b["recall5"]),
            "hit_rate_at_1": avg(b["hit1"]),
            "mrr_at_10": avg(b["mrr10"]),
            "ndcg_at_5": avg(b["ndcg5"]),
            "answer_accuracy": avg(b["accuracy"]),
            "n_retrieval": n_ret,
            "n_qa": n_qa,
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "retrieval_details": b["ret_details"],
            "qa_details": b["qa_details"],
        }

    output_path.write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    run_eval()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_eval.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Verify `python -m certcoach.eval --help` doesn't crash (dry import)**

```bash
python -c "import certcoach.eval; print('import ok')"
```
Expected: `import ok`.

- [ ] **Step 6: Commit**

```bash
git add certcoach/eval.py tests/test_eval.py
git commit -m "feat: eval harness — retrieval metrics, LLM-as-judge, latency, eval_results.json"
```

---

### Task 5: Smoke test and wiring check

**Files:**
- Modify: `tests/test_smoke.py` (extend existing)

**Purpose:** Verify the full stack — from API request through retrieval to generation — works end-to-end with stub LLM calls but a real DB (when available). Also verify `python -m certcoach.eval` can be invoked and exits 0.

- [ ] **Step 1: Read `tests/test_smoke.py`** to understand the existing smoke test pattern.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_smoke.py`:

```python
import os
import pytest
from unittest.mock import patch, MagicMock

def _db_reachable():
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ.get("DATABASE_URL", ""))
        conn.close()
        return True
    except Exception:
        return False

_db_required = pytest.mark.skipif(not _db_reachable(), reason="Postgres not available")


def test_api_health_no_db():
    """Health endpoint returns ok even when DB is unreachable."""
    from certcoach.api import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    with patch("certcoach.api.get_conn", side_effect=Exception("no db")):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["db"] == "error"


@_db_required
def test_ask_end_to_end_stub_llm():
    """Full /ask path with real DB retrieval, stubbed embed+rerank+generate."""
    from certcoach.api import app
    from fastapi.testclient import TestClient
    from certcoach.ai_client import RerankResult

    client = TestClient(app)
    stub_embed = MagicMock(return_value=[[0.0] * 1536])
    stub_rerank = MagicMock(return_value=[RerankResult(index=0, score=0.75)])
    stub_generate = MagicMock(return_value="EC2 is a compute service. (Amazon EC2 FAQs)")

    with patch("certcoach.retrieval.embed", stub_embed), \
         patch("certcoach.retrieval.rerank", stub_rerank), \
         patch("certcoach.generate.generate", stub_generate):
        resp = client.post(
            "/ask",
            json={"question": "What is EC2?"},
            headers={"X-Tenant": "aws-saa"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["answer"], str)
    assert isinstance(body["citations"], list)
    assert "confidence" in body


def test_eval_module_importable():
    import certcoach.eval  # noqa: F401


def test_eval_pure_metrics_smoke():
    from certcoach.eval import recall_at_k, mrr_at_k, ndcg_at_k
    assert recall_at_k(["a", "b"], {"a"}, k=5) == 1.0
    assert mrr_at_k(["a", "b"], {"b"}, k=10) == pytest.approx(0.5)
    assert 0.0 <= ndcg_at_k(["a", "b"], {"b"}, k=5) <= 1.0
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_smoke.py -v
```
Expected: all non-DB tests pass; DB test skips cleanly if Postgres unavailable.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: smoke tests for API health, end-to-end /ask stub, eval import"
```
