from __future__ import annotations

import logging
import os
import re

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from backend.db import get_conn, apply_migrations
from backend.retrieval import retrieve
from backend.generate import answer_question, generate_quiz, grade_answer
from backend.tracing import configure_tracing

logger = logging.getLogger(__name__)

# Patterns that are strong signals of prompt injection attempts
_INJECTION_PATTERNS = re.compile(
    r"ignore\b.{0,30}\b(instructions?|rules?|rubric|guidelines?|context|prompt|format)\b"
    r"|forget\b.{0,20}\b(everything|instructions?|rules?|context|prompt)"
    r"|you are now\b"
    r"|new (system )?prompt\b"
    r"|disregard\b.{0,30}\b(rules?|instructions?|guidelines?|context)"
    r"|reveal\b.{0,20}\b(system )?(prompt|instructions?)"
    r"|print\b.{0,60}\b(prompt|instructions?|system message|system prompt)"
    r"|act as (if you have no|a )",
    re.IGNORECASE,
)

configure_tracing()

app = FastAPI(title="CertCoach", version="0.1.0")

if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)

_VALID_TENANTS = {"aws-saa", "gcp-ace", "terraform-assoc"}


@app.on_event("startup")
def startup():
    conn = get_conn()
    apply_migrations(conn)
    conn.close()


def _require_tenant(x_tenant: str | None) -> str:
    if not x_tenant or x_tenant not in _VALID_TENANTS:
        raise HTTPException(status_code=400, detail=f"X-Tenant must be one of {sorted(_VALID_TENANTS)}")
    return x_tenant


def _flag_injection(field: str, value: str) -> None:
    if _INJECTION_PATTERNS.search(value):
        logger.warning("prompt_injection_attempt field=%s snippet=%.120r", field, value)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    below_threshold: bool

class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    n: int = Field(default=5, ge=1, le=20)

class QuizResponse(BaseModel):
    questions: str

class GradeRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    learner_answer: str = Field(..., min_length=1, max_length=4000)
    reference: str = Field(..., min_length=1, max_length=4000)

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
    _flag_injection("question", req.question)
    conn = get_conn()
    try:
        result = retrieve(conn, req.question, tenant, retrieval_mode="hybrid_no_rerank")
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
    _flag_injection("topic", req.topic)
    questions = generate_quiz(req.topic, tenant, n=req.n)
    return QuizResponse(questions=questions)


@app.post("/grade", response_model=GradeResponse)
def grade(req: GradeRequest, x_tenant: str | None = Header(default=None)):
    _require_tenant(x_tenant)
    _flag_injection("learner_answer", req.learner_answer)
    result = grade_answer(req.question, req.learner_answer, req.reference)
    return GradeResponse(result=result)
