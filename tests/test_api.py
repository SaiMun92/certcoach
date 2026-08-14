import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _make_retrieval_result(below_threshold=False):
    from backend.retrieval import RetrievalResult, ExpandedChunk, ChunkResult
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
    from backend.generate import CoachingResponse
    return CoachingResponse(
        answer="EC2 is a compute service. (Amazon EC2 FAQs)",
        citations=["Amazon EC2 FAQs"],
        below_threshold=below_threshold,
        confidence=0.85,
    )


def test_health_ok():
    from backend.api import app
    client = TestClient(app)
    with patch("backend.api.get_conn") as mock_conn:
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_returns_answer():
    from backend.api import app
    client = TestClient(app)
    with patch("backend.api.get_conn") as mock_conn, \
         patch("backend.api.retrieve", return_value=_make_retrieval_result()), \
         patch("backend.api.answer_question", return_value=_make_coaching_response()):
        mock_conn.return_value = MagicMock()
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
    from backend.api import app
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "What is EC2?"})
    assert resp.status_code == 400


def test_ask_invalid_tenant_returns_400():
    from backend.api import app
    client = TestClient(app)
    resp = client.post(
        "/ask",
        json={"question": "What is EC2?"},
        headers={"X-Tenant": "invalid-tenant"},
    )
    assert resp.status_code == 400


def test_quiz_returns_questions():
    from backend.api import app
    client = TestClient(app)
    quiz_json = '[{"question": "Q1", "options": ["A","B","C","D"], "answer": "A", "explanation": "x"}]'
    with patch("backend.api.generate_quiz", return_value=quiz_json):
        resp = client.post(
            "/quiz",
            json={"topic": "EC2 instance types", "n": 1},
            headers={"X-Tenant": "aws-saa"},
        )
    assert resp.status_code == 200
    assert resp.json()["questions"] == quiz_json


def test_grade_returns_result():
    from backend.api import app
    client = TestClient(app)
    grade_json = '{"score": 0.9, "correct": true, "feedback": "Good", "gap": ""}'
    with patch("backend.api.grade_answer", return_value=grade_json):
        resp = client.post(
            "/grade",
            json={"question": "What is EC2?", "learner_answer": "compute", "reference": "EC2 is compute."},
            headers={"X-Tenant": "aws-saa"},
        )
    assert resp.status_code == 200
    assert resp.json()["result"] == grade_json
