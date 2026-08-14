from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


def _make_retrieval_result(below_threshold=False, confidence=0.85):
    from backend.retrieval import RetrievalResult, ExpandedChunk, ChunkResult
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
    from backend.generate import generate
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="test response"))]
    )
    with patch("backend.generate._get_client", return_value=mock_client):
        result = generate([{"role": "user", "content": "hello"}])
    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_generate_respects_model_override():
    from backend.generate import generate
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch("backend.generate._get_client", return_value=mock_client):
        generate([{"role": "user", "content": "hi"}], model="gpt-5")
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-5"


def test_answer_question_returns_coaching_response():
    from backend.generate import answer_question, CoachingResponse
    result = _make_retrieval_result()
    with patch("backend.generate.generate", return_value="EC2 is a compute service."):
        response = answer_question(result)
    assert isinstance(response, CoachingResponse)
    assert "EC2" in response.answer
    assert response.citations == ["Amazon EC2 FAQs"]
    assert not response.below_threshold
    assert response.confidence == pytest.approx(0.85)


def test_answer_question_below_threshold_skips_generate():
    from backend.generate import answer_question, CoachingResponse
    result = _make_retrieval_result(below_threshold=True, confidence=0.04)
    with patch("backend.generate.generate") as mock_gen:
        response = answer_question(result)
    mock_gen.assert_not_called()
    assert response.below_threshold
    assert "not enough information" in response.answer.lower()
    assert response.citations == []


def test_generate_quiz_returns_string():
    from backend.generate import generate_quiz
    with patch("backend.generate.generate", return_value='[{"q": "Q1"}]') as mock_gen:
        result = generate_quiz("EC2 instance types", "aws-saa")
    assert result == '[{"q": "Q1"}]'
    mock_gen.assert_called_once()


def test_grade_answer_returns_string():
    from backend.generate import grade_answer
    with patch("backend.generate.generate", return_value='{"score": 0.8}') as mock_gen:
        result = grade_answer("What is EC2?", "A compute service", "EC2 is compute.")
    assert result == '{"score": 0.8}'
    mock_gen.assert_called_once()


def test_answer_question_uses_prompt_version():
    from backend.generate import answer_question
    from backend.retrieval import RetrievalResult, ExpandedChunk, ChunkResult

    chunk = ChunkResult(
        id=1, tenant="aws-saa", source_id="faq-ec2", chunk_index=0,
        content="EC2 is compute.", token_count=5, citation="Amazon EC2 FAQs",
        doc_type="faq", source_date=None, prev_chunk_id=None, next_chunk_id=None, score=0.9,
    )
    result = RetrievalResult(
        chunks=[ExpandedChunk(chunk=chunk, prev_content=None, next_content=None)],
        confidence=0.9, below_threshold=False, query="What is EC2?", tenant="aws-saa",
    )
    captured = {}

    def fake_generate(messages, *, model):
        captured["system"] = messages[0]["content"]
        return "EC2 answer."

    with patch("backend.generate.generate", fake_generate):
        answer_question(result, prompt_version=2)

    from backend.prompts import ANSWER_SYSTEM_V2
    assert captured["system"] == ANSWER_SYSTEM_V2
