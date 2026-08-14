import pytest
import math


def test_recall_at_k_hit():
    from backend.eval import recall_at_k
    retrieved = ["faq-ec2", "faq-s3", "faq-vpc"]
    relevant = {"faq-s3"}
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_at_k_miss():
    from backend.eval import recall_at_k
    retrieved = ["faq-ec2", "faq-ec2", "faq-ec2"]
    relevant = {"faq-s3"}
    assert recall_at_k(retrieved, relevant, k=5) == 0.0


def test_recall_at_k_respects_k():
    from backend.eval import recall_at_k
    retrieved = ["faq-ec2", "faq-ec2", "faq-s3"]  # relevant is at index 2
    relevant = {"faq-s3"}
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_mrr_at_k_first_rank():
    from backend.eval import mrr_at_k
    retrieved = ["faq-s3", "faq-ec2"]
    relevant = {"faq-s3"}
    assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(1.0)


def test_mrr_at_k_second_rank():
    from backend.eval import mrr_at_k
    retrieved = ["faq-ec2", "faq-s3"]
    relevant = {"faq-s3"}
    assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.5)


def test_mrr_at_k_miss():
    from backend.eval import mrr_at_k
    retrieved = ["faq-ec2"]
    relevant = {"faq-s3"}
    assert mrr_at_k(retrieved, relevant, k=10) == 0.0


def test_ndcg_at_k_perfect():
    from backend.eval import ndcg_at_k
    retrieved = ["faq-s3", "faq-ec2"]
    relevant = {"faq-s3"}
    # Perfect: relevant at rank 1; DCG = 1/log2(2) = 1.0; IDCG = 1.0
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)


def test_ndcg_at_k_second_rank():
    from backend.eval import ndcg_at_k
    retrieved = ["faq-ec2", "faq-s3"]
    relevant = {"faq-s3"}
    # DCG = 1/log2(3); IDCG = 1/log2(2)
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(expected)


def test_ndcg_at_k_empty_relevant():
    from backend.eval import ndcg_at_k
    retrieved = ["faq-ec2"]
    relevant = set()
    assert ndcg_at_k(retrieved, relevant, k=5) == 0.0


def test_judge_answer_parses_score():
    from backend.eval import judge_answer
    from unittest.mock import patch
    with patch("backend.eval.generate", return_value='{"score": 0.85, "rationale": "good"}'):
        score = judge_answer("What is EC2?", "EC2 is compute.", "EC2 is a compute service.")
    assert score == pytest.approx(0.85)


def test_run_eval_passes_prompt_version(tmp_path):
    """run_eval forwards prompt_version to answer_question."""
    import backend.eval as ev
    from backend.retrieval import RetrievalResult, ExpandedChunk, ChunkResult
    from backend.generate import CoachingResponse
    from unittest.mock import patch, MagicMock
    import json

    # Write minimal gold files
    ret_path = tmp_path / "ret.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    ret_path.write_text(json.dumps({"question": "q", "relevant_source_ids": ["s"], "tenant": "aws-saa"}) + "\n")
    qa_path.write_text(json.dumps({"question": "q", "reference_answer": "a", "tenant": "aws-saa"}) + "\n")

    chunk = ChunkResult(
        id=1, tenant="aws-saa", source_id="s", chunk_index=0,
        content="text", token_count=5, citation="Src", doc_type="faq",
        source_date=None, prev_chunk_id=None, next_chunk_id=None, score=0.9,
    )
    ret_result = RetrievalResult(
        chunks=[ExpandedChunk(chunk=chunk, prev_content=None, next_content=None)],
        confidence=0.9, below_threshold=False, query="q", tenant="aws-saa",
    )
    coaching = CoachingResponse(answer="ans", citations=["Src"], below_threshold=False, confidence=0.9)

    captured_versions = []

    def fake_answer(result, *, model="claude-sonnet-4-6", prompt_version=1):
        captured_versions.append(prompt_version)
        return coaching

    with patch.object(ev, "retrieve", return_value=ret_result), \
         patch.object(ev, "get_conn", return_value=MagicMock()), \
         patch.object(ev, "embed", side_effect=[[[0.1] * 1536], [[0.1] * 1536]]), \
         patch.object(ev, "judge_answer", return_value=0.9), \
         patch.object(ev, "answer_question", fake_answer):
        ev.run_eval(str(ret_path), str(qa_path), prompt_version=2,
                    output_path=str(tmp_path / "out.json"))

    assert captured_versions == [2]
