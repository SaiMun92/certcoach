import os
import pytest
from unittest.mock import patch, MagicMock


def test_ingestion_package_imports():
    import ingestion  # noqa: F401


# ── DB availability helper ────────────────────────────────────────────────────

def _db_reachable():
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ.get("DATABASE_URL", ""))
        conn.close()
        return True
    except Exception:
        return False


_db_required = pytest.mark.skipif(not _db_reachable(), reason="Postgres not available")


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_api_health_no_db():
    """Health endpoint returns ok even when DB is unreachable."""
    from backend.api import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    with patch("backend.api.get_conn", side_effect=Exception("no db")):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["db"] == "error"


@_db_required
def test_ask_end_to_end_stub_llm():
    """Full /ask path with real DB retrieval, stubbed embed+rerank+generate."""
    from backend.api import app
    from fastapi.testclient import TestClient
    from backend.ai_client import RerankResult

    client = TestClient(app)
    stub_embed = MagicMock(return_value=[[0.0] * 1536])
    stub_rerank = MagicMock(return_value=[RerankResult(index=0, score=0.75)])
    stub_generate = MagicMock(return_value="EC2 is a compute service. (Amazon EC2 FAQs)")

    with patch("backend.retrieval.embed", stub_embed), \
         patch("backend.retrieval.rerank", stub_rerank), \
         patch("backend.generate.generate", stub_generate):
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
    import backend.eval  # noqa: F401


def test_eval_pure_metrics_smoke():
    from backend.eval import recall_at_k, mrr_at_k, ndcg_at_k
    assert recall_at_k(["a", "b"], {"a"}, k=5) == 1.0
    assert mrr_at_k(["a", "b"], {"b"}, k=10) == pytest.approx(0.5)
    assert 0.0 <= ndcg_at_k(["a", "b"], {"b"}, k=5) <= 1.0
