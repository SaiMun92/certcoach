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
    from backend.ai_client import embed
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock = _make_mock_client(vectors)
    with patch("backend.ai_client._get_client", return_value=mock):
        result = embed(["hello", "world"], dimensions=3)
    assert result == vectors
    mock.embeddings.create.assert_called_once_with(
        input=["hello", "world"],
        model="text-embedding-3-large",
        dimensions=3,
    )


def test_embed_empty_raises():
    from backend.ai_client import embed
    with pytest.raises(ValueError, match="empty"):
        embed([])


def test_embed_default_dimensions():
    from backend.ai_client import embed
    mock = _make_mock_client([[0.0] * 1536])
    with patch("backend.ai_client._get_client", return_value=mock):
        result = embed(["text"])
    assert len(result[0]) == 1536
    mock.embeddings.create.assert_called_once_with(
        input=["text"],
        model="text-embedding-3-large",
        dimensions=1536,
    )


def test_rerank_returns_results(monkeypatch):
    from unittest.mock import patch, MagicMock
    from backend.ai_client import rerank, RerankResult

    monkeypatch.setenv("AICORE_DIRECT_BASE_URL", "https://aicore.example.com")
    monkeypatch.setenv("AICORE_AUTH_URL", "https://auth.example.com")
    monkeypatch.setenv("AICORE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "test-secret")

    mock_token_resp = MagicMock()
    mock_token_resp.json.return_value = {"access_token": "test-token", "expires_in": 3600}
    mock_token_resp.raise_for_status = MagicMock()

    mock_rerank_resp = MagicMock()
    mock_rerank_resp.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]
    }
    mock_rerank_resp.raise_for_status = MagicMock()

    with patch("httpx.post", side_effect=[mock_token_resp, mock_rerank_resp]) as mock_post:
        # Clear cached token so the mock is used
        from backend import ai_client
        ai_client._token_cache.clear()
        results = rerank("what is EC2?", ["about storage", "about compute"])

    assert len(results) == 2
    assert results[0] == RerankResult(index=1, score=0.9)
    assert results[1] == RerankResult(index=0, score=0.3)
    rerank_call = mock_post.call_args_list[1]
    assert rerank_call.kwargs["json"]["query"] == "what is EC2?"
    assert rerank_call.kwargs["json"]["documents"] == ["about storage", "about compute"]
    assert rerank_call.kwargs["json"]["top_n"] == 5


def test_rerank_empty_documents_returns_empty(monkeypatch):
    from backend.ai_client import rerank
    from unittest.mock import patch

    monkeypatch.setenv("AICORE_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("AICORE_API_KEY", "test-key")

    with patch("httpx.post") as mock_post:
        result = rerank("query", [])
    assert result == []
    mock_post.assert_not_called()


def test_rerank_uses_correct_url_and_auth(monkeypatch):
    from unittest.mock import patch, MagicMock
    from backend.ai_client import rerank
    from backend import ai_client

    monkeypatch.setenv("AICORE_DIRECT_BASE_URL", "https://aicore.example.com")
    monkeypatch.setenv("AICORE_AUTH_URL", "https://auth.example.com")
    monkeypatch.setenv("AICORE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "my-group")

    mock_token_resp = MagicMock()
    mock_token_resp.json.return_value = {"access_token": "tok-xyz", "expires_in": 3600}
    mock_token_resp.raise_for_status = MagicMock()

    mock_rerank_resp = MagicMock()
    mock_rerank_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.7}]}
    mock_rerank_resp.raise_for_status = MagicMock()

    ai_client._token_cache.clear()
    with patch("httpx.post", side_effect=[mock_token_resp, mock_rerank_resp]) as mock_post:
        rerank("q", ["doc1"], top_n=1)

    rerank_call = mock_post.call_args_list[1]
    assert rerank_call.args[0] == "https://aicore.example.com/v2/inference/deployments/d7ebe1fa9a08238f/rerank"
    assert rerank_call.kwargs["headers"]["Authorization"] == "Bearer tok-xyz"
    assert rerank_call.kwargs["headers"]["AI-Resource-Group"] == "my-group"
