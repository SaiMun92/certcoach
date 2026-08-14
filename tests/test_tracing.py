import os
import pytest
from unittest.mock import patch, MagicMock


def test_configure_tracing_noop_when_no_endpoint(monkeypatch):
    monkeypatch.setattr("backend.tracing._tracing_configured", False)
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    # Should not raise, should not attempt any network connection
    from backend.tracing import configure_tracing
    configure_tracing()  # no-op


def test_configure_tracing_registers_instrumentor_when_endpoint_set(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    mock_instrumentor = MagicMock()
    mock_instrumentor_class = MagicMock(return_value=mock_instrumentor)

    with patch("openinference.instrumentation.openai.OpenAIInstrumentor", mock_instrumentor_class), \
         patch("backend.tracing._setup_otel_provider"):
        from backend import tracing
        # Force re-execution by resetting the cached state
        tracing._tracing_configured = False
        tracing.configure_tracing()

    mock_instrumentor.instrument.assert_called_once()


def test_retrieval_span_returns_context_manager():
    from backend.tracing import retrieval_span
    with retrieval_span("what is EC2?", "aws-saa", "hybrid") as span:
        assert span is not None


def test_rerank_span_returns_context_manager():
    from backend.tracing import rerank_span
    with rerank_span("what is EC2?", n_candidates=20) as span:
        assert span is not None
