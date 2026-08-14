from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracing_configured = False
_tracer_name = "certcoach"


def _setup_otel_provider(endpoint: str) -> None:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def configure_tracing() -> None:
    global _tracing_configured
    if _tracing_configured:
        return
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return
    from openinference.instrumentation.openai import OpenAIInstrumentor  # lazy
    _setup_otel_provider(endpoint)
    OpenAIInstrumentor().instrument()
    _tracing_configured = True


def _tracer() -> trace.Tracer:
    return trace.get_tracer(_tracer_name)


@contextmanager
def retrieval_span(
    query: str,
    tenant: str,
    retrieval_mode: str,
) -> Generator[trace.Span, None, None]:
    with _tracer().start_as_current_span("certcoach.retrieve") as span:
        span.set_attribute("certcoach.query", query)
        span.set_attribute("certcoach.tenant", tenant)
        span.set_attribute("certcoach.retrieval_mode", retrieval_mode)
        yield span


@contextmanager
def rerank_span(
    query: str,
    n_candidates: int,
) -> Generator[trace.Span, None, None]:
    with _tracer().start_as_current_span("certcoach.rerank") as span:
        span.set_attribute("certcoach.query", query)
        span.set_attribute("certcoach.rerank.n_candidates", n_candidates)
        yield span
