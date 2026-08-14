from __future__ import annotations
from dataclasses import dataclass

from backend.ai_client import _get_client
from backend.prompts import build_answer_messages, build_quiz_messages, build_grade_messages, format_context


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


def answer_question(
    result,
    *,
    model: str = "claude-sonnet-4-6",
    prompt_version: int = 1,
) -> CoachingResponse:
    if result.below_threshold:
        return CoachingResponse(
            answer="There is not enough information in the corpus to answer this confidently.",
            citations=[],
            below_threshold=True,
            confidence=result.confidence,
        )
    context = format_context(result.chunks)
    messages = build_answer_messages(result.query, context, version=prompt_version)
    answer_text = generate(messages, model=model)
    citations = list(dict.fromkeys(ec.chunk.citation for ec in result.chunks))
    return CoachingResponse(
        answer=answer_text or "",
        citations=citations,
        below_threshold=False,
        confidence=result.confidence,
    )


def _strip_code_fence(text: str) -> str:
    """Remove optional ```lang ... ``` wrapper that models sometimes add."""
    import re
    m = re.match(r"^```[a-zA-Z]*\n?(.*?)```\s*$", text.strip(), re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def generate_quiz(topic: str, tenant: str, *, n: int = 5, model: str = "claude-sonnet-4-6") -> str:
    messages = build_quiz_messages(topic, tenant, n)
    return _strip_code_fence(generate(messages, model=model))


def grade_answer(
    question: str,
    learner_answer: str,
    reference: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> str:
    messages = build_grade_messages(question, learner_answer, reference)
    return generate(messages, model=model)
