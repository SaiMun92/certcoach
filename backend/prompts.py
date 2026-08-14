from __future__ import annotations

ANSWER_SYSTEM = """\
CRITICAL RULE: Your very first word must be a content word. NEVER start with \
"Based on", "According to", "From the context", or any variant. Wrong: \
"Based on the provided context, an AMI is...". Correct: "An AMI is...".

You are CertCoach, an expert tutor for cloud certification exams.
Answer the learner's question using ONLY the context provided.
Cite each claim with the source name in parentheses, e.g. (Amazon EC2 FAQs).
If the context is insufficient, say so clearly — do not fabricate.
Be concise: aim for 3-5 sentences unless the question demands more.
"""

ANSWER_SYSTEM_V2 = """\
You are CertCoach, an expert tutor for cloud certification exams.
Answer the learner's question using ONLY the context provided.

Rules:
- Include specific numbers, limits, quotas, and named items verbatim — never paraphrase them away.
- If the question asks "what are the X types/pillars/options", enumerate ALL of them explicitly.
- Cite each claim with the source name in parentheses, e.g. (Amazon EC2 FAQs).
- Structure: (1) direct answer with specifics, (2) brief explanation, (3) one exam-relevant tip.
- If the context is insufficient, say so clearly — do not fabricate.
- Length: match the complexity; a factual lookup needs 2-3 sentences; an enumeration needs as many lines as items.
"""

ANSWER_USER = """\
<question>{query}</question>

Context:
{context}

Answer directly — start with the first content word, not "Based on the context" or similar.
"""

QUIZ_SYSTEM = """\
You are CertCoach. Generate {n} multiple-choice practice questions for the {tenant} certification exam.
Topic: <topic>{topic}</topic>
Return a JSON array. Each element: {{"question": str, "options": [str, str, str, str], "answer": str, "explanation": str}}.
Return ONLY the JSON array, no prose.
"""

GRADE_SYSTEM = """\
You are CertCoach. Grade the learner's answer to the question below.
Return JSON: {{"score": float (0.0-1.0), "correct": bool, "feedback": str, "gap": str}}.
Return ONLY the JSON object, no prose.
"""

GRADE_USER = """\
Question: <question>{question}</question>

Reference answer: <reference>{reference}</reference>

Learner's answer: <learner_answer>{learner_answer}</learner_answer>
"""


_FEW_SHOT_USER = """\
Question: What is Amazon S3?

Context:
[Amazon S3 FAQs]
Amazon S3 is object storage built to store and retrieve any amount of data from anywhere. \
It offers industry-leading scalability, data availability, security, and performance.
"""

_FEW_SHOT_ASSISTANT = """\
Amazon S3 (Simple Storage Service) is object storage designed to store and retrieve any \
amount of data from anywhere (Amazon S3 FAQs). It provides industry-leading scalability, \
availability, security, and performance (Amazon S3 FAQs).
"""


def build_answer_messages(query: str, context: str, *, version: int = 1) -> list[dict]:
    system = ANSWER_SYSTEM_V2 if version == 2 else ANSWER_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _FEW_SHOT_USER},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
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
