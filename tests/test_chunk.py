from __future__ import annotations
import pytest


SHORT = "Hello world."
MEDIUM = ("AWS provides a suite of services. " * 50).strip()   # ~300 tokens
LONG = ("Amazon EC2 is a web service that provides resizable compute capacity. " * 100).strip()  # ~1400 tokens


def test_empty_returns_empty():
    from ingestion.chunk import chunk_text
    assert chunk_text("") == []


def test_short_text_is_one_chunk():
    import tiktoken
    from ingestion.chunk import chunk_text
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_text(SHORT)
    assert len(chunks) == 1
    assert chunks[0].content == SHORT
    assert chunks[0].chunk_index == 0
    # Verify token_count accuracy in early-return path
    assert chunks[0].token_count == len(enc.encode(SHORT))


def test_long_text_splits_into_multiple_chunks():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(LONG, target_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 3
    # Every chunk within bounds (last chunk may be small)
    for i, c in enumerate(chunks[:-1]):
        assert 20 <= c.token_count <= 260, f"chunk {i} token_count={c.token_count}"


def test_chunk_indices_are_sequential():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(LONG, target_tokens=200, overlap_tokens=20)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_overlap_content_appears_in_next_chunk():
    from ingestion.chunk import chunk_text
    chunks = chunk_text(LONG, target_tokens=200, overlap_tokens=50)
    if len(chunks) < 2:
        pytest.skip("text too short to produce overlap")
    # Last sentence of chunk 0 should appear somewhere in chunk 1
    tail = chunks[0].content.split()[-5:]
    tail_str = " ".join(tail)
    assert tail_str in chunks[1].content


def test_token_count_matches_encoding():
    import tiktoken
    from ingestion.chunk import chunk_text
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_text(MEDIUM, target_tokens=100, overlap_tokens=10)
    for c in chunks:
        actual = len(enc.encode(c.content))
        assert abs(actual - c.token_count) <= 2, (
            f"reported {c.token_count} but encoded {actual}"
        )


def test_multi_paragraph_no_content_lost():
    """Short paragraph followed by long paragraph — all content must be covered."""
    from ingestion.chunk import chunk_text
    short_para = "This is a short introduction.\n\n"
    long_para = ("Amazon EC2 provides resizable compute capacity in the cloud. " * 60).strip()
    text = short_para + long_para
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=50)
    assert len(chunks) >= 2, "long paragraph should produce multiple chunks"
    # Combined chunk content should cover the long paragraph (allowing for overlap)
    combined = " ".join(c.content for c in chunks)
    # The long paragraph's distinctive text must appear somewhere
    assert "Amazon EC2 provides resizable" in combined


def test_short_paragraph_then_long_no_truncation():
    """Paragraph trim must not drop the tail when the trimmed chunk is <= overlap_tokens."""
    import tiktoken
    from ingestion.chunk import chunk_text
    enc = tiktoken.get_encoding("cl100k_base")
    # ~60-token heading paragraph + long body with no internal blank lines
    heading = "Introduction to AWS EC2\n\n"
    body = ("word " * 400).strip()   # ~400 tokens, no paragraph breaks
    text = heading + body
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=100)
    total_source_tokens = len(enc.encode(text.strip()))
    # Total emitted tokens (minus overlap duplication) must cover the source.
    # A loose check: last chunk's content must contain text from the body's tail.
    tail_words = body.split()[-10:]
    tail_str = " ".join(tail_words)
    last_chunk_content = chunks[-1].content
    assert tail_str in last_chunk_content, (
        f"body tail not found in last chunk — {total_source_tokens} source tokens, "
        f"{len(chunks)} chunks produced, last chunk ends: {last_chunk_content[-100:]!r}"
    )


def test_paragraph_trim_does_not_drop_tail():
    """The 50–overlap_tokens paragraph length is the exact trigger for the old content-loss bug.

    First paragraph is 80 tokens (50 <= 80 <= overlap=100): the trim fires AND the
    old code's 'if chunk_len <= overlap_tokens: break' would have discarded the rest.
    Verify the body appears in the output.
    """
    from ingestion.chunk import chunk_text
    # ~80-token first paragraph (clears the >=50 gate, lands <= overlap=100)
    para1 = ("hello " * 80).strip() + "\n\n"
    body = ("Amazon EC2 provides resizable compute capacity. " * 50).strip()
    text = para1 + body
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=100)
    combined = " ".join(c.content for c in chunks)
    assert "Amazon EC2 provides resizable" in combined, (
        f"body content lost — only {len(chunks)} chunk(s) produced; "
        f"combined[:200]: {combined[:200]!r}"
    )
