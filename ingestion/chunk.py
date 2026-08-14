from __future__ import annotations
import re
from dataclasses import dataclass

import tiktoken


@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int


_PARA_RE = re.compile(r"\n{2,}")


def chunk_text(
    text: str,
    *,
    target_tokens: int = 650,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    if not text.strip():
        return []
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    total = len(tokens)

    # If text fits in target_tokens, return as single chunk
    if total <= target_tokens:
        stripped_content = text.strip()
        return [Chunk(
            content=stripped_content,
            chunk_index=0,
            token_count=len(enc.encode(stripped_content)),
        )]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < total:
        end = min(start + target_tokens, total)

        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)

        # Try to break at last paragraph boundary within the chunk
        para_match = list(_PARA_RE.finditer(chunk_text_str))
        if para_match and end < total:
            last_para = para_match[-1]
            # Only break at paragraph if at least 50 tokens remain after start
            para_end_tokens = len(enc.encode(chunk_text_str[: last_para.end()]))
            if para_end_tokens >= 50:
                chunk_tokens = enc.encode(chunk_text_str[: last_para.end()])
                chunk_text_str = enc.decode(chunk_tokens)

        content = chunk_text_str.strip()
        if content:
            chunk_token_count = len(enc.encode(content))
            chunks.append(Chunk(
                content=content,
                chunk_index=idx,
                token_count=chunk_token_count,
            ))
            idx += 1

        # Advance based on the untrimmed window so a paragraph trim cannot
        # stall forward progress. Break only when the remaining tail already
        # fits within the overlap (it has been fully captured in this chunk).
        untrimmed_len = end - start
        if untrimmed_len <= overlap_tokens:
            break
        advance = max(untrimmed_len - overlap_tokens, 1)
        start += advance

    return chunks
