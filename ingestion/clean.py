from __future__ import annotations
import re
from pathlib import Path

_BLANK_RE = re.compile(r"\n{3,}")
_STRIP_TAGS = ["script", "style", "nav", "footer", "header"]


def _strip_html_tags(html: str, tags: list[str]) -> str:
    for tag in tags:
        html = re.sub(rf"(?is)<{tag}[^>]*>.*?</{tag}>", "", html)
    return html


def clean_doc(path: Path, fmt: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if fmt == "html":
        import markdownify
        html = path.read_text(encoding="utf-8", errors="replace")
        html = _strip_html_tags(html, _STRIP_TAGS)
        md = markdownify.markdownify(html, heading_style="ATX")
        md = _BLANK_RE.sub("\n\n", md)
        return md.strip()
    if fmt == "pdf":
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    raise ValueError(f"unsupported format: {fmt!r}")
