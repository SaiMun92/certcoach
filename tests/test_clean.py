from __future__ import annotations
import re
import pytest
from pathlib import Path


HTML_SIMPLE = """\
<html><body>
<nav>skip nav</nav>
<h1>EC2 FAQ</h1>
<script>alert(1)</script>
<p>What is <strong>EC2</strong>? It is a compute service.</p>
<footer>skip footer</footer>
</body></html>
"""

HTML_BLANK_LINES = "<html><body><p>a</p>\n\n\n\n\n<p>b</p></body></html>"


def test_clean_html_produces_markdown(tmp_path):
    from ingestion.clean import clean_doc
    f = tmp_path / "test.html"
    f.write_text(HTML_SIMPLE, encoding="utf-8")
    md = clean_doc(f, "html")
    assert "# EC2 FAQ" in md
    assert "**EC2**" in md
    assert "skip nav" not in md
    assert "skip footer" not in md
    assert "alert" not in md


def test_clean_html_collapses_blank_lines(tmp_path):
    from ingestion.clean import clean_doc
    f = tmp_path / "test.html"
    f.write_text(HTML_BLANK_LINES, encoding="utf-8")
    md = clean_doc(f, "html")
    assert "\n\n\n" not in md


def test_clean_pdf_returns_text(tmp_path):
    # Use the real fixture PDF from data/ if available, otherwise skip
    pdf = Path("data/aws-saa/saa-exam-guide.pdf")
    if not pdf.exists():
        pytest.skip("data/aws-saa/saa-exam-guide.pdf not fetched")
    from ingestion.clean import clean_doc
    text = clean_doc(pdf, "pdf")
    assert len(text) > 500
    assert isinstance(text, str)


def test_clean_unsupported_format_raises(tmp_path):
    from ingestion.clean import clean_doc
    f = tmp_path / "x.docx"
    f.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="unsupported format"):
        clean_doc(f, "docx")


def test_clean_missing_file_raises(tmp_path):
    from ingestion.clean import clean_doc
    with pytest.raises(FileNotFoundError):
        clean_doc(tmp_path / "nonexistent.html", "html")
