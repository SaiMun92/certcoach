import pytest
from pydantic import ValidationError

from ingestion.sources import Manifest, load_manifest

MANIFEST_YAML = """
version: 1
defaults:
  fetched_date: null
  corpus: true
tenants:
  aws-saa:
    display_name: "AWS SAA"
    license: "test license"
    documents:
      - id: faq-ec2
        title: "EC2 FAQs"
        url: "https://aws.amazon.com/ec2/faqs/"
        type: faq
        format: html
        citation: "Amazon EC2 FAQs"
      - id: saa-sample-questions
        title: "Sample Questions"
        url: "https://example.com/sample.pdf"
        type: sample_questions
        format: pdf
        corpus: false
        citation: "AWS Sample Questions"
"""


def _write(tmp_path, text):
    p = tmp_path / "sources.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_manifest_parses_tenants_and_docs(tmp_path):
    m = load_manifest(_write(tmp_path, MANIFEST_YAML))
    assert isinstance(m, Manifest)
    assert m.version == 1
    aws = m.tenants["aws-saa"]
    assert aws.display_name == "AWS SAA"
    assert len(aws.documents) == 2


def test_corpus_defaults_true_and_override(tmp_path):
    docs = {d.id: d for d in load_manifest(_write(tmp_path, MANIFEST_YAML)).tenants["aws-saa"].documents}
    assert docs["faq-ec2"].corpus is True            # inherited from defaults
    assert docs["saa-sample-questions"].corpus is False  # explicit override


def test_invalid_format_raises(tmp_path):
    bad = MANIFEST_YAML.replace("format: html", "format: docx")
    with pytest.raises(ValidationError):
        load_manifest(_write(tmp_path, bad))
