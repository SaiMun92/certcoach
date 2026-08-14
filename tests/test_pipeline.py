from __future__ import annotations
from pathlib import Path
import pytest


def _db_reachable() -> bool:
    try:
        from backend.db import get_conn
        c = get_conn(); c.close(); return True
    except Exception:
        return False


def _aicore_configured() -> bool:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.environ.get("AICORE_BASE_URL") and os.environ.get("AICORE_API_KEY"))


pytestmark = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres container not available"
)


def test_pipeline_single_doc(tmp_path):
    """Run pipeline on a tiny synthetic HTML doc — no real API call, use a stub embed."""
    from unittest.mock import patch
    from ingestion.pipeline import run_pipeline
    from ingestion.sources import Manifest, TenantSpec, DocSpec

    # Write a tiny HTML fixture
    html = "<html><body><h1>Test</h1><p>" + ("word " * 200) + "</p></body></html>"
    doc_path = tmp_path / "aws-saa" / "test-fixture.html"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(html, encoding="utf-8")

    stub_embed = lambda texts, **kw: [[0.0] * 1536 for _ in texts]

    with patch("ingestion.pipeline.embed", side_effect=stub_embed):
        result = run_pipeline(
            manifest_path="ingestion/sources.yaml",
            data_dir=tmp_path,
            tenants=["aws-saa"],
            _manifest_override=Manifest(
                version=1,
                tenants={
                    "aws-saa": TenantSpec(
                        display_name="AWS SAA",
                        license="test",
                        documents=[
                            DocSpec(
                                id="test-fixture",
                                title="Test Fixture",
                                url="https://example.com",
                                type="faq",
                                format="html",
                                citation="Test Fixture",
                                corpus=True,
                            )
                        ],
                    )
                },
            ),
        )
    assert "test-fixture" in result
    assert result["test-fixture"] > 0

    # Verify rows in DB
    from backend.db import get_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE source_id = 'test-fixture'")
            assert cur.fetchone()[0] == result["test-fixture"]
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_id = 'test-fixture'")
        conn.commit()
        conn.close()


def test_pipeline_skip_missing_file(tmp_path):
    """A doc whose file is absent triggers SKIP; the pipeline continues and processes the next doc."""
    from unittest.mock import patch
    from ingestion.pipeline import run_pipeline
    from ingestion.sources import Manifest, TenantSpec, DocSpec

    # Only the second doc has a real file; first doc is deliberately missing.
    html = "<html><body><h1>Test</h1><p>" + ("word " * 200) + "</p></body></html>"
    good_path = tmp_path / "aws-saa" / "good-doc.html"
    good_path.parent.mkdir(parents=True)
    good_path.write_text(html, encoding="utf-8")
    # "missing-doc.html" is intentionally never created.

    stub_embed = lambda texts, **kw: [[0.0] * 1536 for _ in texts]

    manifest = Manifest(
        version=1,
        tenants={
            "aws-saa": TenantSpec(
                display_name="AWS SAA",
                license="test",
                documents=[
                    DocSpec(
                        id="missing-doc",
                        title="Missing",
                        url="https://example.com/missing",
                        type="faq",
                        format="html",
                        citation="Missing",
                        corpus=True,
                    ),
                    DocSpec(
                        id="good-doc",
                        title="Good",
                        url="https://example.com/good",
                        type="faq",
                        format="html",
                        citation="Good",
                        corpus=True,
                    ),
                ],
            )
        },
    )

    with patch("ingestion.pipeline.embed", side_effect=stub_embed):
        result = run_pipeline(
            manifest_path="ingestion/sources.yaml",
            data_dir=tmp_path,
            tenants=["aws-saa"],
            _manifest_override=manifest,
        )

    # Failed doc must not appear in results (pipeline skipped it)
    assert "missing-doc" not in result
    # Pipeline continued and processed the good doc
    assert "good-doc" in result
    assert result["good-doc"] > 0

    # Cleanup DB rows inserted for good-doc
    from backend.db import get_conn
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_id = 'good-doc'")
        conn.commit()
    finally:
        conn.close()
