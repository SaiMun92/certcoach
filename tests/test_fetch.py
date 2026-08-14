import json
import pytest
from pathlib import Path

from ingestion.sources import DocSpec, Manifest, TenantSpec
from ingestion.fetch import classify_response, target_path, FetchRecord, fetch_all, main, write_index, build_httpx_fetch_fn, USER_AGENT


def _doc(**kw):
    base = dict(
        id="faq-ec2", title="EC2 FAQs", url="https://aws.amazon.com/ec2/faqs/",
        type="faq", format="html", citation="Amazon EC2 FAQs",
        corpus=True, source_version=None,
    )
    base.update(kw)
    return DocSpec(**base)


def test_target_path_uses_tenant_and_format():
    doc = _doc(id="faq-s3", format="html")
    assert target_path(Path("/data"), "aws-saa", doc) == Path("/data/aws-saa/faq-s3.html")


def test_classify_response_ok_html():
    body = ("<html><body>" + "word " * 300 + "</body></html>").encode()
    assert classify_response(_doc(format="html"), 200, body) == (True, "ok")


def test_classify_response_non_200():
    ok, note = classify_response(_doc(), 404, b"")
    assert ok is False and "404" in note


def test_classify_response_thin_html():
    ok, note = classify_response(_doc(format="html"), 200, b"<html></html>")
    assert ok is False and "thin" in note


def test_classify_response_pdf_ok():
    assert classify_response(_doc(format="pdf"), 200, b"%PDF-1.7 body bytes")[0] is True


def test_classify_response_accepts_real_pdf():
    assert classify_response(_doc(format="pdf"), 200, b"%PDF-1.7\nsome content") == (True, "ok")


def test_classify_response_rejects_non_pdf_body():
    ok, note = classify_response(_doc(format="pdf"), 200, b"<html>Not found</html>")
    assert ok is False and "pdf" in note.lower()


def _manifest(**tenants):
    return Manifest(version=1, tenants=tenants)


def _tenant(*docs):
    return TenantSpec(display_name="T", license="L", documents=list(docs))


def _fake_fetch(responses):
    def fetch(url):
        return responses[url]
    return fetch


_HTML_OK = ("<html><body>" + "word " * 300 + "</body></html>").encode()


def test_fetch_all_writes_ok_file_and_record(tmp_path):
    doc = _doc(url="https://x/ec2")
    m = _manifest(**{"aws-saa": _tenant(doc)})
    fetch = _fake_fetch({"https://x/ec2": (200, {"content-type": "text/html"}, _HTML_OK)})
    records = fetch_all(m, tmp_path, fetch, now_fn=lambda: "2026-07-23T00:00:00+00:00")
    assert (tmp_path / "aws-saa" / "faq-ec2.html").read_bytes() == _HTML_OK
    assert len(records) == 1
    r = records[0]
    assert r.ok and r.bytes == len(_HTML_OK) and len(r.sha256) == 64
    assert r.content_type == "text/html"
    assert r.fetched_date == "2026-07-23T00:00:00+00:00"


def test_fetch_all_skips_file_on_failure(tmp_path):
    doc = _doc(url="https://x/ec2")
    m = _manifest(**{"aws-saa": _tenant(doc)})
    fetch = _fake_fetch({"https://x/ec2": (404, {}, b"")})
    records = fetch_all(m, tmp_path, fetch)
    assert not (tmp_path / "aws-saa" / "faq-ec2.html").exists()
    assert records[0].ok is False and "404" in records[0].note
    assert records[0].content_type is None


def test_fetch_all_tenant_scoping(tmp_path):
    a = _doc(id="a", url="https://x/a")
    b = _doc(id="b", url="https://x/b")
    m = _manifest(**{"aws-saa": _tenant(a), "gcp-ace": _tenant(b)})
    fetch = _fake_fetch({
        "https://x/a": (200, {"content-type": "text/html"}, _HTML_OK),
        "https://x/b": (200, {"content-type": "text/html"}, _HTML_OK),
    })
    records = fetch_all(m, tmp_path, fetch, tenants=["aws-saa"])
    assert {r.tenant for r in records} == {"aws-saa"}


def test_fetch_all_unknown_tenant_raises(tmp_path):
    doc = _doc(url="https://x/ec2")
    m = _manifest(**{"aws-saa": _tenant(doc)})
    fetch = _fake_fetch({"https://x/ec2": (200, {"content-type": "text/html"}, _HTML_OK)})
    with pytest.raises(ValueError):
        fetch_all(m, tmp_path, fetch, tenants=["nope"])


def test_fetch_all_calls_sleep_fn_per_doc(tmp_path):
    a = _doc(id="a", url="https://x/a")
    b = _doc(id="b", url="https://x/b")
    m = _manifest(**{"aws-saa": _tenant(a, b)})
    fetch = _fake_fetch({
        "https://x/a": (200, {"content-type": "text/html"}, _HTML_OK),
        "https://x/b": (200, {"content-type": "text/html"}, _HTML_OK),
    })
    calls = []
    fetch_all(m, tmp_path, fetch, delay=0.1, sleep_fn=lambda s: calls.append(s))
    assert calls == [0.1, 0.1]


def _record(tenant, id):
    return FetchRecord(
        tenant=tenant, id=id, url="u", type="faq", format="html", corpus=True,
        citation="c", source_version=None, fetched_date="t", http_status=200,
        content_type="text/html", bytes=1, sha256="0" * 64, ok=True, note="ok",
    )


def test_write_index_merges_preserving_other_tenants(tmp_path):
    write_index(tmp_path, [_record("gcp-ace", "b")])
    write_index(tmp_path, [_record("aws-saa", "a")])  # scoped re-run
    data = json.loads((tmp_path / "fetch_index.json").read_text())
    assert {e["tenant"] for e in data} == {"aws-saa", "gcp-ace"}


def test_write_index_replaces_same_tenant(tmp_path):
    write_index(tmp_path, [_record("aws-saa", "a")])
    write_index(tmp_path, [_record("aws-saa", "a")])
    data = json.loads((tmp_path / "fetch_index.json").read_text())
    assert len(data) == 1


def test_write_index_tolerates_corrupt_existing_index(tmp_path):
    index_path = tmp_path / "fetch_index.json"
    index_path.write_text("{ not valid json", encoding="utf-8")
    rec = _record("aws-saa", "a")
    result_path = write_index(tmp_path, [rec])
    data = json.loads(result_path.read_text())
    assert len(data) == 1 and data[0]["id"] == "a"
    assert (tmp_path / "fetch_index.json.corrupt").exists()


def test_build_httpx_fetch_fn_lowercases_headers_sets_ua_and_follows_redirects():
    import httpx

    seen_uas = []

    def handler(request):
        seen_uas.append(request.headers.get("user-agent"))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "https://example.test/final"})
        return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, content=b"hello")

    fetch = build_httpx_fetch_fn(transport=httpx.MockTransport(handler))
    status, headers, body = fetch("https://example.test/start")

    assert status == 200                       # redirect followed
    assert body == b"hello"
    assert headers["content-type"] == "text/html; charset=utf-8"  # lowercased key
    assert all(ua == USER_AGENT for ua in seen_uas) and seen_uas  # UA on every hop


_MIN_MANIFEST = """
version: 1
defaults:
  corpus: true
tenants:
  aws-saa:
    display_name: "AWS SAA"
    license: "L"
    documents:
      - id: faq-ec2
        title: "EC2 FAQs"
        url: "https://x/ec2"
        type: faq
        format: html
        citation: "Amazon EC2 FAQs"
"""


def test_main_runs_with_injected_fetch(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(_MIN_MANIFEST, encoding="utf-8")
    fetch = _fake_fetch({"https://x/ec2": (200, {"content-type": "text/html"}, _HTML_OK)})
    rc = main(
        ["--manifest", str(manifest), "--data-dir", str(tmp_path / "data"), "--delay", "0"],
        fetch_fn=fetch,
    )
    assert rc == 0
    assert (tmp_path / "data" / "fetch_index.json").exists()
    assert (tmp_path / "data" / "aws-saa" / "faq-ec2.html").exists()


def test_main_returns_nonzero_when_fetch_fails(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(_MIN_MANIFEST, encoding="utf-8")
    fetch = _fake_fetch({"https://x/ec2": (404, {"content-type": "text/html"}, b"Not Found")})
    rc = main(
        ["--manifest", str(manifest), "--data-dir", str(tmp_path / "data"), "--delay", "0"],
        fetch_fn=fetch,
    )
    assert rc != 0
    assert (tmp_path / "data" / "fetch_index.json").exists()
    assert not (tmp_path / "data" / "aws-saa" / "faq-ec2.html").exists()
