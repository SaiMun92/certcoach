from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from ingestion.sources import DocSpec, Manifest, load_manifest

FetchFn = Callable[[str], "tuple[int, dict[str, str], bytes]"]

USER_AGENT = "CertCoach-ingest/0.1 (portfolio project)"
THIN_HTML_MIN_WORDS = 200

_SCRIPT_RE = re.compile(r"(?is)<script.*?</script>")
_STYLE_RE = re.compile(r"(?is)<style.*?</style>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")


class FetchRecord(BaseModel):
    tenant: str
    id: str
    url: str
    type: str
    format: str
    corpus: bool
    citation: str
    source_version: str | None
    fetched_date: str
    http_status: int
    content_type: str | None
    bytes: int
    sha256: str
    ok: bool
    note: str


def target_path(data_dir: Path, tenant: str, doc: DocSpec) -> Path:
    return data_dir / tenant / f"{doc.id}.{doc.format}"


def _visible_word_count(body: bytes) -> int:
    text = body.decode("utf-8", errors="ignore")
    text = _SCRIPT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return len(text.split())


def classify_response(doc: DocSpec, status: int, body: bytes) -> tuple[bool, str]:
    if status != 200:
        return False, f"http {status}"
    if doc.format == "pdf" and not body.startswith(b"%PDF"):
        return False, "not a pdf (missing %PDF header)"
    if doc.format == "html" and _visible_word_count(body) < THIN_HTML_MIN_WORDS:
        return False, "thin content - possible JS shell"
    return True, "ok"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_all(
    manifest: Manifest,
    data_dir: Path,
    fetch_fn: FetchFn,
    *,
    tenants: list[str] | None = None,
    delay: float = 0.0,
    now_fn: Callable[[], str] = now_iso,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[FetchRecord]:
    records: list[FetchRecord] = []
    selected = tenants or list(manifest.tenants.keys())
    unknown = [t for t in selected if t not in manifest.tenants]
    if unknown:
        raise ValueError(
            f"unknown tenant(s): {', '.join(unknown)}; "
            f"known: {', '.join(manifest.tenants)}"
        )
    for tenant_name in selected:
        tenant = manifest.tenants[tenant_name]
        for doc in tenant.documents:
            status, headers, body = fetch_fn(doc.url)
            ok, note = classify_response(doc, status, body)
            if ok:
                path = target_path(data_dir, tenant_name, doc)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            records.append(
                FetchRecord(
                    tenant=tenant_name,
                    id=doc.id,
                    url=doc.url,
                    type=doc.type,
                    format=doc.format,
                    corpus=doc.corpus,
                    citation=doc.citation,
                    source_version=doc.source_version,
                    fetched_date=now_fn(),
                    http_status=status,
                    content_type=headers.get("content-type"),
                    bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                    ok=ok,
                    note=note,
                )
            )
            if delay:
                sleep_fn(delay)
    return records


def write_index(data_dir: Path, records: list[FetchRecord]) -> Path:
    index_path = data_dir / "fetch_index.json"
    new_tenants = {r.tenant for r in records}
    existing: list[dict] = []
    if index_path.exists():
        try:
            existing = [
                e for e in json.loads(index_path.read_text(encoding="utf-8"))
                if e["tenant"] not in new_tenants
            ]
        except (json.JSONDecodeError, OSError):
            # Don't crash or silently discard: preserve the unparseable index
            # for inspection and rebuild from this run's records.
            index_path.replace(index_path.parent / (index_path.name + ".corrupt"))
    combined = existing + [r.model_dump() for r in records]
    combined.sort(key=lambda e: (e["tenant"], e["id"]))
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_path.parent / (index_path.name + ".tmp")
    tmp.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    tmp.replace(index_path)
    return index_path


def build_httpx_fetch_fn(*, timeout: float = 30.0, transport: "httpx.BaseTransport | None" = None) -> FetchFn:
    import httpx

    client = httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    )
    atexit.register(client.close)

    def fetch(url: str) -> tuple[int, dict[str, str], bytes]:
        resp = client.get(url)
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.status_code, headers, resp.content

    return fetch


def main(argv: list[str] | None = None, fetch_fn: FetchFn | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch CertCoach raw sources to disk.")
    parser.add_argument("--manifest", default="ingestion/sources.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tenant", action="append", dest="tenants",
                        help="restrict to a tenant (repeatable)")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    data_dir = Path(args.data_dir)
    records = fetch_all(
        manifest, data_dir, fetch_fn or build_httpx_fetch_fn(),
        tenants=args.tenants, delay=args.delay,
    )
    index_path = write_index(data_dir, records)

    ok = sum(1 for r in records if r.ok)
    print(f"fetched {ok}/{len(records)} ok -> {index_path}")
    for r in records:
        if not r.ok:
            print(f"  FAIL {r.tenant}/{r.id}: {r.note} ({r.url})")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
