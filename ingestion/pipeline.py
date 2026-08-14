from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

from backend.ai_client import embed
from backend.db import apply_migrations, get_conn
from ingestion.chunk import chunk_text
from ingestion.clean import clean_doc
from ingestion.sources import Manifest, load_manifest
from ingestion.store import ChunkRow, store_chunks


def run_pipeline(
    manifest_path: str | Path = "ingestion/sources.yaml",
    data_dir: str | Path = "data",
    *,
    tenants: list[str] | None = None,
    batch_size: int = 32,
    _manifest_override: Manifest | None = None,
) -> dict[str, int]:
    manifest = _manifest_override or load_manifest(manifest_path)
    data_dir = Path(data_dir)
    selected = tenants or list(manifest.tenants.keys())

    conn = get_conn()
    try:
        apply_migrations(conn)

        results: dict[str, int] = {}
        for tenant_name in selected:
            tenant = manifest.tenants[tenant_name]
            for doc in tenant.documents:
                if not doc.corpus:
                    continue
                raw_path = data_dir / tenant_name / f"{doc.id}.{doc.format}"
                try:
                    text = clean_doc(raw_path, doc.format)
                    chunks = chunk_text(text)
                    if not chunks:
                        print(f"[{tenant_name}] {doc.id} -> 0 chunks (empty after clean)")
                        results[doc.id] = 0
                        continue

                    # Embed in batches
                    texts = [c.content for c in chunks]
                    embeddings: list[list[float]] = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        embeddings.extend(embed(batch))

                    chunk_rows = [
                        ChunkRow(
                            content=c.content,
                            chunk_index=c.chunk_index,
                            token_count=c.token_count,
                            embedding=embeddings[j],
                        )
                        for j, c in enumerate(chunks)
                    ]
                    source_date = datetime.now(timezone.utc).date().isoformat()
                    n = store_chunks(
                        conn,
                        tenant=tenant_name,
                        doc=doc,
                        source_date=source_date,
                        chunks=chunk_rows,
                    )
                    conn.commit()
                    print(f"[{tenant_name}] {doc.id} -> {n} chunks")
                    results[doc.id] = n
                except Exception as e:
                    print(f"SKIP {doc.id}: {e}")
                    conn.rollback()

        if any(v > 0 for v in results.values()):
            print("Rebuilding IVFFlat index...")
            with conn.cursor() as cur:
                cur.execute("REINDEX INDEX chunks_embedding_ivfflat_idx;")
            conn.commit()
            print("IVFFlat index rebuilt.")
    finally:
        conn.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CertCoach ingestion pipeline.")
    parser.add_argument("--manifest", default="ingestion/sources.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tenant", action="append", dest="tenants")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    results = run_pipeline(
        args.manifest, args.data_dir,
        tenants=args.tenants, batch_size=args.batch_size,
    )
    total = sum(results.values())
    print(f"\nTotal: {total} chunks across {len(results)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
