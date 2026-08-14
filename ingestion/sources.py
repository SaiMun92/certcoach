from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

DocType = Literal["exam_guide", "faq", "whitepaper", "doc", "sample_questions"]
DocFormat = Literal["pdf", "html"]


class DocSpec(BaseModel):
    id: str
    title: str
    url: str
    type: DocType
    format: DocFormat
    citation: str
    corpus: bool = True
    source_version: str | None = None


class TenantSpec(BaseModel):
    display_name: str
    license: str
    documents: list[DocSpec]


class Manifest(BaseModel):
    version: int
    tenants: dict[str, TenantSpec]


def load_manifest(path: str | Path) -> Manifest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    default_corpus = defaults.get("corpus", True)
    for tenant in (raw.get("tenants") or {}).values():
        for doc in tenant.get("documents", []):
            doc.setdefault("corpus", default_corpus)
    return Manifest(version=raw["version"], tenants=raw.get("tenants") or {})
