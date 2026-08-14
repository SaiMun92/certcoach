from __future__ import annotations
import functools
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _resolve_base_url(url: str) -> str:
    """Swap host between 127.0.0.1 and host.docker.internal based on environment."""
    in_docker = os.path.exists("/.dockerenv")
    if in_docker:
        return url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    return url.replace("host.docker.internal", "127.0.0.1")


@functools.lru_cache(maxsize=1)
def _get_client():
    import openai
    return openai.OpenAI(
        base_url=_resolve_base_url(os.environ["AICORE_BASE_URL"]),
        api_key=os.environ["AICORE_API_KEY"],
    )


def embed(
    texts: list[str],
    *,
    model: str = "text-embedding-3-large",
    dimensions: int = 1536,
) -> list[list[float]]:
    if not texts:
        raise ValueError("texts must not be empty")
    client = _get_client()
    response = client.embeddings.create(input=texts, model=model, dimensions=dimensions)
    return [item.embedding for item in response.data]


@dataclass
class RerankResult:
    index: int
    score: float


# ---------------------------------------------------------------------------
# Rerank — calls the internal model gateway directly (OAuth2 client-credentials) because
# the local proxy does not implement the /rerank subpath for cohere-rerank-pro.
# ---------------------------------------------------------------------------

_RERANK_DEPLOYMENT_ID = "d7ebe1fa9a08238f"

_token_cache: dict = {}  # keys: access_token, expires_at


def _get_aicore_token() -> str:
    import httpx
    now = time.monotonic()
    if _token_cache.get("access_token") and now < _token_cache.get("expires_at", 0):
        return _token_cache["access_token"]

    auth_url = os.environ["AICORE_AUTH_URL"].rstrip("/")
    resp = httpx.post(
        f"{auth_url}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["AICORE_CLIENT_ID"],
            "client_secret": os.environ["AICORE_CLIENT_SECRET"],
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600) - 60
    return _token_cache["access_token"]


def rerank(
    query: str,
    documents: list[str],
    *,
    model: str = "cohere-rerank-pro",  # noqa: ARG001 — kept for call-site compatibility
    top_n: int = 5,
) -> list[RerankResult]:
    if not documents:
        return []
    import httpx
    token = _get_aicore_token()
    base = os.environ["AICORE_DIRECT_BASE_URL"].rstrip("/")
    resource_group = os.environ.get("AICORE_RESOURCE_GROUP", "default")
    url = f"{base}/v2/inference/deployments/{_RERANK_DEPLOYMENT_ID}/rerank"
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
        },
        json={"query": query, "documents": documents, "top_n": top_n},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return [RerankResult(index=r["index"], score=r["relevance_score"]) for r in data["results"]]
