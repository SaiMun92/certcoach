from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from opentelemetry import trace as otel_trace
from backend.ai_client import embed  # module-level for patchability
from backend.generate import generate  # module-level for patchability
from backend.generate import answer_question  # module-level for patchability
from backend.db import get_conn  # module-level for patchability
from backend.retrieval import retrieve  # module-level for patchability
from backend.tracing import configure_tracing


# ── Pure metric functions ────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if retrieved[:k] and retrieved[0] in relevant else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, r in enumerate(retrieved[:k]) if r in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


# ── LLM judge ───────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an objective evaluator. Score the candidate answer against the reference "
    "answer for factual accuracy on a 0.0–1.0 scale. "
    'Return JSON: {"score": float, "rationale": str}. Return ONLY JSON.'
)


def judge_answer(question: str, candidate: str, reference: str, *, model: str = "gpt-5") -> float:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": f"Reference: {reference}\nCandidate: {candidate}"},
    ]
    raw = generate(messages, model=model)
    # Strip markdown code fences if present
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return float(json.loads(raw)["score"])


# ── Main eval runner ─────────────────────────────────────────────────────────

def avg(xs: list[float]) -> float:
    """Compute average of a list, or NaN if empty."""
    return sum(xs) / len(xs) if xs else float("nan")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval(
    gold_retrieval_path: str = "eval/gold_retrieval.jsonl",
    gold_qa_path: str = "eval/gold_qa.jsonl",
    *,
    prompt_version: int = 1,
    output_path: str | None = None,
    judge_model: str = "gpt-5",
    _retrieval_mode: str = "hybrid",
) -> dict:
    configure_tracing()
    tracer = otel_trace.get_tracer("certcoach")
    with tracer.start_as_current_span("certcoach.eval_run") as span:
        span.set_attribute("certcoach.prompt_version", prompt_version)
        span.set_attribute("certcoach.judge_model", judge_model)

        if output_path is None:
            output_path = "eval/results/eval_results.json" if prompt_version == 1 else f"eval/results/eval_results_v{prompt_version}.json"

        gold_ret = _load_jsonl(Path(gold_retrieval_path))
        gold_qa = _load_jsonl(Path(gold_qa_path))

        # Batch-embed all questions in two round-trips (one per gold set).
        ret_questions = [item["question"] for item in gold_ret]
        qa_questions = [item["question"] for item in gold_qa]
        ret_embeddings = embed(ret_questions) if ret_questions else []
        qa_embeddings = embed(qa_questions) if qa_questions else []

        results_by_tenant: dict[str, dict] = {}

        def _new_bucket() -> dict:
            return {"recall5": [], "hit1": [], "mrr10": [], "ndcg5": [],
                    "accuracy": [], "latencies_ms": [], "ret_details": [], "qa_details": []}

        # ── Retrieval eval (parallelised) ─────────────────────────────────────
        def _run_retrieval(item: dict, qemb: list[float]) -> dict:
            conn = get_conn()
            try:
                t0 = time.perf_counter()
                result = retrieve(conn, item["question"], item["tenant"],
                                  query_embedding=qemb, retrieval_mode=_retrieval_mode)
                elapsed_ms = (time.perf_counter() - t0) * 1000
            finally:
                conn.close()
            retrieved_sources = list(dict.fromkeys(ec.chunk.source_id for ec in result.chunks))
            relevant = set(item["relevant_source_ids"])
            return {
                "tenant": item["tenant"],
                "recall5": recall_at_k(retrieved_sources, relevant, k=5),
                "hit1": hit_rate_at_k(retrieved_sources, relevant, k=1),
                "mrr10": mrr_at_k(retrieved_sources, relevant, k=10),
                "ndcg5": ndcg_at_k(retrieved_sources, relevant, k=5),
                "elapsed_ms": elapsed_ms,
                "detail": {"question": item["question"], "relevant": list(relevant),
                           "retrieved": retrieved_sources, "elapsed_ms": round(elapsed_ms, 1)},
            }

        # Note: worker threads don't inherit the eval_run OTel context, so
        # retrieve/LLM child spans appear as root-level traces in Phoenix.
        # Fix: wrap submit() with ctx.run() if nested traces are needed.
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_run_retrieval, item, qemb)
                       for item, qemb in zip(gold_ret, ret_embeddings)]
            for fut in as_completed(futures):
                r = fut.result()
                b = results_by_tenant.setdefault(r["tenant"], _new_bucket())
                b["recall5"].append(r["recall5"])
                b["hit1"].append(r["hit1"])
                b["mrr10"].append(r["mrr10"])
                b["ndcg5"].append(r["ndcg5"])
                b["latencies_ms"].append(r["elapsed_ms"])
                b["ret_details"].append(r["detail"])

        # ── Answer eval (parallelised) ────────────────────────────────────────
        def _run_qa(item: dict, qemb: list[float]) -> dict:
            conn = get_conn()
            try:
                result = retrieve(conn, item["question"], item["tenant"],
                                  query_embedding=qemb, retrieval_mode=_retrieval_mode)
            finally:
                conn.close()
            response = answer_question(result, prompt_version=prompt_version)
            score = judge_answer(item["question"], response.answer,
                                 item["reference_answer"], model=judge_model)
            return {
                "tenant": item["tenant"],
                "score": score,
                "detail": {"question": item["question"],
                           "reference": item["reference_answer"],
                           "candidate": response.answer, "score": score},
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_run_qa, item, qemb)
                       for item, qemb in zip(gold_qa, qa_embeddings)]
            for fut in as_completed(futures):
                r = fut.result()
                b = results_by_tenant.setdefault(r["tenant"], _new_bucket())
                b["accuracy"].append(r["score"])
                b["qa_details"].append(r["detail"])

        # ── Print results ──────────────────────────────────────────────────────
        print("\n=== CertCoach Eval Results ===")
        output: dict = {}
        for tenant, b in sorted(results_by_tenant.items()):
            n_ret = len(b["recall5"])
            n_qa = len(b["accuracy"])
            lats = sorted(b["latencies_ms"])
            p50 = lats[len(lats) // 2] if lats else 0.0
            p95 = lats[int(len(lats) * 0.95)] if lats else 0.0

            print(f"Tenant: {tenant}   n={n_ret}")
            print(f"  Retrieval:  recall@5={avg(b['recall5']):.3f}  hit@1={avg(b['hit1']):.3f}  "
                  f"mrr@10={avg(b['mrr10']):.3f}  ndcg@5={avg(b['ndcg5']):.3f}")
            print(f"  Answer:     accuracy={avg(b['accuracy']):.3f}  (n={n_qa}, judge={judge_model})")
            print(f"  Latency:    retrieval_p50={p50:.0f}ms  retrieval_p95={p95:.0f}ms")
            print()
            output[tenant] = {
                "recall_at_5": avg(b["recall5"]),
                "hit_rate_at_1": avg(b["hit1"]),
                "mrr_at_10": avg(b["mrr10"]),
                "ndcg_at_5": avg(b["ndcg5"]),
                "answer_accuracy": avg(b["accuracy"]),
                "n_retrieval": n_ret,
                "n_qa": n_qa,
                "latency_p50_ms": round(p50, 1),
                "latency_p95_ms": round(p95, 1),
                "retrieval_details": b["ret_details"],
                "qa_details": b["qa_details"],
            }

        Path(output_path).write_text(json.dumps(output, indent=2))
        return output


def run_ab_eval(
    configs: list[dict] | None = None,
    gold_retrieval_path: str = "eval/gold_retrieval.jsonl",
    gold_qa_path: str = "eval/gold_qa.jsonl",
    output_path: str = "eval/results/eval_results_ab.json",
) -> dict:
    """Run eval under multiple retrieval configs and print a comparison table."""
    if configs is None:
        configs = [
            {"label": "dense_only",       "retrieval_mode": "dense_only",       "prompt_version": 1},
            {"label": "hybrid_no_rerank", "retrieval_mode": "hybrid_no_rerank", "prompt_version": 1},
            {"label": "hybrid+rerank",    "retrieval_mode": "hybrid",            "prompt_version": 1},
            {"label": "hybrid+rerank+v2", "retrieval_mode": "hybrid",            "prompt_version": 2},
        ]
    all_results: dict = {}
    for cfg in configs:
        label = cfg["label"]
        mode = cfg.get("retrieval_mode", "hybrid")
        pv = cfg.get("prompt_version", 1)
        print(f"\n--- Running config: {label} ---")
        result = run_eval(
            gold_retrieval_path,
            gold_qa_path,
            prompt_version=pv,
            output_path=f"eval/results/eval_results_{label}.json",
            _retrieval_mode=mode,
        )
        all_results[label] = result

    print("\n=== A/B Comparison ===")
    header = f"{'Config':<22} {'recall@5':>8} {'hit@1':>6} {'mrr@10':>7} {'ndcg@5':>7} {'accuracy':>9}"
    print(header)
    print("-" * len(header))
    for label, res in all_results.items():
        tenants = list(res.keys())
        r5  = avg([res[t]["recall_at_5"]     for t in tenants])
        h1  = avg([res[t]["hit_rate_at_1"]   for t in tenants])
        m10 = avg([res[t]["mrr_at_10"]       for t in tenants])
        n5  = avg([res[t]["ndcg_at_5"]       for t in tenants])
        acc = avg([res[t]["answer_accuracy"]  for t in tenants])
        print(f"{label:<22} {r5:>8.3f} {h1:>6.3f} {m10:>7.3f} {n5:>7.3f} {acc:>9.3f}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", type=int, default=1)
    parser.add_argument("--ab", action="store_true", help="Run A/B comparison across retrieval configs")
    args = parser.parse_args()
    try:
        if args.ab:
            run_ab_eval()
        else:
            run_eval(prompt_version=args.prompt_version)
    except Exception as exc:  # noqa: BLE001
        print(f"eval failed: {exc}", file=sys.stderr)
    sys.exit(0)
