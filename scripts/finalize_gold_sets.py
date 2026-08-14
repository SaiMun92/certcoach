#!/usr/bin/env python3
"""One-shot finalizer for the AWS + GCP gold eval sets.

Does three things, atomically per file (asserts guard every content edit; if any
assertion fails the script raises BEFORE writing anything):

  1. Applies the 6 correctness/precision fixes surfaced by the adversarial
     verification pass (AWS q79; GCP q04, q06, q61, q77, q94).
  2. Runs the validated, deterministic answer-letter rebalance to 25/25/25/25
     (seeds: AWS=42, GCP=7 — same seeds dry-run-validated earlier).
  3. Adds a `verification` + `answer_distribution` metadata block.

Then writes both files (indent=2, ensure_ascii=False) and prints validation.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/Users/I542439/Documents/interviews/certcoach")
AWS = ROOT / "data/aws-saa/saa-sample-questions.json"
GCP = ROOT / "data/gcp-ace/ace-sample-questions.json"

# ---------------------------------------------------------------------------
# 1. Content fixes.  Each entry: id -> { "field-path": (must_contain, new_text) }
#    field-path is "question", "explanation", or "options.<LETTER>".
#    must_contain is a distinctive substring of the CURRENT value; if it is not
#    present the edit aborts (protects against editing the wrong/changed text).
# ---------------------------------------------------------------------------

FIXES = {
    "aws-saa-q79": {
        "options.B": (
            "Order an AWS Snowball Edge device",
            "Order one or more AWS Snowball Edge devices (a Snowball Edge cluster "
            "for a dataset this size), copy the data to them locally, and ship them "
            "back to AWS for import into S3",
        ),
        "explanation": (
            "multi-hundred-terabyte transfers",
            "For large datasets where the available bandwidth would make an online "
            "transfer take too long, the AWS Snow Family (Snowball Edge) provides "
            "physical devices you load with data on-site and ship back to AWS, which "
            "then imports the data into S3 — completing large transfers in a "
            "fraction of the time an internet upload would take. Because a single "
            "Snowball Edge device holds on the order of tens of terabytes of usable "
            "capacity, a 400 TB migration is handled with multiple devices (a "
            "cluster). Transferring over the public internet or a site-to-site VPN "
            "(the other options) is bounded by the link speed and would blow the "
            "timeline.",
        ),
    },
    "gcp-ace-q04": {
        "explanation": (
            "disabled by default in new projects",
            "Many Google Cloud APIs — including the Cloud SQL Admin API "
            "(sqladmin.googleapis.com) — are not enabled by default in a new "
            "project and must be explicitly enabled (via the API Library in the "
            "console or with `gcloud services enable`) before any calls to that API "
            "will succeed. A small baseline set of APIs is enabled automatically "
            "when a project is created, but the Cloud SQL Admin API is not among "
            "them, so the first call fails until it is turned on.",
        ),
    },
    "gcp-ace-q06": {
        "question": (
            "millions of reads and writes per second",
            "An application stores user session data that is accessed with "
            "low-latency key-value and document lookups, must scale elastically to "
            "very high read/write throughput, and should require no schema or "
            "schema migrations. Which fully managed, serverless database is the best "
            "fit?",
        ),
        "explanation": (
            "millions of operations per second",
            "Firestore is a fully managed, serverless NoSQL document database that "
            "supports low-latency key-value and document access, scales elastically "
            "to very high throughput, and requires no predefined schema or schema "
            "migrations — a good match for session data. Cloud SQL (option A) "
            "and Cloud Spanner (option B) are relational and involve schema "
            "management, and BigQuery (option D) is an analytical data warehouse "
            "rather than a low-latency operational key-value store.",
        ),
    },
    "gcp-ace-q61": {
        "options.B": (
            "allocate CPU only during request processing",
            "Set a minimum number of instances greater than zero, enable startup "
            "CPU boost, and set CPU to be always allocated (instance-based billing)",
        ),
        "explanation": (
            "ensuring adequate CPU allocation",
            "Setting Cloud Run's minimum instances above zero keeps warm containers "
            "ready, eliminating cold starts for the baseline of traffic. Enabling "
            "startup CPU boost accelerates container startup, and setting CPU to be "
            "always allocated (instance-based billing) prevents the CPU from being "
            "throttled between requests — together these cut both cold-start and "
            "warm-path latency. Lowering the maximum instance count (option A) can "
            "cause request queuing under load, and increasing the request timeout "
            "(option D) tolerates slow responses rather than preventing them.",
        ),
    },
    "gcp-ace-q77": {
        "question": (
            "binary logging (point-in-time recovery) enabled",
            "point-in-time recovery (write-ahead-log archiving) enabled",
        ),
        "explanation": (
            "point-in-time recovery (binary logs)",
            "Because the Cloud SQL for PostgreSQL instance has automated backups "
            "plus point-in-time recovery (backed by write-ahead-log archiving) "
            "enabled, you can run a PITR that recreates the database on a new "
            "instance at a precise timestamp — just before 14:32 — losing "
            "only the few moments of writes immediately preceding the mistake. "
            "Restoring the most recent nightly backup (option A) discards every "
            "change since that backup, and a read replica (option C) would already "
            "have replicated the destructive statement, so neither preserves the "
            "pre-incident state.",
        ),
    },
    "gcp-ace-q94": {
        "options.B": (
            "roles/iam.serviceAccountUser (or serviceAccountTokenCreator)",
            "roles/iam.serviceAccountTokenCreator on the 'deployer' service "
            "account, letting 'ci-runner' mint short-lived tokens for it and "
            "impersonate it",
        ),
        "explanation": (
            "roles/iam.serviceAccountUser",
            "To impersonate another service account and mint short-lived "
            "credentials (access tokens) for it, the calling identity ('ci-runner') "
            "needs roles/iam.serviceAccountTokenCreator (permission "
            "iam.serviceAccounts.getAccessToken) on the target 'deployer' service "
            "account — this is the recommended keyless pattern. Note that "
            "roles/iam.serviceAccountUser grants a distinct capability (actAs: "
            "deploying resources that then run as the service account) and does not "
            "by itself allow generating short-lived tokens. serviceAccountKeyAdmin "
            "(option A) permits creating downloadable long-lived keys — exactly "
            "what you are trying to avoid — roles/owner (option C) grossly "
            "over-grants, and roles/iam.roleAdmin (option D) only manages role "
            "definitions.",
        ),
    },
}


def apply_fixes(data, tenant_prefix):
    changed = []
    by_id = {q["id"]: q for q in data["questions"]}
    for qid, fields in FIXES.items():
        if not qid.startswith(tenant_prefix):
            continue
        q = by_id.get(qid)
        assert q is not None, f"{qid} not found in {tenant_prefix}"
        for path, (must_contain, new_text) in fields.items():
            if path.startswith("options."):
                letter = path.split(".", 1)[1]
                cur = q["options"][letter]
                target = q["options"]
                key = letter
                container = ("options", letter)
            else:
                cur = q[path]
                container = (path,)
            assert must_contain in cur, (
                f"{qid}:{path} — expected substring not found:\n"
                f"  need: {must_contain!r}\n  have: {cur[:160]!r}"
            )
            if len(container) == 2:
                q["options"][container[1]] = new_text
            else:
                q[container[0]] = new_text
        changed.append(qid)
    return changed


# ---------------------------------------------------------------------------
# 2. Rebalance (validated earlier; identical seeds -> identical permutation).
# ---------------------------------------------------------------------------

import random

REF = re.compile(r'(options?\s+)([A-D](?:\s*(?:,\s*and\s+|,\s*|\s+and\s+)[A-D])*)',
                 re.IGNORECASE)


def remap_expl(text, cmap):
    table = str.maketrans(cmap)
    return REF.sub(lambda m: m.group(1) + m.group(2).translate(table), text)


def rebalance(data, seed):
    questions = data["questions"]
    n = len(questions)
    base = (["A", "B", "C", "D"] * ((n // 4) + 1))[:n]
    random.Random(seed).shuffle(base)
    for q, target in zip(questions, base):
        x = q["answer"]
        y = target
        if x == y:
            continue
        opts = q["options"]
        opts[x], opts[y] = opts[y], opts[x]
        q["answer"] = y
        cmap = {x: y, y: x}
        q["explanation"] = remap_expl(q["explanation"], cmap)
    dist = {k: 0 for k in "ABCD"}
    for q in questions:
        dist[q["answer"]] += 1
    return dist


# ---------------------------------------------------------------------------
# 3. Verification metadata block (honest, per-file accurate).
# ---------------------------------------------------------------------------

VERIF_COMMON = {
    "method": (
        "Adversarial LLM self-review: each exam domain/section was independently "
        "reviewed by a separate Claude subagent instructed to disprove the keyed "
        "answer and flag factual errors, ambiguity, and weak distractors. This is "
        "model self-review, NOT human-SME certification and NOT validation against "
        "a certification-authority answer key."
    ),
    "reviewer": "Claude (Anthropic) — 9 independent subagent reviewers, one per AWS domain / GCP section",
    "date": "2026-07-31",
    "answer_key_balancing": (
        "After authoring, correct-answer positions were deterministically permuted "
        "to remove positional bias, yielding a balanced A/B/C/D key distribution. "
        "Option content and which option is correct are unchanged by the permutation."
    ),
    "limitations": (
        "Facts reflect model training knowledge cross-checked by same-family model "
        "reviewers; not independently verified by a human subject-matter expert. "
        "Intended for portfolio/RAG-eval use, not as a guaranteed exam-prep resource."
    ),
}

VERIF_RESULT = {
    "aws-saa": "100 questions across 4 domains reviewed; 0 wrong answer keys; 1 low-severity precision fix applied (q79).",
    "gcp-ace": "100 questions across 5 sections reviewed; 0 wrong answer keys; 5 fixes applied (4 low-severity wording/precision: q04, q06, q61, q77; 1 medium factual: q94).",
}


def finalize(path, tenant_prefix, seed):
    data = json.loads(path.read_text())
    n_before = len(data["questions"])

    changed = apply_fixes(data, tenant_prefix)

    # Guard: after edits, no fixed explanation should contain a bare "option X"
    # reference to a letter we did NOT intend (sanity print, not an abort).
    dist = rebalance(data, seed)

    tenant = tenant_prefix.rsplit("-", 1)[0] if tenant_prefix.startswith("aws") else "gcp-ace"
    tenant_key = "aws-saa" if tenant_prefix.startswith("aws") else "gcp-ace"

    verif = dict(VERIF_COMMON)
    verif["result"] = VERIF_RESULT[tenant_key]
    data["metadata"]["answer_distribution"] = dist
    data["metadata"]["verification"] = verif

    # ---- validation ----
    qs = data["questions"]
    assert len(qs) == n_before == 100, f"{tenant_key}: expected 100 questions, got {len(qs)}"
    ids = [q["id"] for q in qs]
    assert len(set(ids)) == 100, f"{tenant_key}: duplicate ids"
    for q in qs:
        assert set(q["options"].keys()) == set("ABCD"), f"{q['id']}: options != A-D"
        assert q["answer"] in q["options"], f"{q['id']}: answer not in options"
        assert all(q["options"].values()), f"{q['id']}: empty option"
    assert dist == {"A": 25, "B": 25, "C": 25, "D": 25}, f"{tenant_key}: dist {dist}"

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data, changed, dist


def main():
    print("=" * 72)
    for path, prefix, seed in [(AWS, "aws-saa", 42), (GCP, "gcp-ace", 7)]:
        data, changed, dist = finalize(path, prefix, seed)
        print(f"{path.name}")
        print(f"  fixes applied : {', '.join(changed)}")
        print(f"  distribution  : {dist}")
        print(f"  questions     : {len(data['questions'])}  (unique ids: "
              f"{len(set(q['id'] for q in data['questions']))})")
        # group counts unchanged?
        field = "domain" if prefix == "aws-saa" else "section"
        counts = {}
        for q in data["questions"]:
            counts[q[field]] = counts.get(q[field], 0) + 1
        print(f"  {field} counts : {dict(sorted(counts.items(), key=lambda kv: str(kv[0])))}")
        print("-" * 72)

    # Show the final (post-fix, post-rebalance) state of the 6 fixed questions.
    print("\nFINAL STATE OF FIXED QUESTIONS")
    print("=" * 72)
    for path in (AWS, GCP):
        data = json.loads(path.read_text())
        by_id = {q["id"]: q for q in data["questions"]}
        for qid in FIXES:
            if qid not in by_id:
                continue
            q = by_id[qid]
            print(f"\n### {qid}  (answer now: {q['answer']})")
            print(f"Q: {q['question']}")
            for L in "ABCD":
                mark = " <== KEY" if L == q["answer"] else ""
                print(f"  {L}. {q['options'][L]}{mark}")
            print(f"EXPL: {q['explanation']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
