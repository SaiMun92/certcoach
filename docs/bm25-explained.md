# BM25 / Sparse Search, Explained Simply

*How keyword search "nails" exact matches — no math required.*

## The one-sentence idea

Sparse search (BM25) matches the **actual words**. If a document contains the
exact word you searched for, it's a candidate; if it doesn't, it isn't. Simple as that.

## An analogy: the index at the back of a textbook

Imagine the index at the back of a cert study guide:

```
IAM ............... p. 12, 88, 140
KMS ............... p. 45
S3 bucket ......... p. 7, 7, 7, 30
```

To find "KMS", you flip to the **K** entry and go straight to page 45. You don't
read the whole book. That's exactly how BM25 works under the hood — it keeps a
list of "which word appears on which page" (called an **inverted index**) and
jumps straight to the pages that literally contain your word.

This is why it's great at exact matches: it only ever looks at documents that
**actually contain the word**. There is no guessing.

## The 3 simple rules that make it smart

BM25 doesn't just find the pages — it *ranks* them, using three common-sense rules.

### Rule 1: Rare words matter most

If you search for "the", almost every document has it — useless for ranking.
If you search for `iam:PassRole`, only a handful of documents have it — so those
few are almost certainly what you want.

**BM25 gives rare words a big score and common words almost none.** This is the
key reason it shines on cert content, which is full of rare, exact terms like
`s3:GetObject`, `provisioned IOPS`, or a specific error code. Match one of those
and BM25 is very confident it found the right page.

### Rule 2: More mentions = more relevant, but with diminishing returns

A page that mentions "KMS" five times is more about KMS than a page that
mentions it once. But a page that mentions it 50 times isn't 50× better — after
a few mentions, extra ones barely help. This stops "keyword stuffing" from
gaming the ranking.

### Rule 3: Shorter, focused pages win

If two pages both mention "bucket versioning", but one is a tight paragraph on
exactly that topic and the other is a giant page that mentions it in passing,
the short focused one ranks higher. BM25 slightly penalizes long documents so a
single lucky mention in a huge page doesn't beat a page that's genuinely about
your topic.

That's the whole thing: **find pages with your exact words → rank by
rarity, count, and focus.**

## Why the "smart" AI search (embeddings) struggles here

The other kind of search — **dense / embeddings** — turns text into *meaning*.
It knows "S3" and "object storage" are related, which is fantastic for questions
like *"how do I make a bucket private?"* where the user's words don't match the
docs.

But that same "meaning blur" is a weakness for exact terms:

- It can confuse `S3`, `S3 Glacier`, and `EBS` because they're all "storage-ish".
- Rare exact strings — an ARN, a version number, a specific flag — get smoothed
  over or lost, because the model was never really trained on them.

So each method covers the other's blind spot:

| Method | Great at | Bad at |
|---|---|---|
| **Sparse / BM25** | exact words, rare identifiers, API names | synonyms, paraphrased questions |
| **Dense / embeddings** | meaning, synonyms, vague questions | exact rare terms, precise identifiers |

## How this fits CertCoach

Cert material is *packed* with exact identifiers — service names, API calls,
quotas, port numbers. So we use **both** searches and combine their results
("hybrid retrieval"):

1. **BM25** catches the exact-term questions (*"what does `iam:PassRole` do?"*).
2. **Embeddings** catch the conceptual ones (*"how do I lock down a bucket?"*).
3. We merge both result lists with RRF. In `hybrid` mode an optional **reranker** picks the
   final best order; the production default (`hybrid_no_rerank`) relies on the RRF ordering
   directly, which is already strong at this corpus scale.

BM25 is the part quietly making sure that when someone asks about an exact API
call, we actually find the page that mentions it — something the "smart" search
alone would sometimes miss.

## The formula

BM25's score for a document is just Rules 1–3 turned into arithmetic. For each
word in your search, you add up: `(rarity) × (a count that saturates) × (a length
penalty)`. Here is the full thing.

```
                    f(t,D) · (k1 + 1)
score(D,Q) =  Σ    ──────────────────────────────────  · IDF(t)
             t∈Q    f(t,D) + k1 · (1 - b + b·|D|/avgdl)
```

What each symbol means:

| Symbol   | Meaning                                                     |
|----------|-------------------------------------------------------------|
| `Q`      | the search query (a set of words `t`)                       |
| `D`      | one document (a chunk) being scored                         |
| `f(t,D)` | how many times word `t` appears in document `D`             |
| `|D|`    | length of document `D`, in words                            |
| `avgdl`  | average document length across the whole corpus             |
| `k1`     | saturation knob (Rule 2), usually ~1.2–2.0                  |
| `b`      | length-penalty knob (Rule 3), usually ~0.75                 |
| `IDF(t)` | rarity weight (Rule 1) — big for rare words, tiny for common |

And **IDF** (the rarity weight) is:

```
              N - n(t) + 0.5
IDF(t) = ln( ──────────────── + 1 )
                n(t) + 0.5
```

where `N` = total documents in the corpus, and `n(t)` = how many documents
contain word `t`. (`ln` is the natural logarithm — any calculator has it.)

That's the whole equation. Everything below just plugs in real numbers.

---

## Worked example 1 — why a rare word dominates (Rule 1)

Say our corpus has **5 documents**. Let's compare three words by how many docs they appear in:

| Word    | Appears in # docs (`n`) | IDF (rarity weight)                          |
|---------|-------------------------|----------------------------------------------|
| `with`  | 5 of 5                  | `ln((5-5+0.5)/(5+0.5) + 1)` = **0.09** (tiny) |
| `kms`   | 2 of 5                  | `ln((5-2+0.5)/(2+0.5) + 1)` = **0.88**        |
| `iam:PassRole` | 1 of 5           | `ln((5-1+0.5)/(1+0.5) + 1)` = **1.39** (big)  |

The word in only **1** document gets ~16× the weight of `with`, which is in every
document. So matching a rare, exact identifier like `iam:PassRole` swamps the
score — that's the "nails exact keywords" behavior in action.

---

## Worked example 2 — short focused doc beats long rambling doc (Rules 2 & 3)

Now let's fully score a search for **`kms`**. Setup:

- Corpus of 5 docs, and `kms` appears in 2 of them → from above, **IDF(kms) = 0.88**
- Knobs: `k1 = 1.5`, `b = 0.75`
- Average doc length across the corpus: **avgdl = 27 words**

Two candidate documents both mention `kms`:

| Doc  | Length `|D|` | Mentions of `kms`, `f(t,D)` | What it is                          |
|------|--------------|-----------------------------|-------------------------------------|
| `D1` | 8 words      | 2                           | a tight chunk *about* KMS           |
| `D5` | 100 words    | 1                           | a long page that mentions kms once  |

**Score D1** (short, 2 mentions):

```
top    = f·(k1+1)                       = 2 · 2.5                     = 5.0
bottom = f + k1·(1 - b + b·|D|/avgdl)   = 2 + 1.5·(0.25 + 0.75·8/27)  = 2.71
TF part = top / bottom                  = 5.0 / 2.71                  = 1.85
score   = TF part · IDF                 = 1.85 · 0.88                 ≈ 1.62
```

**Score D5** (long, 1 mention):

```
top    = f·(k1+1)                       = 1 · 2.5                       = 2.5
bottom = f + k1·(1 - b + b·|D|/avgdl)   = 1 + 1.5·(0.25 + 0.75·100/27)  = 5.51
TF part = top / bottom                  = 2.5 / 5.51                    = 0.45
score   = TF part · IDF                 = 0.45 · 0.88                   ≈ 0.40
```

**Result: D1 scores 1.62, D5 scores 0.40 — the short focused chunk wins by ~4×**,
even though both contain the word. That's Rule 2 (more mentions help) and Rule 3
(long docs get penalized) working together.

---

## What the examples show, in one line

- **Rule 1 (rarity):** rare exact terms get huge weight → BM25 locks onto them.
- **Rule 2 (count):** more mentions score higher, but with diminishing returns.
- **Rule 3 (length):** a focused chunk beats a long page with a passing mention.

You now know everything the formula is doing — it's just these three rules,
multiplied together and summed over your search words.
