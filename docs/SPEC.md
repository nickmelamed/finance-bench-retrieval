# Spec: FinanceBench retrieval evaluation

Status: DRAFT. Each item is tagged **[confirmed]** (you told me), **[code]**
(what the code does today, which may differ from intent), or **[inferred]**
(my guess at intent. Please correct). Section numbers are stable so other
files can cite them.

## 1. Purpose

**[inferred]** Compare four retrieval strategies (BM25, dense, hybrid,
agentic) on the 150-question FinanceBench open-source set, on both answer
quality and token cost, and report the comparison honestly, including the
agentic method's cost premium.

Primary metric **[code, README]** is tokens per correct answer
(total tokens / correct answers). Also tracked: accuracy, recall@k, hit
rate, MRR.

## 2. Rules that must hold

1. **[confirmed]** Gold/eval data is read-only. Do not edit
   `data/financebench/*` or the gold chunk alignment to improve a score.
2. **[confirmed]** Token efficiency counts all LLM cost, including the
   agentic retriever's tool-loop tokens, and is cache-aware. Never exclude
   tokens to flatter a method.
3. **[confirmed]** Every number in the README or dashboard comes from a real
   run in `outputs/runs/`. The cost premium is always disclosed.
4. **[inferred]** All four methods get the same top_k, QA model, prompts,
   and questions. Config differences are limited to each retriever's own
   parameters.
5. **[inferred]** Grading is deterministic first. The LLM judge is a
   fallback only for answers the deterministic check rejects.
6. **[inferred]** Tests never call Anthropic or Qdrant. Secrets (`.env`)
   are never committed.
7. **[inferred]** `make align` runs after `make index`, because indexing
   rewrites `financebench_examples.json` and drops gold chunk IDs.

## 3. Pipeline

**[code]** question -> retriever -> top_k chunks -> Claude answers from
context only -> grader -> token accounting -> bootstrap CI and failure
analysis -> dashboard.

3.1 Ingest. Whitespace-normalize each document, split at SEC-style section
headings, then cut each section into word windows (350 words, 50 overlap).
Chunk IDs are `uuid5` of document, section, start word, end word, so they
are stable across runs. **[code]**

3.2 Gold alignment. For each gold evidence passage, pick the single best
chunk by `max(token_set_ratio, partial_ratio)` (rapidfuzz) over the first
2,000 characters, accepted at score >= 80. Questions with no match have no
gold chunks and are skipped in retrieval metrics. **[code]**

3.3 Retrievers **[code]**
- BM25: `rank_bm25` Okapi, k1=1.5, b=0.75.
- Dense: BGE-large-en-v1.5 embeddings in Qdrant, cosine.
- Hybrid: reciprocal rank fusion of BM25 and dense, `rrf_k=60`, cut to top_k.
- Agentic: Claude tool-use loop with `lexical_search`, `semantic_search`,
  `get_neighbors`, `rerank` (cross-encoder), `submit_evidence`. At most 4
  turns. The last turn is forced to submit. Chunk IDs not seen via a tool
  are dropped. Fallback when nothing valid is submitted: most-frequently-seen,
  then highest-score chunks. Older tool results collapse to chunk IDs after
  5 turns. Prompt caching is applied to system, tools and the message tail.

3.4 Answering. One prompt per question: "answer ONLY from context, else
'Insufficient information.'". Sent as one Message Batch per method.
Temperature 0, max_tokens 512, model `claude-sonnet-4-6`. **[code]**

3.5 Grading. Normalize (lowercase, strip `,` and `$`, `%` -> " percent ").
Correct if normalized strings are equal or either contains the other, or any
number in the answer is within 2% of any number in the gold. Otherwise the
LLM judge (JSON verdict) decides. **[code]**

3.6 Metrics **[code]**
- Accuracy: mean correctness. Bootstrap CI: 10,000 resamples, 95% percentile.
- Retrieval: recall@k = |retrieved ∩ gold| / |gold|, plus hit rate and MRR.
  Computed only over questions that have gold chunk IDs.
- Token efficiency = total tokens / correct answers (inf if none correct).
  Total = retrieval-loop prompt+completion + QA prompt+completion + judge
  prompt+completion + a word count of each context.
- Failure modes: heuristic labels for incorrect answers
  (`retrieval_failure`, `empty_generation`, `numeric_reasoning_failure`,
  `semantic_mismatch`).

3.7 Reproducibility **[code, README]**. Responses are disk-cached by
content, model and max_tokens (`outputs/cache`). A cache hit reports zero
tokens. Temperature is 0. Runs are written to `outputs/runs/<timestamp>/`.

## 4. Reported claims

**[confirmed]** Accuracy with bootstrap CI, recall@k / hit rate / MRR,
tokens per correct answer, and failure-mode breakdown. Shown in the README
table, the dashboard (`dashboard/index.html`, `data.js`) and the matplotlib
plots.

## 5. Check commands

**[confirmed]** Gate: `ruff check .` and `pytest`. mypy joins later, once
its errors are triaged. Install: `uv venv && uv pip install -e .`.

## 6. Out of scope / not covered

**[unconfirmed]** Nothing here is decided. Candidates I noticed, none
confirmed: live API and Qdrant behavior, the two empty notebooks, the
generated `dashboard/data.js`, and the empty `reports/paper/`. Fill this in
or delete it.

## 7. Where the code and the stated intent disagree

Findings from reading the code. You confirmed these are bugs to fix
(2026-10-06). I have not changed any code yet. Fix order and timing are
open (see PROGRESS.md once written). The wording of each intended behavior
below is still my inference.

- **7.1 Empty answers are graded correct.** `pred_norm in gold_norm` is true
  for an empty string, so an empty generated answer passes the deterministic
  check. Verified directly. Bears on rule 5 and every accuracy figure.
- **7.2 Loose numeric match.** Any number in the answer within 2% of any
  number in the gold passes, so a stray year or unrelated figure can match.
  Also `"3.2"` matches `"The year 2023 saw 3.19"`. Verified.
- **7.3 Seeds are not used.** `seed: 42` is in config but nothing seeds
  numpy, so bootstrap CIs vary between runs. README says "deterministic
  seeds."
- **7.4 Possible token double count.** QA input tokens already include the
  context, and a whitespace word count of the context is added again as
  `retrieval_tokens` (both in the batch path). Cache hits count as zero
  tokens, so re-runs understate cost. Bears on rule 2.
- **7.5 Recall@k is not truncated to k.** `recall_at_k` uses the whole
  retrieved list. It equals recall@top_k only because retrievers return
  top_k items.
- **7.6 Retrieval metrics and QA cover different question sets.** Retrieval
  metrics skip questions with no gold chunks. Accuracy covers all of them.
- **7.7 Failure labels are heuristic.** "numeric_reasoning_failure" is
  assigned whenever the gold answer contains any digit, and nothing compares
  retrieved chunks to gold chunks.
- **7.8 mypy findings** at `agentic.py:83`, `:133`, `:397` and
  `run_all.py:395` may be real defects (see Phase 0).
