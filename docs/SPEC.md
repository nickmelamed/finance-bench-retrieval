# Spec: FinanceBench retrieval evaluation

Status: draft. Tags: **[confirmed]** was stated by the owner, **[code]** is
what the code does, which may differ from intent, and **[inferred]** is
intent that has not been confirmed. Section numbers are stable so other files
can cite them.

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
An empty answer is never correct. Correct if normalized strings are equal,
if either contains the other as whole tokens, or if every number in the gold
answer has a number in the answer within 2%. Otherwise the LLM judge (JSON
verdict) decides. **[code]**

3.6 Metrics **[code]**
- Accuracy: mean correctness. Bootstrap CI: 10,000 resamples, 95% percentile,
  drawn from a generator seeded with the experiment seed.
- Retrieval: recall@k = |retrieved[:k] ∩ gold| / |gold|, plus hit rate and
  MRR, with k the retriever's top_k. Computed only over questions that have
  gold chunk IDs. The result records that count as `retrieval_questions`
  next to `qa_questions`.
- Token efficiency = total tokens / correct answers (inf if none correct).
  Total = retrieval-loop prompt+completion + QA prompt+completion + judge
  prompt+completion.
- Failure modes: heuristic labels for incorrect answers
  (`retrieval_failure`, `empty_generation`, `numeric_reasoning_failure`,
  `semantic_mismatch`). Retrieval failure means nothing was retrieved or
  none of the gold chunks were.

3.7 Reproducibility **[code, README]**. Responses are disk-cached by
content, model and max_tokens (`outputs/cache`). A cache hit reports the
usage stored with the entry, and entries written before usage was stored
report zero. Temperature is 0. Runs are written to `outputs/runs/<timestamp>/`.

## 4. Reported claims

**[confirmed]** Accuracy with bootstrap CI, recall@k / hit rate / MRR,
tokens per correct answer, and failure-mode breakdown. Shown in the README
table, the dashboard (`dashboard/index.html`, `data.js`) and the matplotlib
plots.

## 5. Check commands

**[confirmed]** Gate: `ruff check .` and `pytest`. mypy joins later, once
its errors are triaged. Install: `uv venv && uv pip install -e ".[dev]"`.

## 6. Out of scope / not covered

**[unconfirmed]** Nothing here is decided. Candidates: live API and Qdrant
behavior, the generated `dashboard/data.js`, and the empty `reports/paper/`.

## 7. Defects found and their status

Confirmed as bugs on 2026-10-06 and fixed on branch `fix/section-7-bugs`.
Each fix has a test in `tests/`.

- **7.1 Empty answers were graded correct.** An empty string is a substring
  of every gold answer. Fixed: empty answers fail the deterministic check.
- **7.2 Loose matching.** Substring containment matched `100` inside
  `100,000`, and one matching number was enough. Fixed: containment needs
  whole tokens and every gold number must match. Years within 2% of each
  other still match, which the tolerance cannot tell apart.
- **7.3 Seeds were not used.** Fixed: `BootstrapConfig.seed` (default 42) seeds
  a generator, set from `seed` in `experiment.yaml`.
- **7.4 Token double count.** A word count of the context was added to input
  tokens that already included it. Cache hits also reported zero. Fixed:
  the word count is gone and cache entries keep their usage. Clear
  `outputs/cache` before a run you will report, since older entries have no
  stored usage.
- **7.5 Recall was not truncated to k.** Fixed: the metrics take `k` and the
  evaluation passes the retriever's `top_k`.
- **7.6 Different question sets.** All 150 questions currently have gold
  chunks, so nothing differs today. Fixed: the result records both counts
  and a warning prints when they differ.
- **7.7 Failure labels ignored the gold chunks.** Fixed: a failure with no
  gold chunk retrieved is a retrieval failure. The digit heuristic for
  `numeric_reasoning_failure` remains.
- **7.8 mypy findings** at `agentic.py:83`, `:133`, `:397` and
  `run_all.py:395`. Fixed, along with the rest. `mypy` now reports no
  issues and is part of the gate and CI.
- **7.9 Section splitting never fired.** Whitespace was collapsed before the
  heading patterns, which need newlines. Fixed. On the current corpus no
  document has a matching heading, so chunk IDs and text are unchanged and
  the gold alignment stays valid. A corpus with headings will now chunk
  differently, which would need `make index` and `make align`.

- **7.10 Trimming never ran.** `keep_recent_tool_turns` was 5 and
  `max_tool_calls` is 4, so `_collapse_stale_tool_results` always returned
  early. Fixed: the shipped value is 2, so the first tool result collapses
  before the final turn. Agentic token counts will change.
- **7.11 Corpus metadata held the answer.** Chunk metadata carried
  `question`, `gold_answer` and `evidence_pages`. Prompts and BM25 used only
  chunk text, so results were unaffected. Fixed: those fields are no longer
  copied, and `chunks.json` and the Qdrant payloads were regenerated. Chunk
  ids, text and gold chunk ids are unchanged.
- **7.12 `temperature` broke on `anthropic` 1.x.** `messages.create` no
  longer takes `temperature`, and `pyproject.toml` allows 1.x, so a fresh
  install raised `TypeError` on the first call. Fixed: temperature goes
  through `extra_body`, which works on 0.x and 1.x.

- **7.13 The README results predated these fixes.** Fixed: the table comes
  from run `20261006_171209`, made after every fix above. Accuracy fell for
  every method, most for bm25 (46.0% to 28.0%) and least for dense and
  agentic. The ranking is unchanged and agentic still leads.
