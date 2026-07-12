# FinanceBench Retrieval Evaluation

A retrieval evaluation framework comparing four retrieval strategies on the
FinanceBench benchmark:

- **BM25** — sparse lexical retrieval
- **Dense** — vector retrieval (Qdrant + BGE-large embeddings)
- **Hybrid** — reciprocal rank fusion of BM25 + dense
- **Agentic** — a tool-using Claude agent that decides what to search, when
  to expand context, when to rerank, and when it has enough evidence to
  stop

---

# Objective

Evaluate retrieval systems on both answer accuracy and cost.

Primary metric:

Token Efficiency = Total Tokens / Correct Answers

Also tracked: recall@k, hit rate, and MRR against gold evidence chunks —
i.e. not just "did the final answer come out right," but "did the
retriever actually find the right passage."

---

# System Architecture

Question
→ Retriever (BM25 / Dense / Hybrid / Agentic)
→ Retrieved Chunks
→ Claude (answer generation)
→ Correctness Judge (deterministic check → LLM judge fallback)
→ Token Accounting
→ Bootstrap CI + Failure Analysis
→ Results Dashboard

---

# Features

## Retrieval

- Sparse (BM25), dense (Qdrant), and reciprocal-rank-fusion hybrid
  retrieval
- **Agentic retrieval**: a real Claude tool-use loop, not a heuristic. The
  agent has `lexical_search`, `semantic_search`, `get_neighbors`, and
  `rerank` tools, and explicitly calls `submit_evidence` when it decides
  it has enough context — bounded by a turn budget, with a fallback path
  if it runs out of turns without submitting. Tuned for cost via
  Anthropic prompt caching (the growing tool-use conversation is cached
  incrementally turn-to-turn) and context trimming (older tool results
  collapse to just their chunk IDs once they age out of the active
  window).

## Evaluation

- Claude-based correctness grading — a cheap deterministic
  normalized/numeric match first, falling back to an LLM judge only when
  that's ambiguous
- QA generation and judging are submitted via the **Anthropic Message
  Batches API** (50% cheaper than synchronous calls) for all four
  methods — this doesn't apply to the agentic retriever's own tool-use
  loop, since each turn depends on the last and can't be batched
- Bootstrap confidence intervals (10,000 resamples, percentile CIs)
- Retrieval-quality metrics (recall@k, hit rate, MRR) against gold
  evidence chunks aligned via fuzzy matching
- Token accounting (prompt/completion/retrieval tokens, cache-aware)
- Failure mode classification
- Per-run experiment tracking (`outputs/runs/<timestamp>/`)

## Visualization

- A self-contained **results dashboard** (`dashboard/index.html`) — open
  it directly in a browser, no server required. Accuracy with bootstrap
  CI, a Pareto frontier (accuracy vs. token efficiency), retrieval
  quality bars, failure mode breakdowns, and a searchable/filterable
  per-question drill-down table with each method's generated answer and
  retrieved chunk IDs. Supports light/dark mode and comparing across
  multiple past runs via a run picker.
- Legacy matplotlib plots (`src/finance_bench/visualizations/plots.py`)
  still available for accuracy CI / Pareto / failure-mode PNGs.

---

# Real results

From the most recent full run (`make evaluate`), 150 FinanceBench
questions:

| Method  | Accuracy | Recall@k | Tokens/correct |
|---------|----------|----------|-----------------|
| bm25    | 46.0%    | 24.0%    | 7,931           |
| dense   | 84.0%    | 71.7%    | 4,895           |
| hybrid  | 79.3%    | 63.3%    | 4,942           |
| agentic | **92.7%**| **92.0%**| 20,849          |

The agentic retriever leads on both accuracy and retrieval quality by a
wide margin, at a real (and disclosed, not hidden) token-cost premium —
see the dashboard for the full breakdown, including how that cost
tradeoff looks per-question.

---

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Environmental Variables

Copy:

```bash
cp .env.example .env
```

Then populate:
* ANTHROPIC_API_KEY
* QDRANT_URL
* QDRANT_API_KEY

### Setup

```bash
make full
```

runs the whole pipeline end to end: start Qdrant, ingest + index the
corpus, align gold evidence to chunk IDs, run the full evaluation, and
regenerate the dashboard. If the corpus is already indexed, `make
evaluate` alone re-runs just the evaluation, and `make dashboard`
regenerates the dashboard from whatever's in `outputs/runs/`.

Note the ordering matters if running steps individually: `make index`
rebuilds the raw examples file from scratch, so `make align` (which adds
gold chunk IDs) must run *after* it, not before.

### Tests + CI

```bash
make test    # pytest - fast, no live API/Qdrant dependency
make lint    # ruff
make check   # both
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on every
push/PR.

## Repo Structure

* `data` → original dataset and generated chunks
* `dashboard` → the static results dashboard (`index.html` + generated
  `data.js`)
* `src/finance_bench` → the installable `finance_bench` package
  * `ingest` → chunking + preprocessing
  * `retrieval` → retrieval backends, including the agentic tool-use loop
  * `llm` → Anthropic client wrappers (single-shot, batch, and
    multi-turn tool-use), caching, token tracking
  * `evaluation` → metrics, gold alignment, bootstrap statistics,
    correctness grading
  * `visualizations` → dashboard data export + legacy matplotlib plots
* `outputs` → experiment artifacts (gitignored)

## Reproducibility

All experiments:
* use deterministic seeds
* cache LLM responses on disk (content-hashed, model + max_tokens aware)
* additionally use Anthropic prompt caching within the agentic retriever's
  multi-turn conversations
* log token usage per question, including retrieval-phase LLM cost
* use fixed prompts (`configs/prompts/`)
* use structured, Pydantic-validated configs (`configs/`)
