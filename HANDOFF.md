# Handoff

Everything needed to continue this project from another repository or
another session. State as of 2026-10-06.

## What this project is

A retrieval evaluation on the 150-question FinanceBench open-source set. It
compares BM25, dense (Qdrant plus BGE-large), hybrid (RRF) and an agentic
Claude tool-use retriever on accuracy, retrieval quality (recall@k, hit rate,
MRR) and tokens per correct answer. The design is in `docs/SPEC.md`, the
reasons for it in `docs/DECISIONS.md`, and status in `PROGRESS.md`.

## Where the work is

| | |
|---|---|
| Repo | `github.com/nickmelamed/finance-bench-retrieval` |
| PR #1 | `chore/agent-standards` into `main`. Agent tooling, docs, tests. |
| PR #2 | `fix/section-7-bugs` into `chore/agent-standards`. Bug fixes and new results. |

`main` has none of this yet. PR #2 contains all of PR #1, so to work on the
latest code check out `fix/section-7-bugs`. Merge #1 first, then retarget #2
to `main`.

## Latest results

Run `20261006_171209`, made after all fixes. These are in the README and in
`dashboard/data.js`.

| Method | Accuracy | Recall@k | Tokens/correct |
|---|---|---|---|
| bm25 | 28.0% | 24.0% | 8,884 |
| dense | 74.7% | 73.0% | 4,162 |
| hybrid | 68.0% | 63.3% | 4,118 |
| agentic | 86.7% | 93.3% | 20,679 |

The earlier README figures (46.0, 84.0, 79.3, 92.7% accuracy) were inflated by
the grading bugs in SPEC 7.1 and 7.2, so do not compare against them.

## What is not in git

- `.env` holds `ANTHROPIC_API_KEY`, `QDRANT_URL` and related names. See
  `.env.example`. Never commit it. The hooks block reading it.
- `outputs/` is gitignored. The raw run folder
  `outputs/runs/20261006_171209/evaluation_results.json` exists only on the
  original machine. Its numbers are preserved in `dashboard/data.js`. The
  LLM response cache is `outputs/cache`, and `outputs/cache_before_fixes`
  is the pre-fix cache, kept only in case it is wanted.
- Qdrant data lives in the Docker volume `finance-bench-retrieval_qdrant_data`
  on the original machine. On a new machine rebuild it with `make qdrant-up`,
  `make index`, then `make align`. Chunk ids and gold chunk ids are
  deterministic, so the rebuild matches what is committed.

## Set up

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env     # then fill in the keys
make agent-check         # style, ruff, mypy, pytest. No API or Qdrant needed.
```

Run commands from the repo root inside `.venv`. Configs and data load from
relative paths. Docker is only needed for Qdrant.

## Commands that cost money or need services

| Command | Needs | Notes |
|---|---|---|
| `make qdrant-up` / `make qdrant-down` | Docker | Down keeps the data volume. |
| `make index` then `make align` | Qdrant, model download | In that order. Index drops gold chunk ids and align restores them. |
| `make evaluate` | Anthropic key, Qdrant | About 2 hours of running time. Cost was not measured, my estimate is $15 to $25. |
| `make dashboard` | none | Rebuilds `dashboard/data.js` from `outputs/runs/*`. |
| `make numbers` | none | Checks README numbers against `dashboard/data.js`. |

Clear `outputs/cache` before a run you will report. Entries written before
the usage fix report zero tokens.

## Things that will surprise you

- `anthropic` 1.x removed the `temperature` keyword. The code passes it
  through `extra_body`. `tests/test_run_all_characterization.py` checks every
  call against the installed SDK signature.
- The agentic retriever runs its loop twice per question (retrieval metrics,
  then QA). The second pass is uncached. This is the main cost driver.
- Agentic context trimming now works because `keep_recent_tool_turns` is 2
  while the loop allows 4 turns. Keep it below `max_tool_calls - 1`.
- Importing `finance_bench.experiments.run_all` and `retrieval.fusion` loads
  YAML at import time, so tests that import them must run from the repo root.
- Chunk metadata must not carry questions, answers or evidence pages. A test
  in `tests/test_build_corpus.py` guards this.

## Agent tooling in this repo

`CLAUDE.md` holds the rules. `.claude/protected-paths` lists files that need
owner approval to edit, including `README.md`, `configs/`, `data/` and the
evaluation code. `scripts/agent/` is installed tooling, so do not edit it.
The Stop gate (`.claude/gate-commands`) runs the style check, pytest, ruff
and mypy before a turn can end. Repo skills are in `.claude/skills/`:
`run-evaluation`, `rebuild-index` (owner only) and `refresh-dashboard`.

## Open decisions

1. Run the agentic retriever once per question instead of twice.
2. Confirm the `[inferred]` items in `docs/SPEC.md` and rewrite
   `docs/DECISIONS.md` in your own words.
3. Fill in SPEC section 6 (out of scope).
4. Whether to try a newer model than `claude-sonnet-4-6`. It needs a new run.
5. What to do with the empty, untracked `reports/paper/`.

## Verified at handoff

`pytest` 74 passed, `ruff check .` and `mypy` clean, the Stop gate exits 0,
`make numbers` passes against the dashboard and the run folder, and
Qdrant is stopped.
