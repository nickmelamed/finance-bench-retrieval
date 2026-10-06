# FinanceBench retrieval evaluation

Compares four retrievers (BM25, dense, hybrid RRF, agentic Claude tool-use
loop) on the 150-question FinanceBench open-source set. "Good" means the
accuracy, retrieval and token-cost numbers in the README can be traced to a
real run and the comparison is fair to every method.

The full design lives in docs/SPEC.md. Read the relevant section before
changing anything it covers. Current status and next steps are in
PROGRESS.md. Design decisions and their reasons are in docs/DECISIONS.md.

## Non-negotiable rules

1. Never edit `data/financebench/*` or the gold chunk IDs to move a score.
2. Token efficiency counts all LLM cost: agentic loop, answering and judge.
   Never exclude tokens to flatter a method.
3. Every number in the README or dashboard comes from a run in
   `outputs/runs/`. Disclose the agentic cost premium.
4. All four methods get the same questions, top_k, QA model and prompts.
5. Grade deterministically first and use the LLM judge only as a fallback.
   Never loosen grading to raise a method's score.
6. Tests never call Anthropic or Qdrant. Never commit `.env` or keys.
7. `make align` runs after `make index`, because indexing drops gold IDs.
8. Never weaken a test, lint rule, or check to make it pass.
9. Ask before changing anything in `.claude/protected-paths`, the evaluation
   design, dependencies, CI, or hooks.

SPEC section 7 lists known bugs. Do not fix one unless asked, and then
write the failing test first.

## Commands

```bash
uv venv && source .venv/bin/activate && uv pip install -e .
make test         # pytest, no live API or Qdrant
make lint         # ruff check
make agent-check  # the fast checks the Stop hook runs
make numbers      # README numbers against dashboard/data.js
make evaluate     # real run, calls the API and costs money. Ask first.
```

## Where things live

- `src/finance_bench/`: ingest, retrieval, llm, evaluation, experiments,
  visualizations. `tests/`: tests with synthetic fixtures.
- `configs/`: YAML for experiment, retrievers and prompts.
- `outputs/` is gitignored. `dashboard/data.js` is generated from it.
- `docs/SPEC.md`: design spec. `docs/DECISIONS.md`: decision log.
- `.claude/rules/`: path-scoped rules that load when you touch those files.

## How to work here

- Plan before multi-file changes. Write the plan down and wait for approval
  when the task spans several modules or touches the spec.
- A task is done when the Stop hook's checks pass and you have shown the
  output. Show evidence (commands and results), not claims.
- Commit in small atomic Conventional Commits (`type(scope): subject`). Code
  and its tests go in the same commit.
- You may branch and commit locally. Ask before pushing, opening or merging
  pull requests, tagging, or changing dependencies.
- If you repeat a multi-step procedure, propose a skill for it in
  `.claude/skills/` and wait for approval.
