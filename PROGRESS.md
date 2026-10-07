# Progress

Current phase and what is left.

## Now

- [ ] Review the two pull requests. `fix/section-7-bugs` is stacked on
      `chore/agent-standards`.

## Next

- [ ] Confirm or correct the `[inferred]` items in docs/SPEC.md and edit
      docs/DECISIONS.md into your own words.
- [ ] Decide whether the agentic retriever should run once per question.
      It currently runs twice, once for retrieval metrics and once for QA,
      and the second pass is uncached.

## Done

- Retrofit of the repo to the agent standards, with draft spec and decisions.
- Style cleanup, characterization tests and ruff clean-up. The gate runs the
  style check, ruff and pytest.
- Dev dependency group (ruff, mypy, hypothesis). Empty notebooks removed.
- Fixes for SPEC 7.1 to 7.12, including agentic trimming, corpus metadata,
  mypy (now gated) and `make format`.
- `make evaluate` run `20261006_171209` after the fixes, dashboard
  regenerated and README table updated. `make numbers` passes against both
  `dashboard/data.js` and the run folder.
- Qdrant index rebuilt locally (Docker) and gold alignment regenerated.

## Open questions for the owner

- Which source of truth should the README numbers trace to? They match
  `dashboard/data.js`, but `outputs/runs/` has results for only some runs.
- SPEC 6 (out of scope) is unconfirmed.
- `reports/paper/` is empty and untracked.
