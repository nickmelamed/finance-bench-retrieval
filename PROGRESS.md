# Progress

Current phase and what is left.

## Now

- [ ] Review and merge the two pull requests. #2 (`fix/section-7-bugs`) is
      stacked on #1 (`chore/agent-standards`), so merge #1 first, then
      retarget #2 to `main`.
- [ ] Wait for CI on both. It had not reported when the branches were pushed.

## Next

- [ ] Confirm or correct the `[inferred]` items in docs/SPEC.md and edit
      docs/DECISIONS.md into your own words.
- [ ] Fill in SPEC section 6 (out of scope). It is unconfirmed.
- [ ] Decide whether the agentic retriever should run once per question.
      It currently runs twice, once for retrieval metrics and once for QA,
      and the second pass is uncached. Running it once would halve the
      time (about 2 hours) and the cost of a full run.
- [ ] Decide what to do with `reports/paper/`, which is empty and untracked.
- [ ] Consider a newer model than `claude-sonnet-4-6` in
      `configs/experiment.yaml`. Any change means a new `make evaluate`.

## Done

- Retrofit of the repo to the agent standards, with draft spec and decisions.
- Style cleanup, characterization tests and ruff clean-up. The Stop gate
  runs the style check, pytest, ruff and mypy.
- Dev dependency group (ruff, mypy, hypothesis). Empty notebooks removed.
- Fixes for SPEC 7.1 to 7.13, including agentic trimming, corpus metadata,
  the `temperature` break on `anthropic` 1.x, mypy and `make format`.
- Qdrant index rebuilt locally and gold alignment regenerated.
- `make evaluate` run `20261006_171209` after the fixes. The dashboard and
  README table are updated, and `make numbers` passes against both
  `dashboard/data.js` and the run folder.
- Qdrant stopped with `make qdrant-down`. Its data volume is kept.

## Open questions for the owner

- None blocking. See the Next list for decisions.
