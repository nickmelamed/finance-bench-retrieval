# Progress

Current phase and what is left.

## Now

- [ ] Put a valid `ANTHROPIC_API_KEY` in `.env`. The last `make evaluate`
      stopped at the first API call with a 401 (invalid key). Nothing was
      spent and no results were written.
- [ ] Run `make evaluate` and `make dashboard`, then update the README table
      so `make numbers` passes against the new run.

## Next

- [ ] Review the two pull requests. `fix/section-7-bugs` is stacked on
      `chore/agent-standards`.
- [ ] Confirm or correct the `[inferred]` items in docs/SPEC.md and edit
      docs/DECISIONS.md into your own words.

## Done

- Retrofit of the repo to the agent standards, with draft spec and decisions.
- Style cleanup, characterization tests and ruff clean-up. The gate runs the
  style check, ruff and pytest.
- Dev dependency group (ruff, mypy, hypothesis). Empty notebooks removed.
- Fixes for SPEC 7.1 to 7.12, including agentic trimming, corpus metadata,
  mypy (now gated) and `make format`.
- Qdrant index rebuilt locally (Docker) and gold alignment regenerated.

## Open questions for the owner

- Which source of truth should the README numbers trace to? They match
  `dashboard/data.js`, but `outputs/runs/` has results for only some runs.
- SPEC 6 (out of scope) is unconfirmed.
- `reports/paper/` is empty and untracked.
