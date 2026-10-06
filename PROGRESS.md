# Progress

Current phase and what is left.

## Now

- [ ] Review the two pull requests. `fix/section-7-bugs` is stacked on
      `chore/agent-standards`.
- [ ] Decide on SPEC 7.10 (agentic trimming never runs) and 7.11.

## Next

- [ ] Clear `outputs/cache`, then run `make evaluate` and update the README
      table. It costs API money, and the table predates the fixes.
- [ ] Triage the remaining 28 mypy errors, then add `mypy src` to
      `.claude/gate-commands`.
- [ ] Confirm or correct the `[inferred]` items in docs/SPEC.md and edit
      docs/DECISIONS.md into your own words.
- [ ] Fix the leftover `make format` target, which calls black but black is
      no longer installed.

## Done

- Retrofit of the repo to the agent standards, with draft spec and decisions.
- Style cleanup, characterization tests and ruff clean-up. The gate runs the
  style check, ruff and pytest.
- Dev dependency group (ruff, mypy, hypothesis). Empty notebooks removed.
- Fixes for SPEC 7.1 to 7.9.

## Open questions for the owner

- Which source of truth should the README numbers trace to? They match
  `dashboard/data.js`, but `outputs/runs/` has results for only some runs.
- SPEC 6 (out of scope) is unconfirmed.
- `reports/paper/` is empty and untracked.
