# Progress

Current phase and what is left.

## Now

- [ ] Review branch `chore/agent-standards` and decide whether to push it.
- [ ] Decide which SPEC section 7 bugs to fix, and in what order.

## Next

- [ ] Fix confirmed bugs on a separate branch, one `fix:` commit each. Delete
      the matching "current behavior" test and remove the `xfail` mark in the
      same commit (7.1 empty answers, 7.2 substring numbers, 7.3 bootstrap
      seed, 7.4 token double count).
- [ ] Re-run `make evaluate` after the fixes. It costs API money. The README
      table is stale until then.
- [ ] Triage the 45 mypy errors, then add `mypy src` to `.claude/gate-commands`.
- [ ] Add `ruff`, `mypy` and `hypothesis` as a dev dependency group in
      `pyproject.toml`. They are installed in `.venv` only.
- [ ] Confirm or correct the `[inferred]` items in docs/SPEC.md and edit
      docs/DECISIONS.md into your own words.

## Done

- Phase 0: assessment. Retrofit chosen over rebuild.
- Phase 1: draft docs/SPEC.md and docs/DECISIONS.md.
- Phase 2: agent tooling v2.0.1, CLAUDE.md, rules, protected paths, hooks,
  skills `run-evaluation`, `refresh-dashboard`, `rebuild-index`.
- Phase 3: style cleanup of README and four docstrings.
- Phase 4: 64 tests (60 pass, 4 strict xfail). Gate runs style and pytest.
- ruff fixes applied (`81169bf`).

## Open questions for the owner

- SPEC 7.9: section-aware chunking never fires (all 187 chunks have
  `section_index` 0). Is that a bug? Fixing it changes every chunk ID and
  invalidates gold alignment and all results.
- SPEC 6 is unconfirmed. What is out of scope?
- README numbers match `dashboard/data.js`, but `outputs/runs/` has result
  files for only 5 of the runs the README could cite. CLAUDE.md rule 3 says
  numbers come from `outputs/runs/`. Which is the source of truth?
- Seven ruff findings remain in `src/` (4 blind excepts, 1 try-except-pass,
  1 naive `datetime.now()`, 1 needless-bool) and 30 in `scripts/agent/`,
  which is installed tooling and must not be edited here. To gate on
  `ruff check .` I need your approval to exclude `scripts/agent` in
  `pyproject.toml` and to fix or ignore the seven.
- `notebooks/*.ipynb` are empty files. Keep or delete?
- `reports/paper/` is empty and untracked.
