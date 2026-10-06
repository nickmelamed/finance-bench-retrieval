---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Python

- Run commands from the repo root inside `.venv`. Configs and data load
  from relative paths such as `configs/` and `data/processed/`.
- Importing `finance_bench.experiments.run_all` and
  `finance_bench.retrieval.fusion` loads YAML at import time. Keep new
  module-level config loads out of code that tests import.
- Anything that calls Anthropic or Qdrant needs a fake in tests. `ClaudeClient`
  caches to `outputs/cache`, so a test must not rely on that cache.
- `ruff check .` is the lint command and the repo has no formatter step.
  Do not reformat files you are not changing.
- Config models use `extra="forbid"`, so a new YAML key needs a schema field.
