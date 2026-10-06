---
name: run-evaluation
description: Run the full four-method evaluation and refresh the dashboard. Only the owner starts this.
disable-model-invocation: true
---

This spends Anthropic API money and needs Qdrant. Start by telling the owner
roughly what it will run, then wait for a yes.

1. Confirm `.env` has the three keys set (check names only, never print values)
   and that the working tree is clean.
2. `make qdrant-up`, then `make index` and `make align` in that order, since
   indexing rewrites `financebench_examples.json` and drops gold chunk IDs.
3. Confirm gold chunk IDs survived: count the questions in
   `data/processed/financebench_examples.json` that have `gold_chunk_ids`.
   Stop if the count dropped to zero.
4. `make evaluate`. Report the new run folder under `outputs/runs/`.
5. `make dashboard`, then `make numbers`. If `make numbers` fails, the README
   table is stale. Show the diff between the table and the new run and ask
   the owner before editing the README.
