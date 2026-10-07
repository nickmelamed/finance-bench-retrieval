---
name: rebuild-index
description: Rebuild the chunks, the Qdrant index and the gold chunk IDs in the required order. Only the owner starts this.
disable-model-invocation: true
---

Changing chunking or the index invalidates the gold alignment and earlier
results. Confirm with the owner first.

1. `make qdrant-up`.
2. `make index`. This rewrites `data/processed/chunks.json` and
   `financebench_examples.json` and uploads embeddings to Qdrant.
3. `make align`. It must come after `index`.
4. Show how many questions have `gold_chunk_ids` before and after, and the
   chunk count before and after.
5. Remind the owner that earlier runs used the old chunk IDs, so a new
   `make evaluate` is needed before any README number can be trusted.
