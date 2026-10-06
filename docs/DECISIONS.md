# Decisions

Drafted from the code, not from your own account. Edit each entry into
your words and delete any that are not decisions you made. Newest last.

## D-001: Deterministic grading before the LLM judge (2026-07)

Context. Grading every answer with Claude costs tokens and adds noise.

Decision. Normalize and compare strings or numbers first. Call the judge
only when that fails.

Why. Cheaper and repeatable for the clear cases. Rejected: judge-only.

Consequences. The deterministic rules decide accuracy for most answers, so
their leniency matters (see SPEC 7.1, 7.2).

## D-002: Reciprocal rank fusion for the hybrid retriever (2026-07)

Context. BM25 and dense scores are on different scales.

Decision. Fuse by rank with `rrf_k=60`. Config also allows `weighted_sum`
but only RRF is implemented.

Why. Avoids score normalization. Rejected: weighted score sum.

Consequences. No tuning of fusion weights. Hybrid can only be as good as
the ranks it is fed.

## D-003: A real tool-use loop for the agentic retriever (2026-07)

Context. A heuristic "agent" would not show whether model-driven search helps.

Decision. Claude chooses among five tools with a hard cap of 4 turns, a
forced submit on the last turn, and a fallback to best-seen chunks.

Why. The turn cap bounds cost. Dropping unseen chunk IDs blocks
hallucinated evidence.

Consequences. Agentic costs several times more tokens per correct answer
than the others (README: 20,849 vs about 4,900), and its loop cannot be
batched.

## D-004: Batch API for answering and judging (2026-07)

Context. Each method answers 150 questions.

Decision. Submit QA generation and judge calls as one Message Batch per
method (50% cheaper). Retrieval for the agentic method stays synchronous.

Why. Cost. Rejected: synchronous calls for everything.

Consequences. Runs take longer to return. Token counts come from batch
usage.

## D-005: Section-aware word-window chunking with stable IDs (2026-05)

Context. Filings have headings, and gold evidence has to be mapped to
chunks repeatably.

Decision. Split at heading patterns, then 350-word windows with 50 overlap.
Chunk ID is uuid5 of document, section and word offsets.

Why. IDs survive re-chunking with the same parameters, so gold alignment
stays valid. Rejected: random UUIDs.

Consequences. Changing chunk size or the heading regex invalidates the gold
alignment. Rerun `make index` then `make align`.

## D-006: Gold chunks by fuzzy matching, one chunk per evidence passage (2026-07)

Context. FinanceBench gives evidence text, not chunk IDs.

Decision. Match each passage to its best chunk with rapidfuzz at threshold
80 and treat that chunk as gold.

Why. Lets recall, hit rate and MRR be computed without manual labels.

Consequences. Evidence spanning several chunks is represented by only one.
Recall depends on the threshold and the chunker.

## D-007: Disk cache plus prompt caching (2026-07)

Context. Re-running 150 questions x 4 methods is expensive.

Decision. Cache single-shot and batch responses on disk by content, model
and max_tokens. Use Anthropic prompt caching inside the agentic loop and
trim stale tool results.

Why. Cheaper repeat runs and a cheaper agent loop.

Consequences. A cache hit reports zero tokens, so token totals depend on
cache state (see SPEC 7.4).

## D-008: Static dashboard and src layout (2026-07)

Context. Results needed to be viewable without a server.

Decision. `dashboard/index.html` reads a generated `data.js` built from
`outputs/runs/*`. The package lives in `src/finance_bench`.

Why. Open the file in a browser. Rejected: a served app.

Consequences. `data.js` is generated and committed (1.7 MB). It must be
regenerated whenever runs change.
