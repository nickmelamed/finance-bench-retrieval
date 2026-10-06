
def recall_at_k(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: list[str],
    k: int | None = None,
) -> float:

    if not gold_chunk_ids:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])

    gold = set(gold_chunk_ids)

    hits = len(
        retrieved.intersection(gold)
    )

    return hits / len(gold)


def hit_rate(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: list[str],
    k: int | None = None,
) -> float:

    retrieved = set(retrieved_chunk_ids[:k])

    gold = set(gold_chunk_ids)

    return float(
        len(
            retrieved.intersection(gold)
        ) > 0
    )


def mean_reciprocal_rank(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: list[str],
    k: int | None = None,
) -> float:

    gold = set(gold_chunk_ids)

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids[:k],
        start=1,
    ):

        if chunk_id in gold:

            return 1.0 / rank

    return 0.0