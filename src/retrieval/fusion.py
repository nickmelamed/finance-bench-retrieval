from collections import defaultdict

from src.types.schemas import RetrievalResult, HybridConfig
from src.config.loaders import load_yaml_config

config = load_yaml_config(
    "retrieval/hybrid.yaml",
    HybridConfig
)

K = config.rrf_k


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Robust hybrid retrieval fusion strategy that avoids
    score normalization instability.
    """

    def __init__(
        self,
    ):
        self.k = K

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
    ) -> list[RetrievalResult]:
        fused_scores = defaultdict(float)

        lookup = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                fused_scores[result.chunk_id] += (
                    1.0 / (self.k + rank + 1)
                )

                lookup[result.chunk_id] = result

        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True,
        )

        fused_results = []

        for chunk_id in ranked_ids:
            original = lookup[chunk_id]

            fused_results.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=float(
                        fused_scores[chunk_id]
                    ),
                    retrieval_method="hybrid",
                    metadata=original.metadata,
                )
            )

        return fused_results