from sentence_transformers import CrossEncoder

from finance_bench.types.schemas import RetrievalResult


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        pairs = [
            [query, result.text]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    retrieval_method=result.retrieval_method,
                    metadata=result.metadata,
                )
            )

        reranked.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return reranked[:top_k]