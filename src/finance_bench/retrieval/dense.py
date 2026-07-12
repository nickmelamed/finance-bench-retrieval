import os

from dotenv import load_dotenv

from finance_bench.retrieval.embeddings import EmbeddingModel

from finance_bench.retrieval.base import BaseRetriever
from finance_bench.retrieval.qdrant_client import QdrantManager

from finance_bench.types.schemas import (
    RetrievalResult,
    DenseConfig,
)

load_dotenv()


class DenseRetriever(BaseRetriever):
    """
    Dense vector retrieval using Qdrant.
    """

    _embedder = None

    def __init__(
        self,
        config: DenseConfig,
    ):

        super().__init__(
            top_k=config.top_k
        )

        self.config = config

        self.collection_name = os.getenv(
            "COLLECTION_NAME",
            "financebench",
        )

        self.qdrant = QdrantManager()

        # singleton embedding model
        if DenseRetriever._embedder is None:

            DenseRetriever._embedder = (
                EmbeddingModel()
            )

        self.embedder = (
            DenseRetriever._embedder
        )

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        query_vector = (
            self.embedder
            .embed([query])[0]
            .tolist()
        )

        response = self.qdrant.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.top_k,
        )

        hits = response.points

        results = []

        for hit in hits:

            payload = hit.payload or {}

            metadata = {
                k: v
                for k, v in payload.items()
                if k != "text"
            }

            results.append(
                RetrievalResult(
                    chunk_id=str(hit.id),
                    text=payload.get(
                        "text",
                        "",
                    ),
                    score=float(hit.score),
                    metadata=metadata,
                    retrieval_method="dense",
                )
            )

        return results