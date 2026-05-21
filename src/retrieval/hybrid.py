from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import (
    ReciprocalRankFusion,
)

from src.config.loaders import load_yaml_config

from src.types.schemas import (
    RetrievalResult,
    HybridConfig,
    BM25Config,
    DenseConfig,
    DocumentChunk
)


class HybridRetriever(BaseRetriever):
    """
    Hybrid sparse+dense retrieval using RRF.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        hybrid_config: HybridConfig,
        bm25_config: BM25Config,
        dense_config: DenseConfig,
    ):

        super().__init__(
            top_k=hybrid_config.top_k
        )

        self.config = hybrid_config

        self.bm25 = BM25Retriever(
            chunks=chunks,
            config=bm25_config,
        )

        self.dense = DenseRetriever(
            config=dense_config,
        )

        self.rrf = ReciprocalRankFusion()

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        sparse_results = self.bm25.retrieve(
            query
        )

        dense_results = self.dense.retrieve(
            query
        )

        fused = self.rrf.fuse(
            [
                sparse_results,
                dense_results,
            ]
        )

        return fused[: self.top_k]