import re

from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever

from src.types.schemas import (
    RetrievalResult,
    BM25Config,
    DocumentChunk,
)


class BM25Retriever(BaseRetriever):
    """
    Sparse lexical retrieval using BM25.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        config: BM25Config,
    ):

        super().__init__(
            top_k=config.top_k
        )

        self.config = config

        self.chunks = chunks

        self.tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=config.k1,
            b=config.b,
        )

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        text = text.lower()

        text = text.replace("$", " dollar ")

        text = text.replace("%", " percent ")

        text = re.sub(
            r"[^a-z0-9.\-% ]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.split()

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        tokenized_query = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_k]

        results = []

        for chunk, score in ranked:

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    retrieval_method="bm25",
                    metadata={
                        "document_id": chunk.document_id,
                        **chunk.metadata,
                    },
                )
            )

        return results