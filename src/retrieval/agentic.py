import re

from collections import defaultdict

from src.retrieval.base import BaseRetriever

from src.types.schemas import (
    RetrievalResult,
    AgenticConfig,
    DocumentChunk,
)


FINANCE_SYNONYMS = {
    "revenue": ["sales", "net sales"],
    "profit": ["net income", "earnings"],
    "buyback": ["share repurchase"],
    "shares": ["common stock"],
    "debt": ["liabilities"],
    "cash": ["cash equivalents"],
    "expenses": ["operating expenses"],
}


class AgenticGrepRetriever(BaseRetriever):
    """
    Lightweight iterative retrieval system.

    Features:
    - query expansion
    - iterative refinement
    - neighbor expansion
    - diversification
    """

    STOPWORDS = {
        "what",
        "were",
        "was",
        "the",
        "did",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "has",
        "had",
        "their",
        "they",
        "into",
        "during",
        "about",
    }

    def __init__(
        self,
        chunks: list[DocumentChunk],
        config: AgenticConfig,
    ):

        super().__init__(
            top_k=config.top_k
        )

        self.chunks = chunks
        self.config = config

        self.chunk_lookup = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

    def extract_search_terms(
        self,
        query: str,
    ) -> list[str]:

        query = query.lower()

        tokens = re.findall(
            r"\w+",
            query,
        )

        if self.config.remove_stopwords:

            tokens = [
                token
                for token in tokens
                if token not in self.STOPWORDS
            ]

        tokens = [
            token
            for token in tokens
            if len(token)
            >= self.config.min_term_length
        ]

        return list(
            dict.fromkeys(tokens)
        )

    def expand_terms(
        self,
        terms: list[str],
    ) -> list[str]:

        if not self.config.enable_query_expansion:
            return terms

        expanded = set(terms)

        if self.config.finance_synonym_expansion:

            for term in terms:

                if term in FINANCE_SYNONYMS:
                    expanded.update(
                        FINANCE_SYNONYMS[term]
                    )

        return list(expanded)[
            : self.config.max_search_queries
        ]

    def score_chunk(
        self,
        text: str,
        terms: list[str],
        query: str,
    ) -> float:

        text_lower = text.lower()

        score = 0.0

        for term in terms:

            if term in text_lower:
                score += (
                    self.config.term_match_weight
                )

        if query.lower() in text_lower:
            score += (
                self.config.exact_phrase_boost
            )

        numeric_query_terms = re.findall(
            r"\d+(?:\.\d+)?",
            query,
        )

        for num in numeric_query_terms:

            if num in text_lower:
                score += (
                    self.config.numeric_match_boost
                )

        return score

    def diversify(
        self,
        ranked: list[tuple],
    ) -> list[tuple]:

        if not self.config.diversify_results:
            return ranked

        counts = defaultdict(int)

        diversified = []

        for chunk, score in ranked:

            document_id = chunk.document_id

            if (
                counts[document_id]
                >= self.config.max_results_per_document
            ):
                continue

            diversified.append(
                (chunk, score)
            )

            counts[document_id] += 1

        return diversified

    def expand_neighbors(
        self,
        ranked: list[tuple],
    ) -> list[tuple]:

        if not self.config.expand_neighbors:
            return ranked

        expanded = list(ranked)

        seen = {
            chunk.chunk_id
            for chunk, _ in ranked
        }

        chunk_indices = {
            chunk.chunk_id: idx
            for idx, chunk in enumerate(
                self.chunks
            )
        }

        for chunk, score in ranked:

            idx = chunk_indices.get(
                chunk.chunk_id
            )

            if idx is None:
                continue

            for offset in range(
                -self.config.neighbor_window,
                self.config.neighbor_window + 1,
            ):

                if offset == 0:
                    continue

                neighbor_idx = idx + offset

                if (
                    neighbor_idx < 0
                    or neighbor_idx >= len(self.chunks)
                ):
                    continue

                neighbor = self.chunks[
                    neighbor_idx
                ]

                if (
                    neighbor.chunk_id
                    in seen
                ):
                    continue

                expanded.append(
                    (
                        neighbor,
                        score * 0.75,
                    )
                )

                seen.add(
                    neighbor.chunk_id
                )

        return expanded

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        all_ranked = []

        current_query = query

        for _ in range(
            self.config.max_iterations
        ):

            search_terms = (
                self.extract_search_terms(
                    current_query
                )
            )

            search_terms = (
                self.expand_terms(
                    search_terms
                )
            )

            scored = []

            for chunk in self.chunks:

                score = self.score_chunk(
                    text=chunk.text,
                    terms=search_terms,
                    query=current_query,
                )

                if (
                    score
                    >= self.config.min_relevance_score
                ):

                    scored.append(
                        (
                            chunk,
                            score,
                        )
                    )

            scored.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            scored = self.diversify(
                scored
            )

            all_ranked.extend(
                scored
            )

            if (
                len(scored)
                >= self.config.min_unique_results
            ):
                break

            current_query = (
                current_query
                + " financial statement"
            )

        all_ranked = self.expand_neighbors(
            all_ranked
        )

        deduped = {}

        for chunk, score in all_ranked:

            chunk_id = chunk.chunk_id

            if (
                chunk_id not in deduped
                or score
                > deduped[chunk_id][1]
            ):

                deduped[chunk_id] = (
                    chunk,
                    score,
                )

        ranked = sorted(
            deduped.values(),
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
                    retrieval_method="agentic",
                    metadata={
                        "document_id": chunk.document_id,
                        **chunk.metadata,
                    },
                )
            )

        return results