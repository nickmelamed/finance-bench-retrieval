from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

# data structures 


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: Optional[Dict] = None


@dataclass
class AlignmentResult:
    question_id: str
    gold_evidence: List[str]
    matched_chunk_ids: List[str]
    recall: float
    precision: float
    matched_texts: List[str]


# gold alignment 


class GoldEvidenceAligner:
    """
    Align gold evidence passages to chunked documents.

    Used for:
    - retrieval recall evaluation
    - evidence grounding
    - reranker benchmarking
    - chunk attribution
    """

    def __init__(
        self,
        fuzzy_threshold: int = 80,
    ):
        self.fuzzy_threshold = fuzzy_threshold

    # main 

    def align(
        self,
        question_id: str,
        gold_evidence: List[str],
        chunks: List[Chunk],
    ) -> AlignmentResult:

        matched_chunks = []

        matched_texts = []

        for evidence in gold_evidence:

            chunk = self._find_best_chunk_match(
                evidence,
                chunks,
            )

            if chunk is not None:
                matched_chunks.append(chunk.chunk_id)
                matched_texts.append(chunk.text)

        matched_chunks = list(set(matched_chunks))

        recall = (
            len(matched_chunks) / max(len(gold_evidence), 1)
        )

        precision = (
            len(matched_chunks) / max(len(chunks), 1)
        )

        return AlignmentResult(
            question_id=question_id,
            gold_evidence=gold_evidence,
            matched_chunk_ids=matched_chunks,
            recall=recall,
            precision=precision,
            matched_texts=matched_texts,
        )

    # internal 

    def _find_best_chunk_match(
        self,
        evidence: str,
        chunks: List[Chunk],
        question_id: str | None = None,
    ) -> Optional[Chunk]:

        if not evidence:
            return None

        evidence = " ".join(
            evidence.lower().split()
        )

        # use smaller evidence window
        evidence = evidence[:2000]

        best_score = 0
        best_chunk = None

        for chunk in chunks:

            chunk_text = " ".join(
                chunk.text.lower().split()
            )

            # token overlap
            overlap_score = fuzz.token_set_ratio(
                evidence,
                chunk_text,
            )

            # substring overlap
            partial_score = fuzz.partial_ratio(
                evidence,
                chunk_text,
            )

            score = max(
                overlap_score,
                partial_score,
            )

            if score > best_score:

                best_score = score

                best_chunk = chunk

        if best_score >= self.fuzzy_threshold:
            return best_chunk

        return None


# retrieval recall 


def retrieval_recall_at_k(
    retrieved_chunk_ids: List[str],
    gold_chunk_ids: List[str],
    k: int,
) -> float:
    """
    Standard Recall@K metric.
    """

    retrieved_top_k = set(
        retrieved_chunk_ids[:k]
    )

    gold = set(gold_chunk_ids)

    if len(gold) == 0:
        return 0.0

    hits = len(
        retrieved_top_k.intersection(gold)
    )

    return hits / len(gold)


def retrieval_hit_rate_at_k(
    retrieved_chunk_ids: List[str],
    gold_chunk_ids: List[str],
    k: int,
) -> float:
    """
    HitRate@K:
    Did we retrieve ANY gold chunk?
    """

    retrieved_top_k = set(
        retrieved_chunk_ids[:k]
    )

    gold = set(gold_chunk_ids)

    return float(
        len(
            retrieved_top_k.intersection(gold)
        ) > 0
    )


def mean_recall_at_k(
    results: List[Tuple[List[str], List[str]]],
    k: int,
) -> float:
    """
    Average Recall@K over dataset.

    results:
        [
            (retrieved_ids, gold_ids),
        ]
    """

    if len(results) == 0:
        return 0.0

    scores = [
        retrieval_recall_at_k(
            retrieved,
            gold,
            k,
        )
        for retrieved, gold in results
    ]

    return sum(scores) / len(scores)


def mean_hit_rate_at_k(
    results: List[Tuple[List[str], List[str]]],
    k: int,
) -> float:

    if len(results) == 0:
        return 0.0

    scores = [
        retrieval_hit_rate_at_k(
            retrieved,
            gold,
            k,
        )
        for retrieved, gold in results
    ]

    return sum(scores) / len(scores)