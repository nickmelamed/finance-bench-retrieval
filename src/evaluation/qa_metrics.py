import numpy as np
from typing import List


def accuracy(correctness: List[bool]) -> float:
    """
    Mean accuracy over examples.
    """

    if len(correctness) == 0:
        return 0.0

    return float(np.mean(correctness))


def token_efficiency(
    total_tokens: int,
    correct_answers: int,
) -> float:
    """
    Tokens consumed per correct answer.

    Lower is better.
    """

    if total_tokens < 0:
        raise ValueError(
            "total_tokens must be non-negative"
        )

    if correct_answers < 0:
        raise ValueError(
            "correct_answers must be non-negative"
        )

    if correct_answers == 0:
        return float("inf")

    return total_tokens / correct_answers


def average_latency(
    latencies: List[float],
) -> float:
    """
    Mean latency in seconds.
    """

    if len(latencies) == 0:
        return 0.0

    return float(np.mean(latencies))


def tokens_per_question(
    token_counts: List[int],
) -> float:
    """
    Average tokens used per question.
    """

    if len(token_counts) == 0:
        return 0.0

    return float(np.mean(token_counts))


def tokens_per_correct_answer(
    token_counts: List[int],
    correctness: List[bool],
) -> float:
    """
    More robust token efficiency metric.

    Computes:
        total tokens over ONLY correct examples
    """

    if len(token_counts) != len(correctness):
        raise ValueError(
            "token_counts and correctness "
            "must have same length"
        )

    correct_token_total = sum(
        tokens
        for tokens, correct in zip(
            token_counts,
            correctness,
        )
        if correct
    )

    num_correct = sum(correctness)

    if num_correct == 0:
        return float("inf")

    return correct_token_total / num_correct