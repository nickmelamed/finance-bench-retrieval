import pytest
from hypothesis import given
from hypothesis import strategies as st

from finance_bench.evaluation.failure_analysis import FailureAnalyzer
from finance_bench.evaluation.gold_alignment import (
    Chunk,
    GoldEvidenceAligner,
    mean_hit_rate_at_k,
    mean_recall_at_k,
    retrieval_recall_at_k,
)
from finance_bench.evaluation.qa_metrics import (
    accuracy,
    token_efficiency,
    tokens_per_correct_answer,
)
from finance_bench.evaluation.retrieval_metrics import (
    hit_rate,
    mean_reciprocal_rank,
    recall_at_k,
)

ids = st.lists(st.sampled_from(list("abcdefgh")), max_size=8, unique=True)


def test_recall_hit_rate_mrr_worked_example():
    retrieved = ["x", "g1", "y", "g2"]
    gold = ["g1", "g2", "g3"]
    assert recall_at_k(retrieved, gold) == pytest.approx(2 / 3)
    assert hit_rate(retrieved, gold) == 1.0
    assert mean_reciprocal_rank(retrieved, gold) == pytest.approx(1 / 2)


def test_metrics_with_no_gold_or_no_overlap():
    assert recall_at_k(["a"], []) == 0.0
    assert hit_rate(["a"], ["b"]) == 0.0
    assert mean_reciprocal_rank(["a"], ["b"]) == 0.0


def test_current_behavior_recall_ignores_list_length():
    # SPEC 7.5: recall_at_k scores the whole list, so it has no k.
    retrieved = ["x"] * 9 + ["g"]
    assert recall_at_k(retrieved, ["g"]) == 1.0
    assert retrieval_recall_at_k(retrieved, ["g"], k=5) == 0.0


@given(ids, ids)
def test_retrieval_metrics_stay_in_unit_interval(retrieved, gold):
    for value in (
        recall_at_k(retrieved, gold),
        hit_rate(retrieved, gold),
        mean_reciprocal_rank(retrieved, gold),
    ):
        assert 0.0 <= value <= 1.0


@given(ids, ids)
def test_hit_rate_is_one_exactly_when_recall_is_positive(retrieved, gold):
    assert (hit_rate(retrieved, gold) == 1.0) == (recall_at_k(retrieved, gold) > 0)


@given(ids, ids)
def test_mrr_is_reciprocal_of_first_gold_rank(retrieved, gold):
    ranks = [i for i, cid in enumerate(retrieved, start=1) if cid in set(gold)]
    expected = 1 / ranks[0] if ranks else 0.0
    assert mean_reciprocal_rank(retrieved, gold) == pytest.approx(expected)


@given(ids, ids, st.integers(min_value=1, max_value=8))
def test_recall_at_k_never_decreases_as_k_grows(retrieved, gold, k):
    assert retrieval_recall_at_k(retrieved, gold, k) <= retrieval_recall_at_k(
        retrieved, gold, k + 1
    )


def test_mean_metrics_average_over_questions():
    results = [(["a"], ["a"]), (["b"], ["a"])]
    assert mean_recall_at_k(results, k=1) == 0.5
    assert mean_hit_rate_at_k(results, k=1) == 0.5
    assert mean_recall_at_k([], k=1) == 0.0


def test_accuracy_and_token_efficiency_edges():
    assert accuracy([]) == 0.0
    assert accuracy([True, False, True, True]) == 0.75
    assert token_efficiency(1000, 4) == 250
    assert token_efficiency(1000, 0) == float("inf")
    with pytest.raises(ValueError):
        token_efficiency(-1, 1)


def test_tokens_per_correct_answer_counts_only_correct_rows():
    assert tokens_per_correct_answer([10, 20, 30], [True, False, True]) == 20
    assert tokens_per_correct_answer([10], [False]) == float("inf")
    with pytest.raises(ValueError):
        tokens_per_correct_answer([1, 2], [True])


@given(
    st.integers(min_value=0, max_value=10**9),
    st.integers(min_value=1, max_value=10**6),
)
def test_token_efficiency_times_correct_recovers_total(total, correct):
    assert token_efficiency(total, correct) * correct == pytest.approx(total)


def test_gold_alignment_matches_best_chunk_above_threshold():
    chunks = [
        Chunk("c1", "d", "Net income rose to 4.2 billion in fiscal 2022."),
        Chunk("c2", "d", "The board approved a dividend of 0.25 per share."),
    ]
    aligner = GoldEvidenceAligner(fuzzy_threshold=80)
    result = aligner.align(
        "q1", ["Net income rose to 4.2 billion in fiscal 2022."], chunks
    )
    assert result.matched_chunk_ids == ["c1"]
    assert result.recall == 1.0


def test_gold_alignment_drops_evidence_below_threshold():
    chunks = [Chunk("c1", "d", "Completely unrelated sentence about weather.")]
    result = GoldEvidenceAligner(fuzzy_threshold=95).align(
        "q1", ["Net income rose to 4.2 billion."], chunks
    )
    assert result.matched_chunk_ids == []
    assert result.recall == 0.0


def test_failure_analyzer_labels_follow_current_heuristics():
    analyzer = FailureAnalyzer()
    chunk = object()
    analyzer.classify("q", "answer", "gold", [])
    analyzer.classify("q", "  ", "gold", [chunk])
    analyzer.classify("q", "answer", "gold 5", [chunk])
    analyzer.classify("q", "answer", "gold", [chunk])
    assert analyzer.summary() == {
        "retrieval_failure": 1,
        "empty_generation": 1,
        "numeric_reasoning_failure": 1,
        "semantic_mismatch": 1,
    }
