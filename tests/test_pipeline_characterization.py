import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import finance_bench.evaluation.correctness as correctness_module
import finance_bench.evaluation.evaluation_pipeline as pipeline_module
from finance_bench.evaluation.bootstrap import BootstrapCI
from finance_bench.llm.token_tracking import TokenTracker
from finance_bench.retrieval.fusion import ReciprocalRankFusion
from finance_bench.types.schemas import BootstrapConfig, RetrievalResult


class FakeClient:
    def __init__(self, model):
        self.model = model

    def generate_batch(self, prompts, max_tokens=512):
        return [
            {"text": "answer", "usage": {"input_tokens": 100, "output_tokens": 10}}
            for _ in prompts
        ]


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "ClaudeClient", FakeClient)
    monkeypatch.setattr(correctness_module, "ClaudeClient", FakeClient)
    return pipeline_module.EvaluationPipeline("m", "{question}", "{context}{question}")


def result(chunk_id, score=0.0):
    return RetrievalResult(
        chunk_id=chunk_id, text=chunk_id, score=score, retrieval_method="x"
    )


def test_build_context_numbers_chunks_from_one(pipeline):
    context = pipeline.build_context([result("alpha"), result("beta")])
    assert context == "[Chunk 1]\nalpha\n\n[Chunk 2]\nbeta"


def test_context_is_not_counted_twice(pipeline):
    pipeline.answer_batch([("q", "one two three")])
    assert pipeline.token_tracker.retrieval_tokens == 0
    assert pipeline.token_tracker.total_tokens == 100 + 10


def test_token_tracker_total_is_sum_of_parts():
    tracker = TokenTracker()
    tracker.add_prompt_tokens(5)
    tracker.add_completion_tokens(3)
    tracker.add_retrieval_tokens(2)
    summary = tracker.summary()
    assert summary.total_tokens == 10
    assert summary.reranking_tokens == 0


@given(st.lists(st.integers(min_value=0, max_value=10**6), max_size=20))
def test_token_tracker_total_matches_sum(counts):
    tracker = TokenTracker()
    for n in counts:
        tracker.add_prompt_tokens(n)
    assert tracker.total_tokens == sum(counts)


def test_rrf_scores_follow_reciprocal_rank_formula():
    fusion = ReciprocalRankFusion()
    fused = fusion.fuse([[result("a"), result("b")], [result("b"), result("c")]])
    k = fusion.k
    scores = {r.chunk_id: r.score for r in fused}
    assert scores["a"] == pytest.approx(1 / (k + 1))
    assert scores["b"] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert scores["c"] == pytest.approx(1 / (k + 2))
    assert [r.chunk_id for r in fused] == ["b", "a", "c"]
    assert {r.retrieval_method for r in fused} == {"hybrid"}


@given(
    st.lists(st.sampled_from(list("abcdef")), unique=True, max_size=6),
    st.lists(st.sampled_from(list("abcdef")), unique=True, max_size=6),
)
def test_rrf_returns_each_chunk_once_in_score_order(first, second):
    fused = ReciprocalRankFusion().fuse(
        [[result(c) for c in first], [result(c) for c in second]]
    )
    ids = [r.chunk_id for r in fused]
    assert len(ids) == len(set(ids)) == len(set(first) | set(second))
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_bootstrap_mean_is_exact_and_constant_input_has_zero_width():
    config = BootstrapConfig(n_bootstrap=50)
    ci = BootstrapCI(config).compute([1, 1, 1, 1])
    assert ci == {"mean": 1.0, "lower": 1.0, "upper": 1.0}


@given(st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30))
def test_bootstrap_interval_is_ordered_and_within_unit_range(values):
    ci = BootstrapCI(BootstrapConfig(n_bootstrap=50)).compute(values)
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0


def test_bootstrap_ignores_global_random_state():
    config = BootstrapConfig(n_bootstrap=200)
    values = [1, 0, 1, 1, 0, 0, 1, 0]
    np.random.seed(1)
    first = BootstrapCI(config).compute(values)
    np.random.seed(2)
    second = BootstrapCI(config).compute(values)
    assert first == second


def test_bootstrap_seed_changes_the_resamples():
    values = [1, 0, 1, 1, 0, 0, 1, 0]
    a = BootstrapCI(BootstrapConfig(n_bootstrap=200, seed=1)).compute(values)
    b = BootstrapCI(BootstrapConfig(n_bootstrap=200, seed=2)).compute(values)
    assert a["mean"] == b["mean"]
    assert (a["lower"], a["upper"]) != (b["lower"], b["upper"])
