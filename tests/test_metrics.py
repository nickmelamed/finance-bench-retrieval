import math

from finance_bench.evaluation.qa_metrics import accuracy, token_efficiency


def test_accuracy():
    assert accuracy([1, 1, 0, 1]) == 0.75
    assert accuracy([]) == 0.0


def test_token_efficiency():
    assert token_efficiency(total_tokens=1000, correct_answers=5) == 200
    assert math.isinf(token_efficiency(total_tokens=1000, correct_answers=0))
