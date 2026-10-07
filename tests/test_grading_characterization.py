from hypothesis import given
from hypothesis import strategies as st

from finance_bench.evaluation.answer_normalization import (
    extract_numeric_values,
    normalize_text,
    numeric_match,
)
from finance_bench.evaluation.correctness import CorrectnessGrader


def deterministic(gold: str, pred: str) -> bool:
    grader = CorrectnessGrader.__new__(CorrectnessGrader)
    return grader._deterministic_check(gold, pred)


def test_normalize_text_strips_currency_commas_and_percent():
    assert normalize_text("$1,234.5 Million") == "1234.5 million"
    assert normalize_text("12%") == "12 percent"
    assert normalize_text(None) == ""


@given(st.text())
def test_normalize_text_is_idempotent_and_trimmed(text):
    once = normalize_text(text)
    assert normalize_text(once) == once
    assert once == once.strip()


def test_extract_numeric_values_reads_signed_decimals():
    assert extract_numeric_values("Down -3.5 from 1,200") == [-3.5, 1200.0]
    assert extract_numeric_values("no digits") == []


@given(st.floats(min_value=1, max_value=1e9, allow_nan=False))
def test_numeric_match_accepts_identical_values(value):
    text = f"{value:.2f}"
    assert numeric_match(text, text)


def test_numeric_match_uses_two_percent_relative_tolerance():
    assert numeric_match("100", "101.9")
    assert not numeric_match("100", "103")


def test_numeric_match_zero_gold_uses_absolute_tolerance():
    assert numeric_match("0", "0.01")
    assert not numeric_match("0", "0.5")


def test_numeric_match_needs_numbers_on_both_sides():
    assert not numeric_match("none", "5")
    assert not numeric_match("5", "none")


def test_deterministic_check_exact_and_containment():
    assert deterministic("Apple", "apple")
    assert deterministic("$5,000 million", "Revenue was 5000 million dollars")
    assert not deterministic("Apple", "Microsoft")


def test_deterministic_check_numeric_equivalence():
    assert deterministic("$10.0 billion", "about 10.1 billion in revenue")


def test_empty_answer_is_not_correct():
    assert deterministic("Revenue was 5,000 million", "") is False
    assert deterministic("", "") is False
    assert deterministic("   ", "5") is False


def test_containment_requires_whole_tokens():
    assert deterministic("100", "In 2023 we hired 100,000 workers") is False
    assert deterministic("100", "We hired 100 workers") is True
    assert deterministic("3.2", "It reached 13.25 billion") is False
    assert deterministic("5", "It grew 5.5 percent") is False
    assert deterministic("5", "It grew 5 percent") is True


def test_every_gold_number_must_appear_in_the_answer():
    assert numeric_match("5.2 billion in 2022", "5.2 billion")is False
    assert numeric_match("5.2 billion in 2022", "In 2022 it was 5.2 billion")
    assert deterministic("3.2", "The year 2023 saw 3.19") is True
