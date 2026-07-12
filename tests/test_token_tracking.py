from finance_bench.llm.token_tracking import TokenTracker


def test_token_tracking():
    tracker = TokenTracker()

    tracker.add_prompt_tokens(100)
    tracker.add_completion_tokens(50)
    tracker.add_retrieval_tokens(25)

    summary = tracker.summary()

    assert summary.total_tokens == 175