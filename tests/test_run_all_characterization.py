from types import SimpleNamespace

import pytest

import finance_bench.llm.claude_client as client_module
from finance_bench.experiments.run_all import evaluate_retrieval


class FakeRetriever:
    top_k = 2

    def retrieve(self, query):
        return [SimpleNamespace(chunk_id=c) for c in ["x", "y", "g"]]


def test_retrieval_metrics_use_the_retriever_top_k_and_report_coverage(capsys):
    questions = [
        {"question": "a", "gold_chunk_ids": ["g"]},
        {"question": "b", "gold_chunk_ids": ["x"]},
        {"question": "c", "gold_chunk_ids": []},
    ]
    result = evaluate_retrieval("fake", FakeRetriever(), questions)
    assert result["retrieval_questions"] == 2
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.5
    assert "2 of 3 questions" in capsys.readouterr().out


class FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        usage = SimpleNamespace(input_tokens=40, output_tokens=7)
        content = [SimpleNamespace(text="hello")]
        return SimpleNamespace(content=content, usage=usage, stop_reason="end_turn")


def make_client(monkeypatch):
    store = {}
    monkeypatch.setattr(client_module, "cache", store)
    monkeypatch.setattr(
        client_module,
        "Anthropic",
        lambda api_key=None: SimpleNamespace(messages=FakeMessages()),
    )
    return client_module.ClaudeClient("m"), store


def test_cache_hit_reports_the_original_usage(monkeypatch):
    client, _ = make_client(monkeypatch)
    first = client.generate("prompt")
    second = client.generate("prompt")
    assert client.client.messages.calls == 1
    assert second["text"] == first["text"]
    assert second["usage"] == first["usage"] == {
        "input_tokens": 40,
        "output_tokens": 7,
    }


def test_cache_entry_without_usage_reports_zero(monkeypatch):
    client, store = make_client(monkeypatch)
    store[client._cache_key("old", 512)] = {"text": "cached"}
    assert client.generate("old") == {
        "text": "cached",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_batch_cache_hits_report_stored_usage(monkeypatch):
    client, store = make_client(monkeypatch)
    store[client._cache_key("p", 512)] = {
        "text": "t",
        "usage": {"input_tokens": 9, "output_tokens": 2},
    }
    assert client.generate_batch(["p"]) == [
        {"text": "t", "usage": {"input_tokens": 9, "output_tokens": 2}}
    ]


@pytest.mark.parametrize("value", [True, False])
def test_accuracy_accepts_int_and_bool_lists(value):
    from finance_bench.evaluation.qa_metrics import accuracy

    assert accuracy([int(value)]) == float(value)


def test_calls_only_use_arguments_the_installed_sdk_accepts(monkeypatch):
    import inspect

    from anthropic.resources.messages import Messages

    seen = []

    class Recorder(FakeMessages):
        def create(self, **kwargs):
            seen.append(kwargs)
            return super().create(**kwargs)

    monkeypatch.setattr(client_module, "cache", {})
    monkeypatch.setattr(
        client_module,
        "Anthropic",
        lambda api_key=None: SimpleNamespace(messages=Recorder()),
    )
    client = client_module.ClaudeClient("m")
    client.generate("prompt")
    client.generate_with_tools(
        messages=[{"role": "user", "content": "q"}], tools=[], system="s"
    )
    assert len(seen) == 2
    for kwargs in seen:
        inspect.signature(Messages.create).bind(None, **kwargs)
        assert kwargs["extra_body"] == {"temperature": 0.0}
