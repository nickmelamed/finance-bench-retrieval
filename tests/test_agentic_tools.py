from collections import Counter

import pytest

from src.retrieval.agentic import AgenticRetriever
from src.types.schemas import AgenticConfig, BM25Config, DocumentChunk, RetrievalResult


def _make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="c0",
            document_id="doc1",
            text="Revenue was $10 million in FY2022.",
        ),
        DocumentChunk(
            chunk_id="c1",
            document_id="doc1",
            text="Operating expenses rose due to hiring.",
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="doc1",
            text="Net income increased year over year.",
        ),
        DocumentChunk(
            chunk_id="c3",
            document_id="doc2",
            text="Capital expenditures were flat.",
        ),
    ]


@pytest.fixture
def retriever(monkeypatch) -> AgenticRetriever:
    # AgenticRetriever constructs a ClaudeClient eagerly; these tests
    # only exercise its deterministic tool logic, never the Anthropic
    # API itself, so a dummy key is enough.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = AgenticConfig(
        name="agentic",
        top_k=2,
        use_dense_tool=False,
        use_rerank_tool=False,
    )

    return AgenticRetriever(
        chunks=_make_chunks(),
        config=config,
        bm25_config=BM25Config(name="bm25"),
        dense_config=None,
    )


def test_lexical_search_tool_records_seen(retriever):
    output = retriever._execute_tool("lexical_search", {"query": "revenue"})

    assert output
    assert all("chunk_id" in item for item in output)
    assert retriever._seen


def test_get_neighbors_returns_adjacent_chunks(retriever):
    neighbors = retriever._get_neighbors("c1", window=1)

    assert {n.chunk_id for n in neighbors} == {"c0", "c2"}


def test_get_neighbors_unknown_chunk_returns_empty(retriever):
    assert retriever._get_neighbors("does-not-exist", window=1) == []


def test_validate_chunk_ids_drops_hallucinations(retriever):
    retriever._seen = {
        "c0": RetrievalResult(
            chunk_id="c0", text="x", score=1.0, retrieval_method="agentic"
        ),
    }
    retriever._seen_frequency = Counter({"c0": 1})

    valid = retriever._validate_chunk_ids(["c0", "made-up-id"])

    assert valid == ["c0"]


def test_validate_chunk_ids_falls_back_when_all_hallucinated(retriever):
    retriever._seen = {
        "c0": RetrievalResult(
            chunk_id="c0", text="x", score=1.0, retrieval_method="agentic"
        ),
        "c1": RetrievalResult(
            chunk_id="c1", text="y", score=0.5, retrieval_method="agentic"
        ),
    }
    retriever._seen_frequency = Counter({"c0": 2, "c1": 1})

    valid = retriever._validate_chunk_ids(["totally-made-up"])

    assert valid == ["c0", "c1"]


def test_build_results_uses_full_chunk_text(retriever):
    retriever._seen = {
        "c0": RetrievalResult(
            chunk_id="c0", text="ignored", score=2.0, retrieval_method="agentic"
        ),
    }

    results = retriever._build_results(["c0"])

    assert len(results) == 1
    assert results[0].text == "Revenue was $10 million in FY2022."
    assert results[0].score == 2.0


def test_last_retrieval_usage_defaults_to_zero(retriever):
    assert retriever.last_retrieval_usage() == (0, 0)
