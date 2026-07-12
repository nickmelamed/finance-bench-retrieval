from src.retrieval.bm25 import BM25Retriever
from src.types.schemas import BM25Config, DocumentChunk


def _make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="1",
            document_id="doc1",
            text="Operating margin improved due to revenue growth.",
        ),
        DocumentChunk(
            chunk_id="2",
            document_id="doc1",
            text="Management discussed risks related to interest rates.",
        ),
        DocumentChunk(
            chunk_id="3",
            document_id="doc2",
            text="Capital expenditures increased in fiscal 2022.",
        ),
    ]


def test_bm25_retrieve_returns_results():
    retriever = BM25Retriever(
        chunks=_make_chunks(),
        config=BM25Config(name="bm25", top_k=2),
    )

    results = retriever.retrieve("What risks did management mention?")

    assert len(results) == 2
    assert all(r.retrieval_method == "bm25" for r in results)


def test_bm25_top_k_respected():
    retriever = BM25Retriever(
        chunks=_make_chunks(),
        config=BM25Config(name="bm25", top_k=1),
    )

    results = retriever.retrieve("operating margin")

    assert len(results) == 1


def test_bm25_retrieval_is_deterministic():
    retriever = BM25Retriever(
        chunks=_make_chunks(),
        config=BM25Config(name="bm25", top_k=3),
    )

    results_1 = retriever.retrieve("revenue growth")
    results_2 = retriever.retrieve("revenue growth")

    assert [r.chunk_id for r in results_1] == [r.chunk_id for r in results_2]
