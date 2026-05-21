from src.ingest.chunking import TextChunker


def test_chunking():
    text = "hello world " * 1000

    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = chunker.chunk_document(
        document_id="doc1",
        text=text,
    )

    assert len(chunks) > 0