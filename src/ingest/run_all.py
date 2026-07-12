from __future__ import annotations

import json
from pathlib import Path

from src.ingest.build_corpus import (
    build_corpus,
)

from src.ingest.financebench_dataset import (
    FinanceBenchDataset,
)

from src.retrieval.qdrant_client import (
    QdrantManager,
)

from src.retrieval.embeddings import EmbeddingModel


OUTPUT_DIR = Path("data/processed")

CHUNKS_PATH = OUTPUT_DIR / "chunks.json"

EXAMPLES_PATH = (
    OUTPUT_DIR
    / "financebench_examples.json"
)


def save_examples(
    examples: list[dict],
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EXAMPLES_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            examples,
            f,
            indent=2,
        )

    print(
        f"Saved examples to "
        f"{EXAMPLES_PATH}"
    )


def save_chunks(chunks):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = [
        chunk.model_dump()
        for chunk in chunks
    ]

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serialized,
            f,
            indent=2,
        )

    print(
        f"Saved chunks to "
        f"{CHUNKS_PATH}"
    )


def ingest_qdrant(chunks):

    qdrant = QdrantManager()

    embedder = EmbeddingModel()

    print(
        "\nEnsuring collection exists..."
    )

    qdrant.ensure_collection()

    print(
        "\nUploading chunks..."
    )

    texts = [
        chunk.text for chunk in chunks
    ]

    embeddings = (
        embedder.embed(texts).tolist()
    )

    qdrant.upload_documents(
        chunks=chunks,
        embeddings=embeddings
    )

    print(
        "\nUpload complete."
    )


def main():

    print(
        "\n=== FinanceBench Ingestion ===\n"
    )

    dataset = FinanceBenchDataset()

    examples = dataset.load()

    print(
        f"Loaded {len(examples)} examples"
    )

    save_examples(examples)

    chunks = build_corpus(examples)

    print(
        f"Built {len(chunks)} chunks"
    )

    save_chunks(chunks)

    ingest_qdrant(chunks)

    print(
        "\n=== Ingestion Complete ===\n"
    )


if __name__ == "__main__":
    main()