from __future__ import annotations

from typing import List

from src.config.loaders import (
    load_yaml_config,
)

from src.ingest.chunking import (
    TextChunker,
)

from src.types.schemas import (
    DocumentChunk,
    ExperimentConfig,
)


def build_corpus(
    examples: list[dict],
) -> List[DocumentChunk]:

    config = load_yaml_config(
        "experiment.yaml",
        ExperimentConfig,
    )

    chunker = TextChunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )

    all_chunks: List[DocumentChunk] = []

    for example in examples:

        document_text = example.get(
            "document_text",
            "",
        )

        if not document_text:
            continue

        metadata = {
            "question_id": example.get(
                "question_id"
            ),
            "question": example.get(
                "question",
                "",
            ),
            "gold_answer": example.get(
                "gold_answer",
                "",
            ),
            "doc_name": example.get(
                "doc_name",
                "",
            ),
            "company": example.get(
                "company",
                "",
            ),
            "question_type": example.get(
                "question_type",
                "",
            ),
            "evidence_pages": example.get(
                "evidence_pages",
                [],
            ),
        }

        chunks = chunker.chunk_document(
            document_id=str(
                example["question_id"]
            ),
            text=document_text,
            metadata=metadata,
        )

        all_chunks.extend(chunks)

    return all_chunks