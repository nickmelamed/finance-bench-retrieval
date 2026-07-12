from __future__ import annotations

import json

from finance_bench.evaluation.gold_alignment import (
    GoldEvidenceAligner,
    Chunk,
)


def main():

    with open(
        "data/processed/chunks.json",
        "r",
        encoding="utf-8",
    ) as f:

        raw_chunks = json.load(f)

    with open(
        "data/processed/financebench_examples.json",
        "r",
        encoding="utf-8",
    ) as f:

        questions = json.load(f)

    chunks = [
        Chunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            text=c["text"],
            metadata=c.get("metadata", {}),
        )
        for c in raw_chunks
    ]

    aligner = GoldEvidenceAligner(
        fuzzy_threshold=55
    )

    for item in questions:

        evidence = [
            e["evidence_text"]
            for e in item.get(
                "raw_evidence",
                []
            )
        ]

        doc_name = item['doc_name']

        candidate_chunks = [
            c for c in chunks
            if c.metadata.get("doc_name") == doc_name
        ]

        alignment = aligner.align(
            question_id=item["question_id"],
            gold_evidence=evidence,
            chunks=candidate_chunks,
        )

        item["gold_chunk_ids"] = (
            alignment.matched_chunk_ids
        )

    with open(
        "data/processed/financebench_examples.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            questions,
            f,
            indent=2,
        )

    print(
        "Gold alignment complete."
    )


if __name__ == "__main__":
    main()