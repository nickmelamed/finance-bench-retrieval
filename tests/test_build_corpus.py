from finance_bench.ingest.build_corpus import build_corpus

EVAL_ONLY = {"question", "gold_answer", "evidence_pages", "raw_evidence", "gold_chunk_ids"}


def test_chunk_metadata_carries_no_eval_only_fields():
    examples = [
        {
            "question_id": "q1",
            "question": "What was revenue?",
            "gold_answer": "$5 million",
            "doc_name": "ACME_2022_10K",
            "company": "ACME",
            "question_type": "metrics-generated",
            "evidence_pages": [3],
            "raw_evidence": ["Revenue was $5 million."],
            "gold_chunk_ids": ["x"],
            "document_text": "Revenue was $5 million in fiscal 2022.",
        }
    ]
    chunks = build_corpus(examples)
    assert chunks
    for chunk in chunks:
        assert not EVAL_ONLY & set(chunk.metadata)
        assert chunk.metadata["company"] == "ACME"
        assert chunk.document_id == "q1"
