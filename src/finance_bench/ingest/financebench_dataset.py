from __future__ import annotations

import json
import re
from typing import Any


def normalize_whitespace(text: str) -> str:
    """
    Normalize OCR/PDF whitespace artifacts.
    """

    text = text.replace("\xa0", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def flatten_financebench_evidence(
    evidence: list[dict[str, Any]] | list[str] | str,
) -> str:
    """
    Convert FinanceBench evidence structure into
    a canonical retrieval document string.
    """

    if not evidence:
        return ""

    flattened: list[str] = []

    # evidence already plain string
    if isinstance(evidence, str):
        return normalize_whitespace(evidence)

    # evidence is list
    if isinstance(evidence, list):

        for item in evidence:

            # evidence item is dict
            if isinstance(item, dict):

                text = item.get(
                    "evidence_text",
                    "",
                )

                if text:
                    flattened.append(
                        normalize_whitespace(text)
                    )

            # evidence item is already string
            elif isinstance(item, str):

                flattened.append(
                    normalize_whitespace(item)
                )

    return "\n\n".join(flattened)


class FinanceBenchDataset:

    def __init__(
        self,
        path: str = (
            "data/financebench/"
            "financebench_open_source.jsonl"
        ),
    ):
        self.path = path

    def load(self) -> list[dict]:

        examples = []

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as f:

            for i, line in enumerate(f):

                row = json.loads(line)

                raw_evidence = row.get(
                    "evidence",
                    [],
                )

                document_text = (
                    flatten_financebench_evidence(
                        raw_evidence
                    )
                )

                evidence_pages = []

                if isinstance(raw_evidence, list):

                    for item in raw_evidence:

                        if (
                            isinstance(item, dict)
                            and item.get(
                                "evidence_page_num"
                            )
                            is not None
                        ):
                            evidence_pages.append(
                                item[
                                    "evidence_page_num"
                                ]
                            )

                examples.append(
                    {
                        "question_id": row.get(
                            "financebench_id",
                            str(i),
                        ),
                        "question": normalize_whitespace(
                            row["question"]
                        ),
                        "gold_answer": normalize_whitespace(
                            row["answer"]
                        ),
                        "document_text": document_text,
                        "doc_name": row.get(
                            "doc_name",
                            "",
                        ),
                        "company": row.get(
                            "company",
                            "",
                        ),
                        "question_type": row.get(
                            "question_type",
                            "",
                        ),
                        "evidence_pages": evidence_pages,
                        "raw_evidence": raw_evidence,
                    }
                )

        return examples