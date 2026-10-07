from collections import Counter


class FailureAnalyzer:
    def __init__(self):
        self.failures = []

    def classify(
        self,
        question,
        generated_answer,
        gold_answer,
        retrieved_chunks,
        gold_chunk_ids=None,
    ):
        failure_type = "unknown"

        retrieved_ids = {
            chunk.chunk_id for chunk in retrieved_chunks
        }

        missed_gold = bool(gold_chunk_ids) and not retrieved_ids.intersection(
            gold_chunk_ids
        )

        if len(retrieved_chunks) == 0 or missed_gold:
            failure_type = "retrieval_failure"

        elif generated_answer.strip() == "":
            failure_type = "empty_generation"

        elif any(
            str(num) in gold_answer
            for num in range(10)
        ):
            failure_type = "numeric_reasoning_failure"

        else:
            failure_type = "semantic_mismatch"

        self.failures.append(
            {
                "question": question,
                "failure_type": failure_type,
            }
        )

    def summary(self):
        counts = Counter(
            x["failure_type"]
            for x in self.failures
        )

        return dict(counts)