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
    ):
        failure_type = "unknown"

        if len(retrieved_chunks) == 0:
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