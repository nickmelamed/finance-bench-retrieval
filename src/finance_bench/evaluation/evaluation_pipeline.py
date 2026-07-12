from __future__ import annotations

from finance_bench.llm.claude_client import ClaudeClient

from finance_bench.llm.token_tracking import TokenTracker

from finance_bench.evaluation.correctness import (
    CorrectnessGrader,
)

from finance_bench.evaluation.failure_analysis import (
    FailureAnalyzer,
)


class EvaluationPipeline:

    def __init__(
        self,
        model: str,
        judge_prompt_template: str,
        qa_prompt_template: str,
    ):

        self.client = ClaudeClient(model)

        self.qa_prompt_template = (
            qa_prompt_template
        )

        self.grader = CorrectnessGrader(
            model,
            judge_prompt_template,
        )

        self.failure_analyzer = (
            FailureAnalyzer()
        )

        self.token_tracker = (
            TokenTracker()
        )

    def build_context(
        self,
        retrieved_chunks,
    ) -> str:

        formatted = []

        for idx, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):

            formatted.append(
                f"[Chunk {idx}]\n"
                f"{chunk.text}"
            )

        return "\n\n".join(formatted)

    def answer_question(
        self,
        question,
        context,
    ):

        prompt = self.qa_prompt_template.format(
            context=context,
            question=question,
        )

        result = self.client.generate(prompt)

        self.token_tracker.add_prompt_tokens(
            result["usage"]["input_tokens"]
        )

        self.token_tracker.add_completion_tokens(
            result["usage"]["output_tokens"]
        )

        self.token_tracker.add_retrieval_tokens(
            len(context.split())
        )

        return result["text"]

    def answer_batch(
        self,
        questions_and_contexts: list[tuple[str, str]],
    ) -> list[str]:
        """
        Batched replacement for calling `answer_question` once per
        question - submits every QA-generation prompt as ONE
        Anthropic Message Batch instead of N synchronous calls
        (50% cheaper). Token tracking is preserved per item.
        """

        prompts = [
            self.qa_prompt_template.format(
                context=context,
                question=question,
            )
            for question, context in questions_and_contexts
        ]

        results = self.client.generate_batch(prompts)

        for result, (_, context) in zip(
            results, questions_and_contexts
        ):

            self.token_tracker.add_prompt_tokens(
                result["usage"]["input_tokens"]
            )

            self.token_tracker.add_completion_tokens(
                result["usage"]["output_tokens"]
            )

            self.token_tracker.add_retrieval_tokens(
                len(context.split())
            )

        return [r["text"] for r in results]

    def evaluate(
        self,
        question,
        gold_answer,
        retrieved_chunks,
    ):

        context = self.build_context(
            retrieved_chunks
        )

        generated_answer = (
            self.answer_question(
                question=question,
                context=context,
            )
        )

        grading, judge_usage = self.grader.grade(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        self.token_tracker.add_prompt_tokens(
            judge_usage["input_tokens"]
        )

        self.token_tracker.add_completion_tokens(
            judge_usage["output_tokens"]
        )

        if not grading["correct"]:

            self.failure_analyzer.classify(
                question=question,
                generated_answer=generated_answer,
                gold_answer=gold_answer,
                retrieved_chunks=retrieved_chunks,
            )

        return {
            "generated_answer": generated_answer,
            "grading": grading,
            "context": context,
        }