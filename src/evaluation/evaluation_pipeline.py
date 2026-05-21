from __future__ import annotations

from src.llm.claude_client import ClaudeClient

from src.llm.token_tracking import TokenTracker

from src.evaluation.correctness import (
    CorrectnessGrader,
)

from src.evaluation.failure_analysis import (
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

        response = self.client.generate(prompt)

        usage = response.usage

        self.token_tracker.add_prompt_tokens(
            usage.input_tokens
        )

        self.token_tracker.add_completion_tokens(
            usage.output_tokens
        )

        self.token_tracker.add_retrieval_tokens(
            len(context.split())
        )

        return response.content[0].text

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

        grading = self.grader.grade(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
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