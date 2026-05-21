from __future__ import annotations

import json

from src.llm.claude_client import ClaudeClient

from src.evaluation.answer_normalization import (
    normalize_text,
    numeric_match,
)


class CorrectnessGrader:

    def __init__(
        self,
        model: str,
        judge_prompt_template: str,
    ):
        self.client = ClaudeClient(model)

        self.judge_prompt_template = (
            judge_prompt_template
        )

    def _deterministic_check(
        self,
        gold_answer: str,
        generated_answer: str,
    ) -> bool:

        gold_norm = normalize_text(
            gold_answer
        )

        pred_norm = normalize_text(
            generated_answer
        )

        # exact normalized match
        if gold_norm == pred_norm:
            return True

        # containment
        if gold_norm in pred_norm:
            return True

        if pred_norm in gold_norm:
            return True

        # numeric equivalence
        if numeric_match(
            gold_answer,
            generated_answer,
        ):
            return True

        return False

    def grade(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ):

        deterministic_correct = (
            self._deterministic_check(
                gold_answer,
                generated_answer,
            )
        )

        if deterministic_correct:

            return {
                "correct": True,
                "method": "deterministic",
                "reason": "normalized_match",
            }

        prompt = self.judge_prompt_template.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        response = self.client.generate(prompt)

        text = response.content[0].text

        try:

            parsed = json.loads(text)

        except Exception:

            parsed = {
                "correct": False,
                "reason": "invalid_json",
                "raw_response": text,
            }

        parsed["method"] = "llm_judge"

        return parsed