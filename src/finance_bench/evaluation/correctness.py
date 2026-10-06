from __future__ import annotations

import json
import re

from finance_bench.evaluation.answer_normalization import (
    normalize_text,
    numeric_match,
)
from finance_bench.llm.claude_client import ZERO_USAGE, ClaudeClient

_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL
)


def _parse_judge_response(text: str) -> dict:
    """
    Parse the judge's JSON verdict, tolerating a markdown code fence
    around it (the model sometimes wraps its JSON in ```json ... ```
    even when asked for raw JSON) so a well-formed verdict isn't
    misread as a parse failure.
    """

    fenced = _CODE_FENCE_RE.match(text.strip())

    candidate = fenced.group(1) if fenced else text

    try:
        return json.loads(candidate)
    except ValueError:
        return {
            "correct": False,
            "reason": "invalid_json",
            "raw_response": text,
        }


def _contains_whole(text: str, needle: str) -> bool:
    pattern = rf"(?<!\w)(?<!\d\.){re.escape(needle)}(?!\w|\.\d)"

    return re.search(pattern, text) is not None


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

        # an empty string is a substring of everything
        if not gold_norm or not pred_norm:
            return False

        # exact normalized match
        if gold_norm == pred_norm:
            return True

        # containment on whole tokens, so 100 is not found inside 100000
        if _contains_whole(pred_norm, gold_norm):
            return True

        if _contains_whole(gold_norm, pred_norm):
            return True

        # numeric equivalence
        return numeric_match(
            gold_answer,
            generated_answer,
        )

    def grade(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> tuple[dict, dict]:
        """
        Returns (grading, usage). A deterministic match never calls
        the LLM, so its usage is zero.
        """

        deterministic_correct = (
            self._deterministic_check(
                gold_answer,
                generated_answer,
            )
        )

        if deterministic_correct:

            return (
                {
                    "correct": True,
                    "method": "deterministic",
                    "reason": "normalized_match",
                },
                dict(ZERO_USAGE),
            )

        prompt = self.judge_prompt_template.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        result = self.client.generate(prompt)

        parsed = _parse_judge_response(result["text"])

        parsed["method"] = "llm_judge"

        return parsed, result["usage"]

    def grade_batch(
        self,
        items: list[dict],
    ) -> list[tuple[dict, dict]]:
        """
        items: [{"question", "gold_answer", "generated_answer"}, ...]

        Batched replacement for calling `grade` once per item.
        Deterministic matches never touch the API. Only ambiguous
        items go to the LLM judge, and those are submitted as ONE
        Anthropic Message Batch (50% cheaper than N synchronous
        calls). Returns (grading, usage) tuples in the same order
        as `items`.
        """

        results: list[tuple[dict, dict] | None] = [None] * len(items)

        needs_judge: list[tuple[int, str]] = []

        for i, item in enumerate(items):

            if self._deterministic_check(
                item["gold_answer"], item["generated_answer"]
            ):
                results[i] = (
                    {
                        "correct": True,
                        "method": "deterministic",
                        "reason": "normalized_match",
                    },
                    dict(ZERO_USAGE),
                )
            else:
                prompt = self.judge_prompt_template.format(**item)
                needs_judge.append((i, prompt))

        if needs_judge:

            prompts = [p for _, p in needs_judge]

            batch_results = self.client.generate_batch(prompts)

            for (idx, _), result in zip(needs_judge, batch_results):

                parsed = _parse_judge_response(result["text"])

                parsed["method"] = "llm_judge"

                results[idx] = (parsed, result["usage"])

        return results