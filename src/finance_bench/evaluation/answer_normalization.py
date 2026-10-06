from __future__ import annotations

import re


def normalize_text(text: str) -> str:

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = text.replace(",", "")

    text = text.replace("$", "")

    text = text.replace("%", " percent ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_numeric_values(text: str) -> list[float]:

    text = normalize_text(text)

    matches = re.findall(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    return [float(match) for match in matches]


def numeric_match(
    gold: str,
    pred: str,
    tolerance: float = 0.02,
) -> bool:

    gold_vals = extract_numeric_values(gold)

    pred_vals = extract_numeric_values(pred)

    if not gold_vals or not pred_vals:
        return False

    # every number in the gold answer must appear in the prediction
    return all(
        any(_within_tolerance(g, p, tolerance) for p in pred_vals)
        for g in gold_vals
    )


def _within_tolerance(gold: float, pred: float, tolerance: float) -> bool:
    if gold == 0:
        return abs(pred) < tolerance

    return abs(gold - pred) / abs(gold) <= tolerance