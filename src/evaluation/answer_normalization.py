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

    values = []

    for match in matches:
        try:
            values.append(float(match))
        except Exception:
            pass

    return values


def numeric_match(
    gold: str,
    pred: str,
    tolerance: float = 0.02,
) -> bool:

    gold_vals = extract_numeric_values(gold)

    pred_vals = extract_numeric_values(pred)

    if not gold_vals or not pred_vals:
        return False

    for g in gold_vals:

        for p in pred_vals:

            if g == 0:
                if abs(p) < tolerance:
                    return True

            else:
                rel_error = abs(g - p) / abs(g)

                if rel_error <= tolerance:
                    return True

    return False