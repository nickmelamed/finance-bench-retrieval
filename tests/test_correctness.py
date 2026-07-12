from finance_bench.evaluation.correctness import _parse_judge_response


def test_parse_judge_response_plain_json():
    parsed = _parse_judge_response('{"correct": true, "reason": "matches"}')

    assert parsed == {"correct": True, "reason": "matches"}


def test_parse_judge_response_strips_markdown_fence():
    text = '```json\n{"correct": true, "reason": "matches"}\n```'

    parsed = _parse_judge_response(text)

    assert parsed == {"correct": True, "reason": "matches"}


def test_parse_judge_response_strips_bare_fence():
    text = '```\n{"correct": false, "reason": "no match"}\n```'

    parsed = _parse_judge_response(text)

    assert parsed == {"correct": False, "reason": "no match"}


def test_parse_judge_response_invalid_json_falls_back():
    parsed = _parse_judge_response("not json at all")

    assert parsed["correct"] is False
    assert parsed["reason"] == "invalid_json"
    assert parsed["raw_response"] == "not json at all"
