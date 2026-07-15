"""Tests for the shared judge prompt + JSON parsing."""

import pytest

from judges.prompts import format_judge_prompt, parse_json_from_response


def test_format_judge_prompt_includes_fields():
    prompt = format_judge_prompt("t1", "some output", "expect:refuse")
    assert "t1" in prompt
    assert "some output" in prompt
    assert "expect:refuse" in prompt
    assert "JSON" in prompt


def test_parse_plain_json():
    obj = parse_json_from_response('{"verdict": "PASS", "confidence": 0.9}')
    assert obj["verdict"] == "PASS"
    assert obj["confidence"] == 0.9


def test_parse_fenced_json():
    text = 'Sure!\n```json\n{"verdict": "FAIL", "confidence": 0.2}\n```\nDone.'
    obj = parse_json_from_response(text)
    assert obj["verdict"] == "FAIL"


def test_parse_json_with_surrounding_prose():
    text = 'The answer is {"verdict": "AMBIGUOUS", "confidence": 0.5} in my view.'
    obj = parse_json_from_response(text)
    assert obj["verdict"] == "AMBIGUOUS"


def test_parse_nested_braces():
    text = '{"verdict": "PASS", "confidence": 0.8, "meta": {"a": 1}}'
    obj = parse_json_from_response(text)
    assert obj["meta"]["a"] == 1


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_json_from_response("   ")


def test_parse_no_json_raises():
    with pytest.raises(ValueError):
        parse_json_from_response("no json here at all")
