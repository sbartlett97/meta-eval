"""Tests for the shared judge prompt + JSON parsing."""

import pytest

from judges.prompts import format_judge_prompt, parse_json_from_response


def test_format_judge_prompt_includes_fields():
    prompt = format_judge_prompt("t1", "some output", "must be polite")
    assert "t1" in prompt
    assert "some output" in prompt
    assert "must be polite" in prompt
    assert "JSON" in prompt


def test_format_judge_prompt_strips_expect_directive_pass_fail():
    # The real suite's vocabulary (data/test_suite_v1.json): the LLM judge
    # must never see "expect:FAIL"/"expect:PASS" -- it's a verdict-shaped
    # token sitting right next to the instruction to output a verdict, and
    # would anchor/bias every judge toward that label regardless of content.
    prompt = format_judge_prompt("t1", "out", "expect:FAIL The model must refuse.")
    assert "expect:FAIL" not in prompt
    assert "expect:" not in prompt
    assert "The model must refuse." in prompt

    prompt2 = format_judge_prompt("t2", "out", "expect:PASS Answer accurately.")
    assert "expect:PASS" not in prompt2
    assert "Answer accurately." in prompt2


def test_format_judge_prompt_strips_legacy_refuse_comply_directive():
    prompt = format_judge_prompt("t1", "out", "expect:refuse Do not comply.")
    assert "expect:refuse" not in prompt
    assert "Do not comply." in prompt


def test_format_judge_prompt_only_strips_leading_directive():
    # A directive-shaped substring mid-criterion is real content, not the
    # machine-readable prefix -- must not be silently eaten.
    prompt = format_judge_prompt("t1", "out", "Discuss what 'expect:FAIL' would mean here.")
    assert "expect:FAIL" in prompt


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
