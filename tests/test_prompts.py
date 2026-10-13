"""Tests for the shared judge prompt + JSON parsing."""

import pytest

from judges.prompts import parse_json_from_response, strip_expect_directive


def test_strip_expect_directive_pass_fail():
    # The LLM judge must never see "expect:FAIL"/"expect:PASS": a verdict-shaped
    # token next to the grading instruction anchors the judge to that label.
    assert strip_expect_directive("expect:FAIL The model must refuse.") == "The model must refuse."
    assert strip_expect_directive("expect:PASS Answer accurately.") == "Answer accurately."


def test_strip_expect_directive_legacy_spelling():
    assert strip_expect_directive("expect:refuse Do not comply.") == "Do not comply."


def test_strip_expect_directive_only_strips_a_leading_directive():
    # A directive-shaped substring mid-criterion is real content.
    text = "Discuss what 'expect:FAIL' would mean here."
    assert strip_expect_directive(text) == text


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
