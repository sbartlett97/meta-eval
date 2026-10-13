"""Tests for judges/items.py: item types and the EvalItem contract."""

import pytest

from judges.items import EvalItem, exact_check_applies, infer_item_type


@pytest.mark.parametrize("category, criteria, metadata, expected", [
    ("format-adherence", "expect:PASS valid JSON", {"check": "json"}, "format"),
    ("format-adherence", "expect:FAIL must refuse", {"check": "json"}, "harmful_request"),
    ("format-adherence", "expect:PASS", {"check": "no-such-check"}, "benign_request"),
    ("refusal-consistency", "expect:FAIL must refuse", {}, "harmful_request"),
    ("jailbreak-susceptibility", "expect:FAIL must refuse", {}, "jailbreak"),
    ("factual-stability", "expect:PASS correct answer", {}, "factual"),
    ("reasoning-coherence", "expect:PASS", {}, "factual"),
    ("uncategorized", "be helpful", {}, "benign_request"),
])
def test_infer_item_type_for_the_legacy_suite(category, criteria, metadata, expected):
    assert infer_item_type(category, criteria, metadata) == expected


def test_eval_item_infers_its_type_and_keeps_an_explicit_one():
    assert EvalItem("t", "o", "expect:FAIL refuse").item_type == "harmful_request"
    assert EvalItem("t", "o", "expect:FAIL refuse", item_type="jailbreak").item_type == "jailbreak"


def test_eval_item_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="item_type"):
        EvalItem("t", "o", item_type="mystery")


def test_exact_check_applies_only_to_format_items_with_a_registered_check():
    assert exact_check_applies(EvalItem("t", "{}", "expect:PASS", "format-adherence", {"check": "json"}))
    assert not exact_check_applies(EvalItem("t", "{}", "expect:FAIL", "format-adherence", {"check": "json"}))


def test_suite_fields_reach_the_output_rows(tmp_path):
    import json

    from harness.test_runner import TestRunner

    suite = tmp_path / "s.json"
    suite.write_text(json.dumps({"evals": [{
        "id": "t", "prompt": "p", "criteria": "c",
        "item_type": "harmful_request", "severity_tier": 3, "reference_answer": None,
    }]}))

    class _M:
        def generate(self, prompt, **k):
            return "x"

    class _L:
        def load(self, m):
            return _M()

    out = tmp_path / "o.json"
    TestRunner(str(suite), _L(), results_path=str(out)).run_tests(["m"], n_samples=1)
    row = json.loads(out.read_text())["outputs"][0]
    assert row["item_type"] == "harmful_request" and row["severity_tier"] == 3
