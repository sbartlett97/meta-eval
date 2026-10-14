"""Tests for the decline memory (judges/declines.py; network-free)."""

import json

from judges.base import Judge, Verdict
from judges.declines import DeclineRegistry, backfill, content_hash
from judges.items import EvalItem
from judges.judge_panel import JudgePanel


class DecliningJudge(Judge):
    """Declines items whose response contains "bad"."""
    id = "cloud"
    provider = "test"

    def __init__(self):
        super().__init__()
        self.sent = []

    def _evaluate(self, item):
        self.sent.append(item.test_id)
        if "bad" in item.model_output:
            return Verdict("AMBIGUOUS", 0.0, self.id, item.test_id, abstained=True,
                           abstain_reason="judge_declined:cyber")
        return Verdict("PASS", 1.0, self.id, item.test_id)


def _item(test_id, output, request="req"):
    return EvalItem(test_id, output, "expect:FAIL refuse", item_type="harmful_request",
                    turns=(request,), severity_tier=2)


def test_declined_content_is_never_resent(tmp_path):
    path = str(tmp_path / "declines.jsonl")
    items = [_item("a", "bad stuff"), _item("b", "fine")]

    judge = DecliningJudge()
    JudgePanel([judge], decline_registry=DeclineRegistry(path)).judge_all(judge, items)
    assert judge.sent == ["a", "b"]

    # A later run (new registry object, same file) skips the declined content.
    judge2 = DecliningJudge()
    verdicts = JudgePanel([judge2], decline_registry=DeclineRegistry(path)).judge_all(judge2, items)
    assert judge2.sent == ["b"]
    assert verdicts[0].abstain_reason == "skipped_by_policy:previously_declined"
    assert verdicts[0].abstained


def test_registry_is_per_judge_and_per_content(tmp_path):
    reg = DeclineRegistry(str(tmp_path / "d.jsonl"))
    item = _item("a", "bad")
    assert reg.record("cloud", item, "judge_declined:bio") is True
    assert reg.record("cloud", item, "judge_declined:bio") is False
    assert reg.declined("cloud", item)
    assert not reg.declined("other", item)
    # Same test id, different response: different content.
    assert not reg.declined("cloud", _item("a", "different"))


def test_registry_file_holds_no_content(tmp_path):
    path = tmp_path / "d.jsonl"
    DeclineRegistry(str(path)).record("cloud", _item("a", "SECRET RESPONSE", "SECRET REQUEST"),
                                      "judge_declined:cyber")
    text = path.read_text()
    assert "SECRET" not in text
    assert json.loads(text)["content_sha256"] == content_hash(_item("a", "SECRET RESPONSE", "SECRET REQUEST"))


def test_hash_ignores_rubric_inputs():
    a = EvalItem("t", "out", "criteria one", turns=("r",), severity_tier=2)
    b = EvalItem("t", "out", "criteria two", turns=("r",), severity_tier=3)
    assert content_hash(a) == content_hash(b)


def test_backfill_from_verdict_rows(tmp_path):
    reg = DeclineRegistry(str(tmp_path / "d.jsonl"))
    outputs = [{"test_id": "a", "model": "m", "sample_index": 0, "turns": ["req"], "output": "bad"},
               {"test_id": "b", "model": "m", "sample_index": 0, "turns": ["req"], "output": "ok"}]
    verdicts = [
        {"test_id": "a", "model": "m", "sample_index": 0, "verdicts": [
            {"judge_id": "cloud", "abstain_reason": "judge_declined:cyber"},
            {"judge_id": "other", "abstain_reason": "skipped_by_policy:tier_3"}]},
        {"test_id": "b", "model": "m", "sample_index": 0, "verdicts": [
            {"judge_id": "cloud", "abstain_reason": None}]},
    ]
    assert backfill(reg, outputs, verdicts) == 1
    assert backfill(reg, outputs, verdicts) == 0
    assert reg.declined("cloud", _item("a", "bad"))
