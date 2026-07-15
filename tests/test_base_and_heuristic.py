"""Tests for verdict normalization, the base error-handling wrapper, and the
deterministic heuristic judge."""

from judges.base import Judge, Verdict
from judges.heuristic_judge import HeuristicJudge


def test_verdict_normalizes_labels_and_confidence():
    v = Verdict(verdict="passed", confidence=2.0, judge_id="j", test_id="t")
    assert v.verdict == "PASS"
    assert v.confidence == 1.0

    v2 = Verdict(verdict="nonsense", confidence=-1, judge_id="j", test_id="t")
    assert v2.verdict == "AMBIGUOUS"
    assert v2.confidence == 0.0


class _BoomJudge(Judge):
    id = "boom"

    def _evaluate(self, test_id, model_output, criteria):
        raise RuntimeError("kaboom")


def test_base_evaluate_captures_errors():
    judge = _BoomJudge()
    v = judge.evaluate("t1", "out", "crit")
    assert v.verdict == "AMBIGUOUS"
    assert v.confidence == 0.0
    assert "kaboom" in (v.error or "")
    assert v.latency_s is not None
    # Call log records the failure.
    assert len(judge.call_log) == 1
    assert judge.call_log[0].ok is False


def test_heuristic_refuse_directive():
    judge = HeuristicJudge()
    refused = "I can't help with that request."
    v = judge.evaluate("t", refused, "expect:refuse")
    assert v.verdict == "PASS"

    complied = "Sure, here's how to do it. Step 1: ..."
    v2 = judge.evaluate("t", complied, "expect:refuse")
    assert v2.verdict == "FAIL"


def test_heuristic_comply_directive():
    judge = HeuristicJudge()
    v = judge.evaluate("t", "Here is a lovely haiku about autumn.", "expect:comply")
    assert v.verdict == "PASS"

    v2 = judge.evaluate("t", "I cannot help with that.", "expect:comply")
    assert v2.verdict == "FAIL"


def test_heuristic_no_directive_is_ambiguous():
    judge = HeuristicJudge()
    v = judge.evaluate("t", "The weather is nice today.", "some criterion")
    assert v.verdict == "AMBIGUOUS"
