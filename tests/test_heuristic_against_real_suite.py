"""HeuristicJudge run against the actual 61-case suite (not synthetic 1-3 case
fixtures) -- the first place anything exercises the full real suite through
the judging path.

This is the concrete check that surfaced (and now guards) the abstention
finding: on a neutral, non-refusing model output, every factual-stability /
reasoning-coherence case with an `expect:PASS` directive is a structural
abstention (the heuristic has no way to verify factual/logical correctness by
keyword match) -- not a real "AMBIGUOUS, I looked and it's unclear" verdict.
Before `Verdict.abstained` existed, all 22 of these silently counted as real
disagreement in aggregate_verdicts' vote and judges.meta_eval's kappa/bias
stats. See judges/heuristic_judge.py and docs/critical-assessment.md.
"""

from harness.test_suite import load_test_suite
from judges.heuristic_judge import HeuristicJudge

_NEUTRAL_OUTPUT = "This is a neutral, non-refusing, unremarkable model output."


def _evaluate_all(output: str):
    tests = load_test_suite("data/test_suite_v1.json")
    judge = HeuristicJudge(refusal_votes=True)
    return [
        judge.evaluate(t.id, output, t.criteria, t.category, t.metadata) for t in tests
    ]


def test_heuristic_abstains_on_most_factual_and_reasoning_cases():
    verdicts = _evaluate_all(_NEUTRAL_OUTPUT)
    abstained_ids = {v.test_id for v in verdicts if v.abstained}

    # Exact counts observed on the shipped suite; a suite edit that changes
    # this meaningfully should be a deliberate, visible change to this test.
    assert len(abstained_ids) == 22

    tests = load_test_suite("data/test_suite_v1.json")
    by_id = {t.id: t for t in tests}
    abstained_categories = {by_id[tid].category for tid in abstained_ids}
    # Abstentions cluster in the categories a keyword scan structurally can't
    # verify -- refusal/jailbreak cases always get a real (non-abstained)
    # refusal-marker judgement, per the suite's expect:FAIL vocabulary.
    assert "factual-stability" in abstained_categories
    assert "reasoning-coherence" in abstained_categories
    assert "refusal-consistency" not in abstained_categories
    assert "jailbreak-susceptibility" not in abstained_categories


def test_heuristic_never_abstains_on_refusal_or_format_check_cases():
    # A neutral (non-refusing) output on an expect:FAIL case is a real FAIL
    # verdict (not an abstention) -- the heuristic has a definite opinion.
    verdicts = _evaluate_all(_NEUTRAL_OUTPUT)
    tests = load_test_suite("data/test_suite_v1.json")
    by_id = {t.id: t for t in tests}

    for v in verdicts:
        if by_id[v.test_id].category in ("refusal-consistency", "jailbreak-susceptibility"):
            assert v.abstained is False, v.test_id
