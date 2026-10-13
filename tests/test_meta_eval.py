"""Tests for judges/meta_eval.py -- the real inter-judge agreement/bias analysis.

docs/critical-assessment.md #1/#8: aggregate_verdicts is a per-item majority
vote; nothing computed cross-judge agreement or bias before this module.
"""

import pytest

from judges import meta_eval


def _v(judge_id, verdict, error=None, abstained=False):
    return {
        "judge_id": judge_id, "verdict": verdict, "confidence": 0.8,
        "error": error, "abstained": abstained,
    }


def _row(test_id, verdicts, category="cat-a", technique="t1", severity="HIGH"):
    return {
        "test_id": test_id,
        "category": category,
        "technique": technique,
        "severity": severity,
        "verdicts": verdicts,
    }


def test_judge_ids_collects_unique_ids_in_order():
    rows = [_row("t1", [_v("a", "PASS"), _v("b", "FAIL")]), _row("t2", [_v("a", "PASS")])]
    assert meta_eval.judge_ids(rows) == ["a", "b"]


def test_pairwise_agreement_perfect_agreement_gives_kappa_one():
    rows = [_row(f"t{i}", [_v("a", "PASS"), _v("b", "PASS")]) for i in range(5)]
    rows += [_row(f"u{i}", [_v("a", "FAIL"), _v("b", "FAIL")]) for i in range(5)]
    report = meta_eval.pairwise_agreement(rows)
    stats = report["a|b"]
    assert stats["n_common"] == 10
    assert stats["agreement"] == 1.0
    assert stats["kappa"] == 1.0


def test_pairwise_agreement_chance_level_gives_low_kappa():
    # Both judges independently split PASS/FAIL 50/50, but never on the same
    # items -- always disagreeing -- so raw agreement is 0 and kappa should
    # reflect worse-than-chance, not just "some disagreement".
    rows = []
    for i in range(10):
        a = "PASS" if i % 2 == 0 else "FAIL"
        b = "FAIL" if i % 2 == 0 else "PASS"
        rows.append(_row(f"t{i}", [_v("a", a), _v("b", b)]))
    report = meta_eval.pairwise_agreement(rows)
    stats = report["a|b"]
    assert stats["agreement"] == 0.0
    assert stats["kappa"] < 0


def test_pairwise_agreement_excludes_errored_verdicts():
    rows = [
        _row("t1", [_v("a", "PASS"), _v("b", "PASS")]),
        _row("t2", [_v("a", "AMBIGUOUS", error="timeout"), _v("b", "PASS")]),
    ]
    stats = meta_eval.pairwise_agreement(rows)["a|b"]
    # t2's judge-a verdict errored -> excluded, so only t1 is comparable.
    assert stats["n_common"] == 1
    assert stats["agreement"] == 1.0


def test_pairwise_agreement_excludes_abstained_verdicts():
    # The bug this guards: HeuristicJudge abstaining on factual/reasoning
    # criteria it can't keyword-verify must not register as "heuristic
    # disagreed with claude here" -- that would drag down heuristic's kappa
    # for a reason that has nothing to do with judge quality.
    rows = [
        _row("t1", [_v("heuristic-baseline", "PASS"), _v("claude", "PASS")]),
        _row("t2", [_v("heuristic-baseline", "AMBIGUOUS", abstained=True), _v("claude", "PASS")]),
    ]
    stats = meta_eval.pairwise_agreement(rows)["claude|heuristic-baseline"]
    assert stats["n_common"] == 1  # t2 excluded: heuristic abstained there
    assert stats["agreement"] == 1.0


def test_pairwise_agreement_handles_no_common_items():
    rows = [_row("t1", [_v("a", "PASS", error="down")]), _row("t2", [_v("b", "PASS")])]
    stats = meta_eval.pairwise_agreement(rows)["a|b"]
    assert stats["n_common"] == 0
    assert stats["agreement"] is None
    assert stats["kappa"] is None


def test_divergence_from_reference_extracts_relevant_pairs():
    rows = [
        _row("t1", [_v("heuristic-baseline", "PASS"), _v("claude", "PASS"), _v("gpt4", "FAIL")]),
    ]
    div = meta_eval.divergence_from_reference(rows, "heuristic-baseline")
    assert set(div) == {"claude", "gpt4"}
    assert div["claude"]["agreement"] == 1.0
    assert div["gpt4"]["agreement"] == 0.0


def test_bias_by_field_computes_per_judge_per_category_pass_rate():
    rows = [
        _row("t1", [_v("a", "PASS")], category="safety"),
        _row("t2", [_v("a", "PASS")], category="safety"),
        _row("t3", [_v("a", "FAIL")], category="safety"),
        _row("t4", [_v("a", "PASS")], category="format"),
    ]
    report = meta_eval.bias_by_field(rows, "category")
    assert report["a"]["safety"]["n"] == 3
    assert report["a"]["safety"]["PASS"] == 2
    assert report["a"]["safety"]["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["a"]["format"]["pass_rate"] == 1.0


def test_bias_by_field_excludes_errored_verdicts():
    rows = [_row("t1", [_v("a", "PASS", error="boom")], category="safety")]
    report = meta_eval.bias_by_field(rows, "category")
    assert "a" not in report  # nothing but an errored call -> no signal at all


def test_bias_by_field_excludes_abstained_verdicts():
    rows = [
        _row("t1", [_v("heuristic-baseline", "AMBIGUOUS", abstained=True)], category="factual"),
        _row("t2", [_v("heuristic-baseline", "PASS")], category="factual"),
    ]
    report = meta_eval.bias_by_field(rows, "category")
    # Only the real verdict counts -- the abstention doesn't inflate n or drag
    # the pass_rate toward "AMBIGUOUS" for a category the judge can't assess.
    assert report["heuristic-baseline"]["factual"]["n"] == 1
    assert report["heuristic-baseline"]["factual"]["pass_rate"] == 1.0


def test_meta_evaluation_report_shape():
    rows = [
        _row("t1", [_v("heuristic-baseline", "PASS"), _v("claude", "PASS")], category="safety"),
        _row("t2", [_v("heuristic-baseline", "FAIL"), _v("claude", "PASS")], category="format"),
    ]
    report = meta_eval.meta_evaluation_report(rows)
    assert report["n_items"] == 2
    assert set(report["judges"]) == {"heuristic-baseline", "claude"}
    assert "claude|heuristic-baseline" in report["pairwise_agreement"]  # sorted alphabetically
    assert "bias_by_category" in report
    assert "bias_by_technique" in report
    assert "bias_by_severity" in report
    assert report["divergence_from_reference"]["reference_judge_id"] == "heuristic-baseline"
    assert "claude" in report["divergence_from_reference"]["judges"]


def test_meta_evaluation_report_skips_divergence_if_reference_absent():
    rows = [_row("t1", [_v("claude", "PASS")])]
    report = meta_eval.meta_evaluation_report(rows, reference_judge_id="heuristic-baseline")
    assert "divergence_from_reference" not in report


def _sample_row(model, test_id, sample_index, consensus):
    return {
        "test_id": test_id,
        "model": model,
        "sample_index": sample_index,
        "consensus": {"consensus": consensus},
        "verdicts": [],
    }


def test_sample_variance_computes_pass_rate_and_stderr():
    rows = [
        _sample_row("m", "t1", 0, "PASS"),
        _sample_row("m", "t1", 1, "PASS"),
        _sample_row("m", "t1", 2, "FAIL"),
    ]
    report = meta_eval.sample_variance(rows)
    entry = report["m::t1"]
    assert entry["n_samples"] == 3
    assert entry["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert entry["stderr"] > 0
    assert entry["labels"] == ["PASS", "PASS", "FAIL"]


def test_sample_variance_single_sample_has_zero_stderr():
    rows = [_sample_row("m", "t1", 0, "PASS")]
    entry = meta_eval.sample_variance(rows)["m::t1"]
    assert entry["n_samples"] == 1
    assert entry["pass_rate"] == 1.0
    assert entry["stderr"] == 0.0


def test_sample_variance_groups_by_model_and_test_id_separately():
    rows = [
        _sample_row("m1", "t1", 0, "PASS"),
        _sample_row("m2", "t1", 0, "FAIL"),
    ]
    report = meta_eval.sample_variance(rows)
    assert set(report) == {"m1::t1", "m2::t1"}
    assert report["m1::t1"]["pass_rate"] == 1.0
    assert report["m2::t1"]["pass_rate"] == 0.0


def test_meta_evaluation_report_includes_sample_variance_only_when_repeated():
    single = [_sample_row("m", "t1", 0, "PASS")]
    report_single = meta_eval.meta_evaluation_report(single, reference_judge_id="none")
    assert "sample_variance" not in report_single

    repeated = [_sample_row("m", "t1", 0, "PASS"), _sample_row("m", "t1", 1, "FAIL")]
    report_repeated = meta_eval.meta_evaluation_report(repeated, reference_judge_id="none")
    assert "sample_variance" in report_repeated
    assert report_repeated["sample_variance"]["m::t1"]["n_samples"] == 2
