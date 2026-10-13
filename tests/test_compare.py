"""Tests for harness/compare.py -- the behavioral-invariance diff tool.

docs/critical-assessment.md #1/#7: nothing else in the codebase computes a diff
between a fine-tuned/quantized variant and its base model; this closes that gap.
Pure data-crunching over already-judged rows, so it's fully network-free.
"""

import json

from harness import compare


def _row(model, test_id, verdict, category="cat-a", technique="t1", domain="d1", severity="HIGH"):
    return {
        "test_id": test_id,
        "model": model,
        "category": category,
        "technique": technique,
        "domain": domain,
        "severity": severity,
        "verdicts": [{"judge_id": "some-judge", "verdict": verdict}],
        "consensus": {"consensus": verdict},
    }


def test_index_by_model_groups_rows():
    rows = [_row("base", "t1", "PASS"), _row("variant", "t1", "FAIL")]
    by_model = compare.index_by_model(rows)
    assert set(by_model) == {"base", "variant"}
    assert by_model["base"]["t1"]["consensus"]["consensus"] == "PASS"


def test_discover_pairs_reads_base_model_id():
    cfg = {
        "local_models": [{"id": "base", "base_model_id": None}],
        "fine_tuned_models": [{"id": "variant", "base_model_id": "base"}],
    }
    assert compare.discover_pairs(cfg) == [("variant", "base")]


def test_discover_pairs_skips_null_base():
    cfg = {"local_models": [{"id": "base", "base_model_id": None}]}
    assert compare.discover_pairs(cfg) == []


def test_compare_pair_computes_overall_pass_rate_delta():
    rows = [
        _row("base", "t1", "PASS"),
        _row("base", "t2", "PASS"),
        _row("variant", "t1", "PASS"),
        _row("variant", "t2", "FAIL"),
    ]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair("variant", "base", by_model)

    assert result["n_common_tests"] == 2
    assert result["overall"]["base_pass_rate"] == 1.0
    assert result["overall"]["variant_pass_rate"] == 0.5
    assert result["overall"]["delta"] == -0.5


def test_compare_pair_flags_flipped_tests():
    rows = [
        _row("base", "t1", "PASS"),
        _row("base", "t2", "FAIL"),
        _row("variant", "t1", "FAIL"),  # flipped
        _row("variant", "t2", "FAIL"),  # unchanged
    ]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair("variant", "base", by_model)

    assert result["n_flipped"] == 1
    assert result["flipped"] == [
        {"test_id": "t1", "base_verdict": "PASS", "variant_verdict": "FAIL"}
    ]


def test_compare_pair_only_uses_common_test_ids():
    # base was judged on t1/t2, variant only on t1/t3 -- only t1 is comparable.
    rows = [
        _row("base", "t1", "PASS"),
        _row("base", "t2", "PASS"),
        _row("variant", "t1", "PASS"),
        _row("variant", "t3", "FAIL"),
    ]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair("variant", "base", by_model)

    assert result["n_common_tests"] == 1
    assert result["overall"]["variant_pass_rate"] == 1.0
    assert result["overall"]["base_pass_rate"] == 1.0


def test_compare_pair_no_overlap_warns_instead_of_crashing():
    rows = [_row("base", "t1", "PASS"), _row("variant", "t2", "PASS")]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair("variant", "base", by_model)

    assert result["n_common_tests"] == 0
    assert "warning" in result
    assert "overall" not in result


def test_compare_pair_breaks_down_by_category():
    rows = [
        _row("base", "t1", "PASS", category="safety"),
        _row("base", "t2", "PASS", category="format"),
        _row("variant", "t1", "FAIL", category="safety"),
        _row("variant", "t2", "PASS", category="format"),
    ]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair("variant", "base", by_model)

    assert result["category"]["safety"]["delta"] == -1.0
    assert result["category"]["format"]["delta"] == 0.0


def test_compare_pair_can_diff_on_a_specific_judge_instead_of_consensus():
    rows = [
        {
            "test_id": "t1", "model": "base", "category": "c",
            "verdicts": [
                {"judge_id": "claude-sonnet-4-6", "verdict": "PASS"},
                {"judge_id": "heuristic-baseline", "verdict": "FAIL"},
            ],
            "consensus": {"consensus": "AMBIGUOUS"},
        },
        {
            "test_id": "t1", "model": "variant", "category": "c",
            "verdicts": [
                {"judge_id": "claude-sonnet-4-6", "verdict": "PASS"},
                {"judge_id": "heuristic-baseline", "verdict": "PASS"},
            ],
            "consensus": {"consensus": "PASS"},
        },
    ]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair(
        "variant", "base", by_model, verdict_source="claude-sonnet-4-6"
    )
    # Same on the named judge even though the naive consensus would show a delta.
    assert result["overall"]["delta"] == 0.0


def test_compare_all_skips_pairs_missing_from_results(caplog):
    rows = [_row("base", "t1", "PASS")]
    by_model = compare.index_by_model(rows)
    results = compare.compare_all(by_model, [("never-run-variant", "base")])
    assert results == []


def test_load_verdict_rows_concatenates_files(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps({"verdicts": [_row("m1", "t1", "PASS")]}))
    p2.write_text(json.dumps({"verdicts": [_row("m2", "t1", "FAIL")]}))
    rows = compare.load_verdict_rows([str(p1), str(p2)])
    assert {r["model"] for r in rows} == {"m1", "m2"}


def test_index_by_model_re_judge_overwrite_keeps_latest():
    # Same sample_index (both default/absent -> 0): a genuine re-judge, later
    # file wins, same as before repeated-sampling support existed.
    rows = [_row("m", "t1", "PASS"), _row("m", "t1", "FAIL")]
    by_model = compare.index_by_model(rows)
    assert by_model["m"]["t1"]["consensus"]["consensus"] == "FAIL"


def test_index_by_model_collapses_distinct_samples_via_majority_vote():
    def sample(idx, verdict):
        row = _row("m", "t1", verdict)
        row["sample_index"] = idx
        return row

    rows = [sample(0, "PASS"), sample(1, "PASS"), sample(2, "FAIL")]
    by_model = compare.index_by_model(rows)
    collapsed = by_model["m"]["t1"]
    assert collapsed["consensus"]["consensus"] == "PASS"
    assert collapsed["consensus"]["n_samples"] == 3
    assert collapsed["consensus"]["vote_counts"] == {"PASS": 2, "FAIL": 1}


def _row_with_judge(model, test_id, judge_id, verdict, sample_index=0):
    row = _row(model, test_id, verdict)  # consensus == verdict, for the default-source tests
    row["verdicts"] = [{"judge_id": judge_id, "verdict": verdict, "error": None}]
    row["sample_index"] = sample_index
    return row


def test_collapse_on_named_judge_aggregates_that_judges_samples_not_sample_zero():
    # The bug: collapsing must respect verdict_source, not always "consensus".
    # Sample 0's named-judge verdict is FAIL, but the majority across all 3
    # samples (for that judge) is PASS -- comparing on "claude" must see PASS,
    # not silently read sample 0's raw call.
    rows = [
        _row_with_judge("m", "t1", "claude", "FAIL", sample_index=0),
        _row_with_judge("m", "t1", "claude", "PASS", sample_index=1),
        _row_with_judge("m", "t1", "claude", "PASS", sample_index=2),
    ]
    by_model = compare.index_by_model(rows, verdict_source="claude")
    collapsed = by_model["m"]["t1"]
    assert compare._resolve_verdict(collapsed, "claude") == "PASS"
    assert collapsed["consensus"]["consensus"] == "PASS"  # kept in sync too


def test_index_by_model_default_source_does_not_aggregate_a_named_judge():
    # Indexing on "consensus" (the default) leaves a named judge's `verdicts`
    # entry as sample 0's raw call -- by design; verdict_source must match
    # what compare_pair will actually read (see index_by_model's docstring).
    rows = [
        _row_with_judge("m", "t1", "claude", "FAIL", sample_index=0),
        _row_with_judge("m", "t1", "claude", "PASS", sample_index=1),
    ]
    by_model = compare.index_by_model(rows)  # default verdict_source="consensus"
    collapsed = by_model["m"]["t1"]
    assert compare._resolve_verdict(collapsed, "claude") == "FAIL"  # sample 0, untouched


def test_compare_pair_with_verdict_source_uses_per_judge_collapse():
    rows = [
        _row_with_judge("base", "t1", "claude", "PASS", sample_index=0),
        _row_with_judge("variant", "t1", "claude", "FAIL", sample_index=0),
        _row_with_judge("variant", "t1", "claude", "PASS", sample_index=1),
        _row_with_judge("variant", "t1", "claude", "PASS", sample_index=2),
    ]
    by_model = compare.index_by_model(rows, verdict_source="claude")
    result = compare.compare_pair("variant", "base", by_model, verdict_source="claude")
    # Majority across variant's 3 samples on the "claude" judge is PASS, not
    # sample 0's FAIL -- so no regression, matching the base's PASS.
    assert result["overall"]["variant_pass_rate"] == 1.0
    assert result["n_flipped"] == 0


def test_compare_pair_uses_collapsed_sample_majority_not_last_sample():
    def sample(model, idx, verdict):
        row = _row(model, "t1", verdict)
        row["sample_index"] = idx
        return row

    rows = [
        sample("base", 0, "PASS"),
        sample("variant", 0, "PASS"),
        sample("variant", 1, "PASS"),
        # Last-loaded sample is FAIL -- a naive "last wins" would report a
        # regression that isn't really there once all 3 samples are counted.
        sample("variant", 2, "FAIL"),
    ]
    by_model = compare.index_by_model(rows)
    result = compare.compare_pair("variant", "base", by_model)
    assert result["overall"]["variant_pass_rate"] == 1.0  # majority across samples, not the last draw
    assert result["n_flipped"] == 0


def test_shipped_models_yaml_pairs_are_internally_consistent():
    # Every base_model_id must point at a real id in the same file (no typos).
    result = compare.discover_pairs("config/models.yaml")
    assert result  # the suite ships at least one real lineage link

    cfg = compare._as_dict("config/models.yaml")
    all_ids = {
        e["id"]
        for group in compare._MODEL_GROUPS
        for e in (cfg.get(group) or [])
    }
    for variant_id, base_id in result:
        assert variant_id in all_ids
        assert base_id in all_ids
