"""Tests for the test-suite loader, judge panel, factory, and config validity."""

import json

import pytest
import yaml

from harness.test_suite import load_test_suite
from judges.base import Judge, Verdict
from judges.factory import build_judges
from judges.judge_panel import EvalItem, JudgePanel, aggregate_verdicts


def test_load_example_suite():
    tests = load_test_suite("data/test_suite_v1.json")
    assert len(tests) == 61
    ids = {t.id for t in tests}
    assert "refusal-cyber-01" in ids
    assert all(t.criteria for t in tests)
    assert all(t.turns for t in tests)  # every case resolves to >=1 turn

    multi_turn = [t for t in tests if t.is_multi_turn]
    assert len(multi_turn) == 6  # the "[Follow-up]"-marker cases, now real turns

    with_system_prompt = [t for t in tests if t.system_prompt]
    assert len(with_system_prompt) == 14


def test_load_suite_parses_optional_system_prompt(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"evals": [
        {"id": "with", "prompt": "p", "criteria": "c",
         "system_prompt": "You are terse."},
        {"id": "null", "prompt": "p", "criteria": "c", "system_prompt": None},
        {"id": "omitted", "prompt": "p", "criteria": "c"},
    ]}))
    by_id = {t.id: t for t in load_test_suite(str(p))}
    assert by_id["with"].system_prompt == "You are terse."
    assert by_id["null"].system_prompt is None
    assert by_id["omitted"].system_prompt is None


def test_load_json_suite_ignores_extra_top_level_keys(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({
        "_comment": "notes for the human editor",
        "evals": [{"id": "a", "prompt": "p", "criteria": "c", "category": "cat"}],
    }))
    tests = load_test_suite(str(p))
    assert [t.id for t in tests] == ["a"]
    assert tests[0].category == "cat"


def test_load_json_suite_accepts_bare_array(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps([{"id": "a", "prompt": "p", "criteria": "c"}]))
    assert [t.id for t in load_test_suite(str(p))] == ["a"]


def test_load_json_suite_requires_evals_key(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"cases": []}))
    with pytest.raises(ValueError, match="evals"):
        load_test_suite(str(p))


def test_load_suite_rejects_duplicates(tmp_path):
    p = tmp_path / "dup.json"
    row = {"id": "x", "prompt": "p", "criteria": "c"}
    p.write_text(json.dumps({"evals": [row, row]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_test_suite(str(p))


def test_load_suite_rejects_missing_field(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"evals": [{"id": "x", "prompt": "p"}]}))
    with pytest.raises(ValueError, match="missing required"):
        load_test_suite(str(p))


def test_load_legacy_jsonl_still_works(tmp_path):
    p = tmp_path / "suite.jsonl"
    p.write_text(
        "# a comment line\n"
        + json.dumps({"id": "a", "prompt": "p", "criteria": "c"}) + "\n"
        + json.dumps({"id": "b", "prompt": "p", "criteria": "c"}) + "\n"
    )
    assert [t.id for t in load_test_suite(str(p))] == ["a", "b"]


class _FixedJudge(Judge):
    def __init__(self, jid, verdict, conf):
        super().__init__()
        self.id = jid
        self._v = verdict
        self._c = conf

    def _evaluate(self, test_id, model_output, criteria, category="uncategorized", metadata=None):
        return Verdict(self._v, self._c, self.id, test_id)


def test_panel_collects_all_verdicts():
    panel = JudgePanel([_FixedJudge("a", "PASS", 0.9), _FixedJudge("b", "FAIL", 0.5)])
    result = panel.evaluate("t1", "out", "crit")
    assert len(result.verdicts) == 2
    assert set(result.by_judge()) == {"a", "b"}


def test_aggregate_majority_vote():
    result = JudgePanel(
        [
            _FixedJudge("a", "PASS", 0.9),
            _FixedJudge("b", "PASS", 0.8),
            _FixedJudge("c", "FAIL", 0.5),
        ]
    ).evaluate("t1", "out", "crit")
    agg = aggregate_verdicts(result)
    assert agg["consensus"] == "PASS"
    assert agg["agreement"] == pytest.approx(2 / 3, abs=1e-3)


def test_aggregate_tie_broken_by_confidence():
    result = JudgePanel(
        [_FixedJudge("a", "PASS", 0.6), _FixedJudge("b", "FAIL", 0.9)]
    ).evaluate("t1", "out", "crit")
    agg = aggregate_verdicts(result)
    assert agg["consensus"] == "FAIL"


def test_aggregate_excludes_errored_verdicts_from_the_vote():
    from judges.judge_panel import PanelResult

    result = PanelResult(
        test_id="t1",
        verdicts=[
            Verdict("PASS", 0.9, "a", "t1"),
            Verdict("PASS", 0.8, "b", "t1"),
            # This judge's call itself failed -- an AMBIGUOUS from a timeout,
            # not a genuine judgement. It must not swing (or even count toward)
            # the vote (docs/critical-assessment.md #2.9).
            Verdict("AMBIGUOUS", 0.0, "c", "t1", error="timeout"),
        ],
    )
    agg = aggregate_verdicts(result)
    assert agg["consensus"] == "PASS"
    assert agg["n_judges"] == 2  # only the two that actually succeeded
    assert agg["n_errored"] == 1
    assert agg["agreement"] == 1.0  # both real verdicts agreed


def test_aggregate_all_errored_falls_back_to_ambiguous():
    from judges.judge_panel import PanelResult

    result = PanelResult(
        test_id="t1",
        verdicts=[Verdict("AMBIGUOUS", 0.0, "a", "t1", error="boom")],
    )
    agg = aggregate_verdicts(result)
    assert agg["consensus"] == "AMBIGUOUS"
    assert agg["n_judges"] == 0
    assert agg["n_errored"] == 1


def test_aggregate_excludes_abstained_verdicts_from_the_vote():
    from judges.judge_panel import PanelResult

    result = PanelResult(
        test_id="t1",
        verdicts=[
            Verdict("PASS", 0.9, "claude", "t1"),
            Verdict("PASS", 0.8, "gpt4o", "t1"),
            # Heuristic couldn't verify factual correctness -- an abstention,
            # not a real "I looked and it's unclear" AMBIGUOUS vote. Must not
            # count toward n_judges/agreement, same as an errored call.
            Verdict("AMBIGUOUS", 0.3, "heuristic-baseline", "t1", abstained=True),
        ],
    )
    agg = aggregate_verdicts(result)
    assert agg["consensus"] == "PASS"
    assert agg["n_judges"] == 2
    assert agg["n_abstained"] == 1
    assert agg["agreement"] == 1.0  # both real votes agreed; the abstention is invisible to this


def test_panel_requires_a_judge():
    with pytest.raises(ValueError):
        JudgePanel([])


def test_evaluate_batch_is_judge_outer():
    calls = []

    class _RecordingJudge(Judge):
        def __init__(self, jid):
            super().__init__()
            self.id = jid

        def _evaluate(self, test_id, model_output, criteria, category="uncategorized", metadata=None):
            calls.append((self.id, test_id))
            return Verdict("PASS", 1.0, self.id, test_id)

    panel = JudgePanel([_RecordingJudge("a"), _RecordingJudge("b")])
    items = [EvalItem("t1", "o1", "c1"), EvalItem("t2", "o2", "c2")]
    results = panel.evaluate_batch(items)

    # Judge a processes every item before judge b starts (judge-outer).
    assert calls == [("a", "t1"), ("a", "t2"), ("b", "t1"), ("b", "t2")]
    # Per-item results still carry both judges' verdicts, in judge order.
    assert results[0].test_id == "t1"
    assert [v.judge_id for v in results[0].verdicts] == ["a", "b"]
    assert [v.judge_id for v in results[1].verdicts] == ["a", "b"]


def test_factory_builds_from_config():
    # Build only the deterministic judge to avoid needing API keys / servers.
    cfg = {
        "judges": [
            {"id": "heuristic-baseline", "provider": "deterministic", "access": "none",
             "priority": 3, "enabled": True},
            {"id": "mistral-7b-local", "provider": "local", "access": "llamacpp",
             "model": "unsloth/Mistral-7B-v0.3-GGUF", "gguf_file": "*Q4_K_M.gguf",
             "priority": 2, "enabled": True},
            {"id": "disabled", "provider": "deterministic", "access": "none",
             "priority": 3, "enabled": False},
        ]
    }
    judges = build_judges(cfg)
    ids = {j.id for j in judges}
    assert "heuristic-baseline" in ids
    assert "mistral-7b-local" in ids
    assert "disabled" not in ids  # respects enabled: false


def test_factory_respects_max_priority():
    cfg = {
        "judges": [
            {"id": "hi", "provider": "deterministic", "access": "none", "priority": 3},
            {"id": "lo", "provider": "anthropic", "model": "claude-sonnet-4-6",
             "access": "api", "priority": 1},
        ]
    }
    judges = build_judges(cfg, max_priority=1)
    assert {j.id for j in judges} == {"lo"}


def test_shipped_configs_are_valid_yaml():
    for path in ("config/judges.yaml", "config/models.yaml", "config/hardware_profile.yaml"):
        with open(path) as fh:
            assert yaml.safe_load(fh) is not None
