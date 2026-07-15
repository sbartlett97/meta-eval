"""Tests for the test-suite loader, judge panel, factory, and config validity."""

import json

import pytest
import yaml

from harness.test_suite import load_test_suite
from judges.base import Judge, Verdict
from judges.factory import build_judges
from judges.judge_panel import JudgePanel, aggregate_verdicts


def test_load_example_suite():
    tests = load_test_suite("data/test_suite_v1.jsonl")
    assert len(tests) == 3
    ids = {t.id for t in tests}
    assert "safety-refusal-01" in ids
    assert all(t.criteria for t in tests)


def test_load_suite_rejects_duplicates(tmp_path):
    p = tmp_path / "dup.jsonl"
    row = json.dumps({"id": "x", "prompt": "p", "criteria": "c"})
    p.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_test_suite(str(p))


def test_load_suite_rejects_missing_field(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "x", "prompt": "p"}) + "\n")
    with pytest.raises(ValueError, match="missing required"):
        load_test_suite(str(p))


class _FixedJudge(Judge):
    def __init__(self, jid, verdict, conf):
        super().__init__()
        self.id = jid
        self._v = verdict
        self._c = conf

    def _evaluate(self, test_id, model_output, criteria):
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


def test_panel_requires_a_judge():
    with pytest.raises(ValueError):
        JudgePanel([])


def test_factory_builds_from_config():
    # Build only the deterministic judge to avoid needing API keys / servers.
    cfg = {
        "judges": [
            {"id": "heuristic-baseline", "provider": "deterministic", "access": "none",
             "priority": 3, "enabled": True},
            {"id": "mistral-7b-local", "provider": "local", "access": "vllm",
             "vllm_port": 8000, "model": "unsloth/Mistral-7B-v0.3-GGUF",
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
