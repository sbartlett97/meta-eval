"""Tests for the per-judge content policy (judges.yaml `skip:`; network-free)."""

import pytest

from judges.base import Judge, Verdict
from judges.factory import build_judge
from judges.items import EvalItem
from judges.judge_panel import JudgePanel


class CountingJudge(Judge):
    id = "counting"
    provider = "test"

    def __init__(self, skip_tiers=(), skip_domains=()):
        super().__init__()
        self.skip_tiers, self.skip_domains = tuple(skip_tiers), tuple(skip_domains)
        self.seen = []

    def _evaluate(self, item):
        self.seen.append(item.test_id)
        return Verdict("PASS", 1.0, self.id, item.test_id)


def _harm(test_id, tier=None, domain="", item_type="harmful_request"):
    return EvalItem(test_id, "out", "expect:FAIL refuse", item_type=item_type,
                    severity_tier=tier, domain=domain)


def test_policy_skips_by_tier_and_domain_without_a_call():
    judge = CountingJudge(skip_tiers=(3,), skip_domains=("biology",))
    items = [_harm("t2", 2), _harm("t3", 3), _harm("bio", 2, "biology"),
             _harm("jb", 2, "biology", item_type="jailbreak")]
    verdicts = JudgePanel([judge]).judge_all(judge, items)
    assert judge.seen == ["t2"]
    assert [v.abstain_reason for v in verdicts] == [
        None, "skipped_by_policy:tier_3", "skipped_by_policy:domain:biology",
        "skipped_by_policy:domain:biology"]
    assert all(v.abstained for v in verdicts[1:])


def test_untiered_harmful_item_counts_as_tier_3():
    judge = CountingJudge(skip_tiers=(3,))
    assert judge.policy_skip_reason(_harm("x")) == "untiered"


def test_policy_never_skips_non_harmful_items():
    judge = CountingJudge(skip_tiers=(3,), skip_domains=("biology",))
    item = EvalItem("f", "out", "expect:PASS 4", category="factual-stability", domain="biology")
    assert item.item_type == "factual"
    assert judge.policy_skip_reason(item) is None


def test_factory_reads_skip_and_rejects_unknown_keys():
    entry = {"id": "h", "provider": "deterministic", "skip": {"severity_tiers": [3], "domains": ["cyber"]}}
    judge = build_judge(entry)
    assert (judge.skip_tiers, judge.skip_domains) == ((3,), ("cyber",))
    assert judge.describe()["skip"] == {"severity_tiers": [3], "domains": ["cyber"]}
    with pytest.raises(ValueError):
        build_judge({"id": "h", "provider": "deterministic", "skip": {"tiers": [3]}})


def test_shipped_cloud_judges_skip_tier_3():
    import yaml
    judges = {j["id"]: j for j in yaml.safe_load(open("config/judges.yaml"))["judges"]}
    for entry in judges.values():
        if entry.get("access") == "api":
            assert 3 in entry["skip"]["severity_tiers"], entry["id"]
    claude = judges["claude-sonnet-5-5"]["skip"]["domains"]
    assert {"cybersecurity", "biology"} <= set(claude)
