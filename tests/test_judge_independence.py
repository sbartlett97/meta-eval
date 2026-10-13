"""Judge/model independence, enforced on the shipped configs (validation gap C20).

A judge that shares weights with a model it scores is grading itself, which
silently invalidates any judge-bias analysis (see CLAUDE.md). This was only
documented before; these tests make it fail loudly.
"""

import yaml

_MODEL_GROUPS = ("local_models", "fine_tuned_models", "remote_models")


def _models():
    cfg = yaml.safe_load(open("config/models.yaml"))
    return [e for g in _MODEL_GROUPS for e in cfg.get(g) or []]


def _judges():
    return yaml.safe_load(open("config/judges.yaml"))["judges"]


def _names(entry):
    return {entry.get(k) for k in ("id", "checkpoint", "model")} - {None}


def test_no_judge_shares_an_id_or_checkpoint_with_a_model_under_test():
    model_names = set().union(*(_names(m) for m in _models()))
    clashes = {j["id"]: _names(j) & model_names for j in _judges() if _names(j) & model_names}
    assert clashes == {}, f"judge(s) share weights with a model under test: {clashes}"


def test_every_model_and_judge_declares_a_family():
    missing = [e["id"] for e in _models() + _judges() if "family" not in e]
    assert missing == []


def test_name_claimed_teachers_are_known_judge_families():
    # Needed to report "Claude judge grading a model named as Claude-distilled".
    judge_families = {j.get("family") for j in _judges()}
    claims = {m["id"]: m["name_claims_teacher"] for m in _models() if m.get("name_claims_teacher")}
    assert claims and set(claims.values()) <= judge_families
