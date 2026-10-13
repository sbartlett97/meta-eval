"""Tests for judge selection in judges.factory (network-free)."""

import pytest

from judges.factory import build_judges, select_judge_entries

_CFG = {
    "judges": [
        {"id": "cheap", "provider": "deterministic", "priority": 1},
        {"id": "local", "provider": "deterministic", "priority": 2},
        {"id": "off", "provider": "deterministic", "priority": 1, "enabled": False},
    ]
}


def _ids(entries):
    return [e["id"] for e in entries]


def test_default_selection_applies_enabled_and_priority():
    assert _ids(select_judge_entries(_CFG)) == ["cheap", "local"]
    assert _ids(select_judge_entries(_CFG, max_priority=1)) == ["cheap"]


def test_judge_ids_select_exactly_those_in_order_and_override_filters():
    # Naming a judge is explicit: priority and the enabled flag don't apply.
    assert _ids(select_judge_entries(_CFG, max_priority=1, judge_ids=["off", "local"])) == [
        "off", "local",
    ]


def test_unknown_judge_id_is_an_error():
    with pytest.raises(ValueError, match="nope"):
        select_judge_entries(_CFG, judge_ids=["cheap", "nope"])


def test_build_judges_honours_judge_ids():
    judges = build_judges(_CFG, judge_ids=["local"], hardware_profile={})
    assert len(judges) == 1
