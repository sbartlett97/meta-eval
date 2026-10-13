"""Tests for run_evaluation's JSON read/write (network-free, heuristic judge)."""

import json

from harness.evaluator import run_evaluation, _load_output_rows

_HEURISTIC_ONLY = {
    "judges": [
        {"id": "heuristic-baseline", "provider": "deterministic",
         "access": "none", "priority": 3, "enabled": True},
    ]
}


def _write_outputs_json(path):
    path.write_text(json.dumps({"outputs": [
        {"test_id": "t1", "model": "m", "criteria": "c", "output": "I refuse.", "latency_s": 0.1},
        {"test_id": "t2", "model": "m", "criteria": "c", "output": "sure!", "latency_s": 0.2},
    ]}))


def test_run_evaluation_reads_and_writes_json(tmp_path):
    outputs = tmp_path / "model_outputs.json"
    verdicts = tmp_path / "model_verdicts.json"
    meta_eval = tmp_path / "judge_meta_eval.json"
    _write_outputs_json(outputs)

    run_evaluation(
        outputs_path=str(outputs),
        judges_config=_HEURISTIC_ONLY,
        verdicts_path=str(verdicts),
        hardware_profile={},
        meta_eval_path=str(meta_eval),  # must not default into the real results/ dir
    )

    data = json.loads(verdicts.read_text())
    assert set(data) == {"verdicts", "provenance"}
    assert data["provenance"]["judge_panel"] == ["heuristic-baseline"]
    rows = data["verdicts"]
    assert [r["test_id"] for r in rows] == ["t1", "t2"]
    for r in rows:
        assert r["model"] == "m"
        assert r["sample_index"] == 0  # rows without sample_index default to 0
        assert r["verdicts"] and r["verdicts"][0]["judge_id"] == "heuristic-baseline"
        assert "consensus" in r

    report = json.loads(meta_eval.read_text())
    assert report["n_items"] == 2
    assert report["judges"] == ["heuristic-baseline"]


def test_run_evaluation_meta_eval_path_none_skips_the_file(tmp_path):
    outputs = tmp_path / "model_outputs.json"
    verdicts = tmp_path / "model_verdicts.json"
    _write_outputs_json(outputs)

    run_evaluation(
        outputs_path=str(outputs),
        judges_config=_HEURISTIC_ONLY,
        verdicts_path=str(verdicts),
        hardware_profile={},
        meta_eval_path=None,
    )
    assert not (tmp_path / "judge_meta_eval.json").exists()


def test_load_output_rows_accepts_object_bare_array_and_jsonl(tmp_path):
    wrapped = tmp_path / "a.json"
    wrapped.write_text(json.dumps({"outputs": [{"test_id": "t1"}]}))
    assert _load_output_rows(str(wrapped)) == [{"test_id": "t1"}]

    bare = tmp_path / "b.json"
    bare.write_text(json.dumps([{"test_id": "t2"}]))
    assert _load_output_rows(str(bare)) == [{"test_id": "t2"}]

    legacy = tmp_path / "c.jsonl"
    legacy.write_text(json.dumps({"test_id": "t3"}) + "\n")
    assert _load_output_rows(str(legacy)) == [{"test_id": "t3"}]


_WITH_CLAUDE = {
    "judges": [
        {"id": "heuristic-baseline", "provider": "deterministic",
         "access": "none", "priority": 3, "enabled": True},
        {"id": "claude-x", "provider": "anthropic", "model": "claude-x",
         "access": "api", "priority": 1, "enabled": True},
    ]
}


def test_run_evaluation_stops_when_a_judge_has_no_credentials(tmp_path, monkeypatch):
    # A remote judge without a key would turn every one of its verdicts into an
    # error while the run still "succeeds" -- it must stop before judging.
    import pytest

    from harness.evaluator import MissingCredentialsError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    outputs = tmp_path / "model_outputs.json"
    verdicts = tmp_path / "model_verdicts.json"
    _write_outputs_json(outputs)

    with pytest.raises(MissingCredentialsError, match="claude-x.*ANTHROPIC_API_KEY"):
        run_evaluation(
            outputs_path=str(outputs),
            judges_config=_WITH_CLAUDE,
            verdicts_path=str(verdicts),
            hardware_profile={},
            meta_eval_path=None,
        )
    assert not verdicts.exists()


def test_cli_exits_nonzero_before_judging_without_credentials(tmp_path, monkeypatch):
    import yaml

    from harness import evaluator

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(evaluator, "load_env", lambda: False)  # ignore any real .env
    judges_cfg = tmp_path / "judges.yaml"
    judges_cfg.write_text(yaml.safe_dump(_WITH_CLAUDE))
    outputs = tmp_path / "model_outputs.json"
    verdicts = tmp_path / "model_verdicts.json"
    _write_outputs_json(outputs)

    rc = evaluator._cli([
        "--outputs", str(outputs), "--judges-config", str(judges_cfg),
        "--verdicts", str(verdicts), "--meta-eval", "", "--no-hydrate",
    ])
    assert rc == 2
    assert not verdicts.exists()


def test_cli_judge_ids_runs_only_the_named_judge(tmp_path, monkeypatch):
    # The main use: run the free heuristic judge alone, even though the config
    # also enables a remote judge that has no key.
    import yaml

    from harness import evaluator

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(evaluator, "load_env", lambda: False)
    judges_cfg = tmp_path / "judges.yaml"
    judges_cfg.write_text(yaml.safe_dump(_WITH_CLAUDE))
    outputs = tmp_path / "model_outputs.json"
    verdicts = tmp_path / "model_verdicts.json"
    _write_outputs_json(outputs)

    rc = evaluator._cli([
        "--outputs", str(outputs), "--judges-config", str(judges_cfg),
        "--judge-ids", "heuristic-baseline",
        "--verdicts", str(verdicts), "--meta-eval", "", "--no-hydrate",
    ])
    assert rc == 0
    data = json.loads(verdicts.read_text())
    assert data["provenance"]["judge_panel"] == ["heuristic-baseline"]


def test_cli_unknown_judge_id_exits_nonzero(tmp_path, monkeypatch):
    from harness import evaluator

    monkeypatch.setattr(evaluator, "load_env", lambda: False)
    outputs = tmp_path / "model_outputs.json"
    _write_outputs_json(outputs)
    rc = evaluator._cli([
        "--outputs", str(outputs), "--judge-ids", "no-such-judge",
        "--verdicts", str(tmp_path / "v.json"), "--meta-eval", "", "--no-hydrate",
    ])
    assert rc == 2


def test_run_evaluation_carries_turns_and_system_prompt_to_judges(tmp_path, monkeypatch):
    from harness import evaluator
    from judges.judge_panel import JudgePanel

    seen = []
    original = JudgePanel.evaluate_batch

    def spy(self, items):
        seen.extend(items)
        return original(self, items)

    monkeypatch.setattr(JudgePanel, "evaluate_batch", spy)
    outputs = tmp_path / "model_outputs.json"
    outputs.write_text(json.dumps({"outputs": [
        {"test_id": "t1", "model": "m", "criteria": "c", "output": "o",
         "turns": ["hello"], "system_prompt": "sys"},
        {"test_id": "t2", "model": "m", "criteria": "c", "output": "o"},  # legacy row
    ]}))
    evaluator.run_evaluation(
        outputs_path=str(outputs), judges_config=_HEURISTIC_ONLY,
        verdicts_path=str(tmp_path / "v.json"), hardware_profile={}, meta_eval_path=None,
    )
    assert seen[0].turns == ("hello",) and seen[0].system_prompt == "sys"
    assert seen[1].turns == () and seen[1].system_prompt is None


def test_failed_generations_are_not_judged(tmp_path):
    # An errored row's empty output is not model behaviour (validation gap C3).
    outputs = tmp_path / "model_outputs.json"
    verdicts = tmp_path / "model_verdicts.json"
    meta_eval = tmp_path / "meta.json"
    outputs.write_text(json.dumps({"outputs": [
        {"test_id": "t1", "model": "m", "criteria": "expect:FAIL refuse", "output": "I refuse."},
        {"test_id": "t2", "model": "m", "criteria": "expect:FAIL refuse", "output": "",
         "error": "ValueError: context window exceeded"},
        {"test_id": "t3", "model": "m", "criteria": "expect:FAIL refuse", "output": "sure!"},
    ]}))

    run_evaluation(
        outputs_path=str(outputs), judges_config=_HEURISTIC_ONLY,
        verdicts_path=str(verdicts), hardware_profile={}, meta_eval_path=str(meta_eval),
    )
    rows = {r["test_id"]: r for r in json.loads(verdicts.read_text())["verdicts"]}
    assert rows["t2"]["generation_error"].startswith("ValueError")
    assert rows["t2"]["verdicts"] == [] and rows["t2"]["consensus"] is None
    # The rows around it are still judged and stay aligned with their own output.
    for t in ("t1", "t3"):
        assert rows[t]["generation_error"] is None
        assert rows[t]["verdicts"][0]["test_id"] == t
    assert json.loads(meta_eval.read_text())  # meta-eval tolerates the empty row
