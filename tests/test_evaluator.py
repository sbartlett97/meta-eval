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
