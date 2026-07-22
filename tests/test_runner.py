"""Tests for TestRunner output rows (incl. generation timing)."""

import json

# Aliased so pytest doesn't try to collect the imported ``TestRunner`` as a
# test class (it starts with "Test").
from harness.test_runner import TestRunner as Runner


class _FakeModel:
    id = "m"

    def generate(self, prompt, **kwargs):
        return "out:" + prompt


class _BoomModel:
    id = "boom"

    def generate(self, prompt, **kwargs):
        raise RuntimeError("kaboom")


class _FakeLoader:
    def __init__(self, model):
        self._model = model

    def load(self, model_id):
        return self._model


def _write_suite(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text(json.dumps({"evals": [
        {"id": "t1", "prompt": "p1", "criteria": "c"},
        {"id": "t2", "prompt": "p2", "criteria": "c"},
    ]}))
    return str(p)


def test_runner_records_generation_latency(tmp_path):
    out = tmp_path / "outputs.jsonl"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_FakeModel()), results_path=str(out))
    runner.run_tests(["m"])

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["test_id"] for r in rows] == ["t1", "t2"]
    for r in rows:
        assert isinstance(r["latency_s"], (int, float))
        assert r["latency_s"] >= 0
        assert r["error"] is None


def test_runner_times_failed_generations_too(tmp_path):
    out = tmp_path / "outputs.jsonl"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_BoomModel()), results_path=str(out))
    runner.run_tests(["boom"])

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    # A failed generation still records output="" + the error + a latency.
    assert all(r["output"] == "" for r in rows)
    assert all("kaboom" in r["error"] for r in rows)
    assert all("latency_s" in r for r in rows)
