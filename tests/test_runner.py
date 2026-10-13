"""Tests for TestRunner output rows (incl. generation timing)."""

import json

# Aliased so pytest doesn't try to collect the imported ``TestRunner`` as a
# test class (it starts with "Test").
from harness.test_runner import TestRunner as Runner


class _FakeModel:
    id = "m"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
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
        {"id": "t1", "prompt": "p1", "criteria": "c",
         "system_prompt": "Be terse."},
        {"id": "t2", "prompt": "p2", "criteria": "c", "system_prompt": None},
    ]}))
    return str(p)


def _read_outputs(path):
    data = json.loads(path.read_text())
    assert set(data) == {"outputs", "provenance"}
    assert data["provenance"]["suite_path"]  # provenance is always populated
    return data["outputs"]


def test_runner_records_generation_latency(tmp_path):
    out = tmp_path / "outputs.json"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_FakeModel()), results_path=str(out))
    runner.run_tests(["m"])

    rows = _read_outputs(out)
    assert [r["test_id"] for r in rows] == ["t1", "t2"]
    for r in rows:
        assert isinstance(r["latency_s"], (int, float))
        assert r["latency_s"] >= 0
        assert r["error"] is None
        assert r["sample_index"] == 0  # default n_samples=1 -> always sample 0


def test_runner_n_samples_repeats_each_test(tmp_path):
    out = tmp_path / "outputs.json"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_FakeModel()), results_path=str(out))
    runner.run_tests(["m"], n_samples=3)

    rows = _read_outputs(out)
    assert [r["test_id"] for r in rows] == ["t1", "t1", "t1", "t2", "t2", "t2"]
    assert [r["sample_index"] for r in rows if r["test_id"] == "t1"] == [0, 1, 2]

    data = json.loads(out.read_text())
    assert data["provenance"]["n_samples"] == 3
    assert data["provenance"]["models"] == ["m"]


def test_runner_passes_and_records_system_prompt(tmp_path):
    out = tmp_path / "outputs.json"
    model = _FakeModel()
    runner = Runner(_write_suite(tmp_path), _FakeLoader(model), results_path=str(out))
    runner.run_tests(["m"])

    # t1's system prompt reaches the model as a generate kwarg; t2 has none, so
    # it is passed as None (every backend's generate() declares the parameter).
    assert model.calls[0]["system_prompt"] == "Be terse."
    assert model.calls[1]["system_prompt"] is None

    # And it is persisted on the output rows (null for t2).
    rows = _read_outputs(out)
    assert rows[0]["system_prompt"] == "Be terse."
    assert rows[1]["system_prompt"] is None


def test_runner_times_failed_generations_too(tmp_path):
    out = tmp_path / "outputs.json"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_BoomModel()), results_path=str(out))
    runner.run_tests(["boom"])

    rows = _read_outputs(out)
    # A failed generation still records output="" + the error + a latency.
    assert all(r["output"] == "" for r in rows)
    assert all("kaboom" in r["error"] for r in rows)
    assert all("latency_s" in r for r in rows)
