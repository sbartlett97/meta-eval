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


# ---------------------------------------------------------------------- #
# Checkpointing / resume (validation gap R2)
# ---------------------------------------------------------------------- #
class _CrashAfter:
    """Returns outputs for the first `n` calls, then simulates the process
    dying (KeyboardInterrupt is not caught by _run_one's `except Exception`)."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.calls > self.n:
            raise KeyboardInterrupt
        return "first-run:" + prompt


class _CountingLoader(_FakeLoader):
    def __init__(self, model):
        super().__init__(model)
        self.loads = []

    def load(self, model_id):
        self.loads.append(model_id)
        return self._model


def test_successful_run_removes_the_checkpoint(tmp_path):
    out = tmp_path / "outputs.json"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_FakeModel()), results_path=str(out))
    runner.run_tests(["m"])
    assert len(_read_outputs(out)) == 2
    assert not (tmp_path / "outputs.json.partial.jsonl").exists()


def test_crash_then_resume_keeps_finished_rows_and_generates_the_rest(tmp_path):
    import pytest

    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Runner(suite, _FakeLoader(_CrashAfter(1)), results_path=str(out)).run_tests(["m"])
    assert not out.exists()  # no final file from a crashed run

    model = _FakeModel()
    Runner(suite, _FakeLoader(model), results_path=str(out)).run_tests(["m"], resume=True)

    rows = _read_outputs(out)
    assert [r["test_id"] for r in rows] == ["t1", "t2"]
    assert rows[0]["output"] == "first-run:p1"  # kept from the checkpoint
    assert rows[1]["output"] == "out:p2"  # generated on resume
    assert [c["prompt"] for c in model.calls] == ["p2"]  # t1 not regenerated


def test_resume_reruns_errored_rows(tmp_path):
    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    runner = Runner(suite, _FakeLoader(_BoomModel()), results_path=str(out))
    runner.run_tests(["m"])
    # Re-create a checkpoint from the errored rows, as if the run had died.
    checkpoint = tmp_path / "outputs.json.partial.jsonl"
    rows = _read_outputs(out)
    header = {"header": {"suite_sha256": json.loads(out.read_text())["provenance"]["suite_sha256"],
                         "n_samples": 1}}
    checkpoint.write_text("\n".join(json.dumps(x) for x in [header, *rows]) + "\n")

    model = _FakeModel()
    Runner(suite, _FakeLoader(model), results_path=str(out)).run_tests(["m"], resume=True)
    assert len(model.calls) == 2  # both errored rows re-run
    assert all(r["error"] is None for r in _read_outputs(out))


def test_resume_skips_loading_a_model_whose_rows_are_all_done(tmp_path):
    import pytest

    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    # Model "a" finishes; the crash happens on model "b".
    crash = _CrashAfter(2)
    with pytest.raises(KeyboardInterrupt):
        Runner(suite, _FakeLoader(crash), results_path=str(out)).run_tests(["a", "b"])

    loader = _CountingLoader(_FakeModel())
    Runner(suite, loader, results_path=str(out)).run_tests(["a", "b"], resume=True)
    assert loader.loads == ["b"]
    assert [(r["model"], r["test_id"]) for r in _read_outputs(out)] == [
        ("a", "t1"), ("a", "t2"), ("b", "t1"), ("b", "t2"),
    ]


def test_fresh_run_refuses_to_discard_an_interrupted_checkpoint(tmp_path):
    import pytest

    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Runner(suite, _FakeLoader(_CrashAfter(1)), results_path=str(out)).run_tests(["m"])
    with pytest.raises(FileExistsError, match="--resume"):
        Runner(suite, _FakeLoader(_FakeModel()), results_path=str(out)).run_tests(["m"])


def test_fresh_run_refuses_to_overwrite_results_unless_asked(tmp_path):
    import pytest

    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    Runner(suite, _FakeLoader(_FakeModel()), results_path=str(out)).run_tests(["m"])
    with pytest.raises(FileExistsError, match="--overwrite"):
        Runner(suite, _FakeLoader(_FakeModel()), results_path=str(out)).run_tests(["m"])
    Runner(suite, _FakeLoader(_FakeModel()), results_path=str(out)).run_tests(["m"], overwrite=True)
    assert len(_read_outputs(out)) == 2


def test_resume_refuses_a_checkpoint_from_a_different_configuration(tmp_path):
    import pytest

    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Runner(suite, _FakeLoader(_CrashAfter(1)), results_path=str(out)).run_tests(["m"])
    with pytest.raises(ValueError, match="n_samples"):
        Runner(suite, _FakeLoader(_FakeModel()), results_path=str(out)).run_tests(
            ["m"], n_samples=3, resume=True
        )


def test_resume_ignores_a_truncated_last_line(tmp_path):
    import pytest

    out = tmp_path / "outputs.json"
    suite = _write_suite(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Runner(suite, _FakeLoader(_CrashAfter(1)), results_path=str(out)).run_tests(["m"])
    checkpoint = tmp_path / "outputs.json.partial.jsonl"
    checkpoint.write_text(checkpoint.read_text() + '{"test_id": "t2", "mod')  # crash mid-write

    Runner(suite, _FakeLoader(_FakeModel()), results_path=str(out)).run_tests(["m"], resume=True)
    assert [r["test_id"] for r in _read_outputs(out)] == ["t1", "t2"]


def test_rows_record_what_the_model_was_asked(tmp_path):
    # Judges need the request, not only the response (validation gap C1).
    out = tmp_path / "outputs.json"
    runner = Runner(_write_suite(tmp_path), _FakeLoader(_FakeModel()), results_path=str(out))
    runner.run_tests(["m"])
    rows = _read_outputs(out)
    assert rows[0]["turns"] == ["p1"]
    assert rows[0]["system_prompt"] == "Be terse."
    assert rows[1]["turns"] == ["p2"]
