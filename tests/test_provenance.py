"""Tests for harness/provenance.py (docs/critical-assessment.md #10).

git_commit() is exercised against this actual repo checkout rather than mocked
-- it's a local, network-free subprocess call, consistent with how the rest of
the suite treats local filesystem/subprocess operations as fair game (only
network calls and real model loads are avoided).
"""

from harness import provenance


def test_git_commit_returns_a_hash_in_this_checkout():
    commit = provenance.git_commit()
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))


def test_git_commit_returns_none_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert provenance.git_commit() is None


def test_file_sha256_is_stable_and_content_dependent(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    h1 = provenance.file_sha256(str(p))
    assert h1 == provenance.file_sha256(str(p))  # stable across calls

    p.write_text("hello world")
    h2 = provenance.file_sha256(str(p))
    assert h1 != h2  # content-dependent


def test_file_sha256_missing_file_returns_none():
    assert provenance.file_sha256("/nonexistent/path/does/not/exist.json") is None


def test_collect_always_has_timestamp_and_commit():
    data = provenance.collect()
    assert "generated_at" in data
    assert "harness_git_commit" in data
    assert "suite_path" not in data  # only present when suite_path is given


def test_collect_with_suite_path_adds_hash(tmp_path):
    p = tmp_path / "suite.json"
    p.write_text('{"evals": []}')
    data = provenance.collect(suite_path=str(p))
    assert data["suite_path"] == str(p)
    assert data["suite_sha256"] is not None


def test_collect_passes_through_extra_fields():
    data = provenance.collect(models=["a", "b"], n_samples=3)
    assert data["models"] == ["a", "b"]
    assert data["n_samples"] == 3


# ---------------------------------------------------------------------- #
# Extended provenance (validation gap W3)
# ---------------------------------------------------------------------- #
def test_collect_records_dirty_flag_and_environment():
    p = provenance.collect()
    assert p["harness_git_dirty"] in (True, False, None)
    env = p["environment"]
    assert env["python"].count(".") >= 1
    assert "llama_cpp_python" in env["packages"] and "anthropic" in env["packages"]


def test_git_dirty_is_none_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert provenance.git_dirty() is None


def test_weights_info_reads_the_sha256_from_the_hf_cache_blob(tmp_path):
    # The HF cache layout: snapshots/<rev>/<file> -> ../../blobs/<sha256>.
    sha = "a" * 64
    blob = tmp_path / "blobs" / sha
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "m-Q4_K_M.gguf").symlink_to(blob)

    info = provenance.weights_info(str(snap / "m-Q4_K_M.gguf"))
    assert info == {"file": "m-Q4_K_M.gguf", "size_bytes": 7, "sha256": sha}


def test_weights_info_outside_the_cache_has_no_hash(tmp_path):
    f = tmp_path / "local.gguf"
    f.write_bytes(b"x")
    assert provenance.weights_info(str(f))["sha256"] is None
    assert provenance.weights_info(None) is None


def test_outputs_provenance_records_model_details_and_sampling(tmp_path):
    import json

    from harness.test_runner import TestRunner

    class _Model:
        def generate(self, prompt, **kwargs):
            return "ok"

        def describe(self):
            return {"backend": "fake", "n_ctx": 8192}

    class _Loader:
        def load(self, model_id):
            return _Model()

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"evals": [{"id": "t", "prompt": "p", "criteria": "c"}]}))
    out = tmp_path / "out.json"
    TestRunner(str(suite), _Loader(), results_path=str(out)).run_tests(
        ["m"], gen_kwargs={"temperature": 0.5}, seed=3
    )
    prov = json.loads(out.read_text())["provenance"]
    assert prov["model_details"] == {"m": {"backend": "fake", "n_ctx": 8192}}
    assert prov["gen_kwargs"] == {"temperature": 0.5} and prov["seed"] == 3


def test_verdict_provenance_carries_judges_rubric_hash_and_outputs_provenance(tmp_path):
    import json

    from harness.evaluator import run_evaluation

    outputs = tmp_path / "o.json"
    outputs.write_text(json.dumps({
        "outputs": [{"test_id": "t", "model": "m", "criteria": "c", "output": "x",
                     "sample_index": 0, "turns": ["p"]}],
        "provenance": {"suite_sha256": "abc"},
    }))
    verdicts = tmp_path / "v.json"
    run_evaluation(
        outputs_path=str(outputs),
        judges_config={"judges": [{"id": "heuristic-baseline", "provider": "deterministic"}]},
        verdicts_path=str(verdicts), hardware_profile={}, meta_eval_path=None,
    )
    prov = json.loads(verdicts.read_text())["provenance"]
    assert prov["outputs_provenance"] == {"suite_sha256": "abc"}
    assert prov["judge_details"] == [
        {"id": "heuristic-baseline", "provider": "deterministic", "refusal_votes": False}
    ]
    from judges import rubric
    assert prov["rubric_version"] == rubric.RUBRIC_VERSION
    assert prov["rubric_sha256"] == rubric.rubric_sha256()
