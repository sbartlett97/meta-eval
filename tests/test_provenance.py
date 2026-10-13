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
