"""Tests for the weight-hydration step.

Hydration downloads through llama-cpp-python (``llamacpp_engine.download_pretrained``),
so the download tests monkeypatch that call to record what would be fetched,
without touching the network or loading weights. The engine-level behaviour of
``download_pretrained`` itself lives in ``test_llamacpp_engine.py``.
"""

import os

import pytest

from harness import hydrate
from harness import llamacpp_engine


# ---------------------------------------------------------------------- #
# collect_weights: dedup + cloud skipping + judge inclusion
# ---------------------------------------------------------------------- #
def test_collect_weights_dedups_and_skips_cloud():
    models = {
        "local_models": [
            {"id": "a", "checkpoint": "org/a", "serving": {"engine": "llamacpp"}},
        ],
        "fine_tuned_models": [
            {"id": "ft", "checkpoint": "org/ft", "provider": "replicate"},  # cloud -> skip
            {"id": "ft2", "checkpoint": "org/ft2"},                          # local -> keep
        ],
        "remote_models": [
            {"id": "gpt", "provider": "openai", "model": "gpt-4o"},          # no checkpoint
        ],
    }
    judges = {
        "judges": [
            {"id": "a-judge", "provider": "local", "access": "llamacpp", "model": "org/a"},  # dup
            {"id": "llama", "provider": "local", "access": "llamacpp", "model": "org/llama"},
            {"id": "claude", "provider": "anthropic", "access": "api", "model": "claude"},  # api -> skip
        ]
    }
    repos = sorted(s.repo_id for s in hydrate.collect_weights(models, judges))
    assert repos == ["org/a", "org/ft2", "org/llama"]


# ---------------------------------------------------------------------- #
# Run-scoping filters
# ---------------------------------------------------------------------- #
_SCOPED_MODELS = {
    "local_models": [
        {"id": "mistral", "checkpoint": "org/mistral", "serving": {"engine": "llamacpp"}},
        {"id": "llama", "checkpoint": "org/llama", "serving": {"engine": "llamacpp"}},
    ],
}
_SCOPED_JUDGES = {
    "judges": [
        {"id": "claude", "provider": "anthropic", "access": "api", "priority": 1},
        {"id": "mistral-local", "provider": "local", "access": "llamacpp",
         "model": "org/mistral", "priority": 2, "enabled": True},
        {"id": "off", "provider": "local", "access": "llamacpp",
         "model": "org/off", "priority": 2, "enabled": False},
    ]
}


def _repos(**kwargs):
    return sorted(s.repo_id for s in hydrate.collect_weights(
        _SCOPED_MODELS, _SCOPED_JUDGES, **kwargs))


def test_scope_model_ids_restricts_models():
    # Only the requested model, plus the enabled local judge.
    assert _repos(model_ids=["mistral"]) == ["org/mistral"]


def test_scope_empty_model_ids_hydrates_no_models():
    # A re-judge run generates nothing: no model weights, judges still included.
    assert _repos(model_ids=[]) == ["org/mistral"]


def test_scope_none_model_ids_includes_all_models():
    assert _repos(model_ids=None) == ["org/llama", "org/mistral"]


def test_scope_include_judges_false_skips_judge_weights():
    assert _repos(model_ids=["mistral"], include_judges=False) == ["org/mistral"]


def test_scope_judge_max_priority_filters_local_judges():
    # priority<=1 excludes the priority-2 local judge -> remote-only run, nothing.
    assert _repos(model_ids=[], judge_max_priority=1) == []


def test_scope_skips_disabled_judges():
    # The disabled "off" judge (org/off) is never hydrated.
    assert "org/off" not in _repos(model_ids=[])


# ---------------------------------------------------------------------- #
# hydrate_weights: download plumbing
# ---------------------------------------------------------------------- #
def test_hydrate_weights_downloads_each_repo(monkeypatch):
    calls = []

    def fake_download(repo_id, filename=None, additional_files=None):
        calls.append({
            "repo_id": repo_id,
            "filename": filename,
            "additional_files": tuple(additional_files or ()),
        })

    monkeypatch.setattr(llamacpp_engine, "download_pretrained", fake_download)

    models = {
        "local_models": [
            {"id": "a", "checkpoint": "org/a",
             "serving": {"engine": "llamacpp", "gguf_file": "a-Q4.gguf",
                         "additional_files": ["a-2.gguf"]}},
        ]
    }
    hydrated = hydrate.hydrate_weights(models_config=models, judges_config={}, token="tok")

    assert hydrated == ["org/a"]
    assert calls == [
        {"repo_id": "org/a", "filename": "a-Q4.gguf", "additional_files": ("a-2.gguf",)}
    ]
    # A provided token is exported so from_pretrained's huggingface_hub sees it.
    assert os.environ.get("HF_TOKEN") == "tok"


def test_hydrate_wraps_download_errors(monkeypatch):
    def boom(repo_id, filename=None, additional_files=None):
        raise Exception("404 Client Error: Entry Not Found")

    monkeypatch.setattr(llamacpp_engine, "download_pretrained", boom)

    models = {
        "local_models": [
            {"id": "q", "checkpoint": "org/q",
             "serving": {"engine": "llamacpp", "gguf_file": "missing.gguf"}}
        ]
    }
    with pytest.raises(RuntimeError, match="Could not hydrate .*missing.gguf"):
        hydrate.hydrate_weights(models_config=models, judges_config={})


def test_hydrate_dry_run_downloads_nothing(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("dry run must not download")

    monkeypatch.setattr(llamacpp_engine, "download_pretrained", boom)
    models = {"local_models": [
        {"id": "a", "checkpoint": "org/a",
         "serving": {"engine": "llamacpp", "gguf_file": "a.gguf"}}]}
    hydrated = hydrate.hydrate_weights(models_config=models, judges_config={}, dry_run=True)
    assert hydrated == ["org/a"]
