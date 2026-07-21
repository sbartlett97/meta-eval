"""Tests for the HuggingFace weight-hydration step.

huggingface_hub is not installed in the test environment (and is heavy), so the
download tests inject a lightweight fake into ``sys.modules`` to exercise the
wiring without touching the network. The llama.cpp-specific hydration scoping
(``gguf_file`` -> ``allow_patterns``) lives in ``test_llamacpp_engine.py``.
"""

import sys
import types

import pytest

from harness import hydrate


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

    def fake_snapshot_download(repo_id, allow_patterns=None, token=None):
        calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns, "token": token})
        return f"/cache/{repo_id}"

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    models = {"local_models": [{"id": "a", "checkpoint": "org/a",
                                "hf_allow_patterns": ["*.gguf"]}]}
    hydrated = hydrate.hydrate_weights(models_config=models, judges_config={}, token="tok")

    assert hydrated == ["org/a"]
    assert calls == [{"repo_id": "org/a", "allow_patterns": ["*.gguf"], "token": "tok"}]


def test_hydrate_exact_filename_uses_hf_hub_download(monkeypatch):
    calls = []

    def fake_hf_hub_download(repo_id, filename=None, token=None):
        calls.append({"repo_id": repo_id, "filename": filename, "token": token})
        return f"/cache/{repo_id}/{filename}"

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    models = {
        "local_models": [
            {
                "id": "q",
                "checkpoint": "org/q",
                "serving": {
                    "engine": "llamacpp",
                    "gguf_file": "model-BF16.gguf",
                    "additional_files": ["shard-2.gguf"],
                },
            }
        ]
    }
    hydrated = hydrate.hydrate_weights(models_config=models, judges_config={}, token="tok")

    assert hydrated == ["org/q"]
    # The exact file and its additional shard are each fetched by name.
    assert calls == [
        {"repo_id": "org/q", "filename": "model-BF16.gguf", "token": "tok"},
        {"repo_id": "org/q", "filename": "shard-2.gguf", "token": "tok"},
    ]


def test_hydrate_missing_exact_file_raises_clear_error(monkeypatch):
    def fake_hf_hub_download(repo_id, filename=None, token=None):
        raise Exception("404 Client Error: Entry Not Found")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    models = {
        "local_models": [
            {"id": "q", "checkpoint": "org/q",
             "serving": {"engine": "llamacpp", "gguf_file": "missing.gguf"}}
        ]
    }
    with pytest.raises(RuntimeError, match="Could not fetch 'missing.gguf'"):
        hydrate.hydrate_weights(models_config=models, judges_config={})


def test_hydrate_dry_run_downloads_nothing(monkeypatch):
    # Even without huggingface_hub importable, a dry run must not try to import it.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    models = {"local_models": [{"id": "a", "checkpoint": "org/a"}]}
    hydrated = hydrate.hydrate_weights(models_config=models, judges_config={}, dry_run=True)
    assert hydrated == ["org/a"]


def test_hydrate_missing_hub_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # force ImportError
    models = {"local_models": [{"id": "a", "checkpoint": "org/a"}]}
    with pytest.raises(RuntimeError, match="huggingface_hub is required"):
        hydrate.hydrate_weights(models_config=models, judges_config={})
