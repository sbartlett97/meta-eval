"""Tests for the in-process vLLM engine and the HuggingFace hydration step.

Neither vLLM nor huggingface_hub is installed in the test environment (and both
are heavy), so we inject lightweight fakes into ``sys.modules`` to exercise the
wiring without loading real weights or touching the network.
"""

import sys
import types

import pytest

from harness import hydrate
from harness import vllm_engine
from harness.model_loader import VLLMModel
from judges.mistral_local_judge import MistralLocalJudge


# ---------------------------------------------------------------------- #
# Fake vLLM
# ---------------------------------------------------------------------- #
class _FakeCompletion:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeCompletion(text)]


class _FakeLLM:
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.calls = []
        _FakeLLM.instances.append(self)

    def generate(self, prompts, params):
        self.calls.append((list(prompts), params))
        # Echo the prompt so tests can assert it was passed through.
        return [_FakeRequestOutput(f"OUT::{p}") for p in prompts]


class _FakeSamplingParams:
    def __init__(self, max_tokens=None, temperature=None):
        self.max_tokens = max_tokens
        self.temperature = temperature


@pytest.fixture
def fake_vllm(monkeypatch):
    _FakeLLM.instances = []
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    vllm_engine.reset_engines()
    yield mod
    vllm_engine.reset_engines()


# ---------------------------------------------------------------------- #
# Engine
# ---------------------------------------------------------------------- #
def test_engine_is_lazy_until_generate(fake_vllm):
    engine = vllm_engine.get_engine("some/checkpoint", {"dtype": "auto"})
    assert engine.loaded is False
    assert _FakeLLM.instances == []  # constructing the engine loads nothing

    text = engine.generate("hello", max_tokens=8, temperature=0.1)
    assert text == "OUT::hello"
    assert engine.loaded is True
    assert len(_FakeLLM.instances) == 1
    assert _FakeLLM.instances[0].kwargs == {"dtype": "auto"}


def test_get_engine_caches_by_checkpoint(fake_vllm):
    a = vllm_engine.get_engine("repo/x")
    b = vllm_engine.get_engine("repo/x")
    assert a is b
    a.generate("p")
    b.generate("q")
    # Same underlying engine -> a single loaded LLM instance.
    assert len(_FakeLLM.instances) == 1


def test_reset_engines_drops_cache(fake_vllm):
    a = vllm_engine.get_engine("repo/x")
    vllm_engine.reset_engines()
    b = vllm_engine.get_engine("repo/x")
    assert a is not b


def test_sampling_params_forwarded(fake_vllm):
    engine = vllm_engine.get_engine("repo/x")
    engine.generate("p", max_tokens=32, temperature=0.9)
    _prompts, params = _FakeLLM.instances[0].calls[0]
    assert params.max_tokens == 32
    assert params.temperature == pytest.approx(0.9)


def test_missing_vllm_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)  # force ImportError
    vllm_engine.reset_engines()
    engine = vllm_engine.get_engine("repo/x")
    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        engine.generate("p")
    vllm_engine.reset_engines()


def test_engine_kwargs_from_profile_filters_keys():
    profile = {
        "vllm_defaults": {
            "gpu_memory_utilization": 0.7,
            "max_model_len": 4096,
            "extra_args": ["--foo"],  # CLI-only; must be dropped
            "unknown_key": 1,          # not a known engine kwarg
        }
    }
    kwargs = vllm_engine.engine_kwargs_from_profile(profile)
    assert kwargs == {"gpu_memory_utilization": 0.7, "max_model_len": 4096}
    assert vllm_engine.engine_kwargs_from_profile(None) == {}


# ---------------------------------------------------------------------- #
# In-process callers reuse the same engine
# ---------------------------------------------------------------------- #
def test_model_and_judge_share_one_engine(fake_vllm):
    model = VLLMModel(id="m", checkpoint="shared/ckpt")
    judge = MistralLocalJudge(model_id="shared/ckpt", judge_id="j")

    out = model.generate("prompt-1")
    assert out == "OUT::prompt-1"

    # The judge prompt is a formatted template, but it still runs in-process
    # through the same cached engine (one loaded LLM instance total).
    verdict = judge.evaluate("t1", "some model output", "expect:refuse")
    assert verdict.judge_id == "j"
    assert len(_FakeLLM.instances) == 1


# ---------------------------------------------------------------------- #
# Hydration
# ---------------------------------------------------------------------- #
def test_collect_weights_dedups_and_skips_cloud():
    models = {
        "local_models": [
            {"id": "a", "checkpoint": "org/a", "serving": {"engine": "vllm"}},
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
            {"id": "a-judge", "provider": "local", "access": "vllm", "model": "org/a"},  # dup of org/a
            {"id": "llama", "provider": "local", "access": "vllm", "model": "org/llama"},
            {"id": "claude", "provider": "anthropic", "access": "api", "model": "claude"},  # api -> skip
        ]
    }
    repos = sorted(s.repo_id for s in hydrate.collect_weights(models, judges))
    assert repos == ["org/a", "org/ft2", "org/llama"]


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
