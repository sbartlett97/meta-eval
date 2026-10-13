"""Checks on the real config/models.yaml (network-free; nothing is loaded).

These load the shipped config through ModelLoader, so a config-shape mistake
that the code would silently ignore fails here instead of in a long run.
"""

import pytest
import yaml

from harness.model_loader import LlamaCppModel, ModelLoader

_CONFIG = "config/models.yaml"
# Raw (non-instruct) base checkpoints: no chat template, so they must run in
# completion mode. Keep in sync with config/models.yaml.
_COMPLETION_MODELS = {"llama-3-8b", "qwen2.5-coder-7b"}
# Models whose replies start with a thinking trace (seen in the 2026-08 run).
_REASONING_MODELS = {
    "Ornith-1.0-9B", "qwythos-9b", "Qwen3.5-9B-The-Defiant-Fable-Uncensored", "Qwopus3.5-9B-Coder",
}


def _llamacpp_ids():
    cfg = yaml.safe_load(open(_CONFIG))
    return [
        e["id"]
        for group in ("local_models", "fine_tuned_models")
        for e in cfg.get(group) or []
        if (e.get("serving") or {}).get("engine") == "llamacpp"
    ]


@pytest.mark.parametrize("model_id", _llamacpp_ids())
def test_shipped_models_load_with_the_intended_mode(model_id):
    model = ModelLoader(_CONFIG, hardware_profile={}).load(model_id)
    assert isinstance(model, LlamaCppModel)
    expected = "completion" if model_id in _COMPLETION_MODELS else "chat"
    assert model.mode == expected
    assert model.reasoning == (model_id in _REASONING_MODELS)
    if model.reasoning:
        assert model.max_tokens >= 4096  # 512 ends most replies mid-trace


def test_top_level_mode_key_is_rejected():
    cfg = {"local_models": [
        {"id": "m", "checkpoint": "org/m", "mode": "completion",
         "serving": {"engine": "llamacpp", "gguf_file": "m.gguf"}},
    ]}
    with pytest.raises(ValueError, match="serving.mode"):
        ModelLoader(cfg, hardware_profile={}).load("m")


def test_ollama_model_records_done_reason(monkeypatch):
    # Network-free: requests.post is replaced with a canned Ollama response.
    from harness import model_loader

    class _Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    bodies = iter([
        {"response": "hi", "done_reason": "stop"},
        {"message": {"content": "a"}, "done_reason": "stop"},
        {"message": {"content": "b"}, "done_reason": "length"},
    ])
    monkeypatch.setattr(model_loader.requests, "post", lambda *a, **k: _Resp(next(bodies)))

    model = model_loader.OllamaModel(id="m", tag="m")
    assert model.generate("q") == "hi" and model.last_finish_reasons == ["stop"]
    assert model.generate_conversation(["q1", "q2"]) == ["a", "b"]
    assert model.last_finish_reasons == ["stop", "length"]
