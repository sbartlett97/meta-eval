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


def test_top_level_mode_key_is_rejected():
    cfg = {"local_models": [
        {"id": "m", "checkpoint": "org/m", "mode": "completion",
         "serving": {"engine": "llamacpp", "gguf_file": "m.gguf"}},
    ]}
    with pytest.raises(ValueError, match="serving.mode"):
        ModelLoader(cfg, hardware_profile={}).load("m")
