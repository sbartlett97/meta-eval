"""Tests for the in-process llama.cpp (GGUF) engine and its wiring.

llama-cpp-python is not installed in the test environment (and is heavy), so we
inject a lightweight fake ``llama_cpp`` module into ``sys.modules`` to exercise
the engine, the model-loader/judge wiring, and the sequential-loading cap without
loading real weights or touching the network.
"""

import sys
import types

import pytest

from harness import hydrate
from harness import llamacpp_engine
from harness.model_loader import LlamaCppModel
from judges.factory import build_judge
from judges.local_llamacpp_judge import LocalLlamaCppJudge


# ---------------------------------------------------------------------- #
# Fake llama_cpp
# ---------------------------------------------------------------------- #
class _FakeLlama:
    instances = []

    def __init__(self, model_path=None, **kwargs):
        self.model_path = model_path
        self.repo_id = kwargs.pop("_repo_id", None)
        self.filename = kwargs.pop("_filename", None)
        self.kwargs = kwargs
        self.closed = False
        self.calls = []
        self.chat_calls = []
        _FakeLlama.instances.append(self)

    @classmethod
    def from_pretrained(cls, repo_id=None, filename=None, **kwargs):
        return cls(model_path=None, _repo_id=repo_id, _filename=filename, **kwargs)

    def create_completion(self, prompt, max_tokens=None, temperature=None):
        self.calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        )
        return {"choices": [{"text": f"OUT::{prompt}"}]}

    def create_chat_completion(self, messages, max_tokens=None, temperature=None,
                               response_format=None):
        self.chat_calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
        })
        if response_format is not None:  # JSON-constrained -> a valid verdict
            content = '{"verdict": "PASS", "confidence": 0.9, "rationale": "ok"}'
        else:  # echo, so tests can assert the prompt went through the chat path
            content = f"CHAT::{messages[-1]['content']}"
        return {"choices": [{"message": {"content": content}}]}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    _FakeLlama.instances = []
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    llamacpp_engine.reset_engines()
    llamacpp_engine.set_max_resident(1)
    yield mod
    llamacpp_engine.reset_engines()
    llamacpp_engine.set_max_resident(1)


# ---------------------------------------------------------------------- #
# Engine basics
# ---------------------------------------------------------------------- #
def test_engine_is_lazy_until_generate(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(repo_id="org/repo", filename="*Q4.gguf")
    assert engine.loaded is False
    assert _FakeLlama.instances == []  # constructing the engine loads nothing

    text = engine.generate("hello", max_tokens=8, temperature=0.1)
    assert text == "OUT::hello"
    assert engine.loaded is True
    assert len(_FakeLlama.instances) == 1


def test_from_pretrained_receives_repo_and_filename(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(repo_id="org/repo", filename="*Q4_K_M.gguf")
    engine.generate("p")
    llm = _FakeLlama.instances[0]
    assert llm.repo_id == "org/repo"
    assert llm.filename == "*Q4_K_M.gguf"  # the specific GGUF is selected
    assert llm.model_path is None


def test_local_model_path_bypasses_hub(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(model_path="/models/m.gguf")
    engine.generate("p")
    llm = _FakeLlama.instances[0]
    assert llm.model_path == "/models/m.gguf"
    assert llm.repo_id is None


def test_exact_filename_passthrough(fake_llama_cpp):
    # The HF snippet form: an exact GGUF file name, not a glob.
    engine = llamacpp_engine.get_engine(
        repo_id="empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",
        filename="Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf",
    )
    engine.generate("p")
    llm = _FakeLlama.instances[0]
    assert llm.filename == "Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf"


def test_additional_files_forwarded(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(
        repo_id="org/repo",
        filename="model-00001-of-00002.gguf",
        additional_files=["model-00002-of-00002.gguf"],
    )
    engine.generate("p")
    llm = _FakeLlama.instances[0]
    assert llm.kwargs.get("additional_files") == ["model-00002-of-00002.gguf"]


def test_missing_filename_raises_clear_error(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(repo_id="org/repo")  # no filename
    with pytest.raises(RuntimeError, match="needs a GGUF filename"):
        engine.generate("p")


def test_sampling_params_forwarded(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(repo_id="org/repo", filename="m.gguf")
    engine.generate("p", max_tokens=32, temperature=0.9)
    call = _FakeLlama.instances[0].calls[0]
    assert call["max_tokens"] == 32
    assert call["temperature"] == pytest.approx(0.9)


def test_chat_uses_chat_completion_and_json_grammar(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(repo_id="org/repo", filename="m.gguf")
    text = engine.chat("judge this", json_only=True, max_tokens=64, temperature=0.0)
    assert '"verdict"' in text

    llm = _FakeLlama.instances[0]
    assert llm.calls == []  # chat path, not raw completion
    (call,) = llm.chat_calls
    assert call["messages"] == [{"role": "user", "content": "judge this"}]
    assert call["response_format"] == {"type": "json_object"}  # JSON-constrained
    assert call["temperature"] == 0.0


def test_chat_without_json_only_has_no_response_format(fake_llama_cpp):
    engine = llamacpp_engine.get_engine(repo_id="org/repo", filename="m.gguf")
    engine.chat("hello")
    (call,) = _FakeLlama.instances[0].chat_calls
    assert call["response_format"] is None


def test_get_engine_caches_by_ref(fake_llama_cpp):
    a = llamacpp_engine.get_engine(repo_id="org/x", filename="*Q4.gguf")
    b = llamacpp_engine.get_engine(repo_id="org/x", filename="*Q4.gguf")
    assert a is b
    a.generate("p")
    b.generate("q")
    assert len(_FakeLlama.instances) == 1  # one loaded copy of the weights


def test_different_gguf_files_are_distinct_engines(fake_llama_cpp):
    llamacpp_engine.set_max_resident(2)
    a = llamacpp_engine.get_engine(repo_id="org/x", filename="*Q4.gguf")
    b = llamacpp_engine.get_engine(repo_id="org/x", filename="*Q8.gguf")
    assert a is not b


def test_get_engine_requires_a_source(fake_llama_cpp):
    with pytest.raises(ValueError, match="repo_id or model_path"):
        llamacpp_engine.get_engine()


def test_missing_llama_cpp_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # force ImportError
    llamacpp_engine.reset_engines()
    engine = llamacpp_engine.get_engine(repo_id="org/x")
    with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
        engine.generate("p")
    llamacpp_engine.reset_engines()


# ---------------------------------------------------------------------- #
# Sequential loading (the headline feature)
# ---------------------------------------------------------------------- #
def test_max_resident_one_unloads_previous(fake_llama_cpp):
    llamacpp_engine.set_max_resident(1)
    a = llamacpp_engine.get_engine(repo_id="org/a", filename="a.gguf")
    b = llamacpp_engine.get_engine(repo_id="org/b", filename="b.gguf")

    a.generate("p")
    assert a.loaded is True
    # Loading b must free a first: only one model resident at a time.
    b.generate("q")
    assert b.loaded is True
    assert a.loaded is False
    assert _FakeLlama.instances[0].closed is True  # a's weights were freed
    assert llamacpp_engine.resident_engines() == [b]


def test_evicted_engine_reloads_on_next_use(fake_llama_cpp):
    llamacpp_engine.set_max_resident(1)
    a = llamacpp_engine.get_engine(repo_id="org/a", filename="a.gguf")
    b = llamacpp_engine.get_engine(repo_id="org/b", filename="b.gguf")

    a.generate("p")
    b.generate("q")  # evicts a
    assert a.loaded is False
    text = a.generate("again")  # a transparently reloads
    assert text == "OUT::again"
    assert a.loaded is True
    assert b.loaded is False  # ... which in turn evicts b
    # a was loaded twice (once initially, once after reload) -> two instances.
    assert sum(1 for i in _FakeLlama.instances if i.repo_id == "org/a") == 2


def test_max_resident_two_keeps_both(fake_llama_cpp):
    llamacpp_engine.set_max_resident(2)
    a = llamacpp_engine.get_engine(repo_id="org/a", filename="a.gguf")
    b = llamacpp_engine.get_engine(repo_id="org/b", filename="b.gguf")
    a.generate("p")
    b.generate("q")
    assert a.loaded and b.loaded  # both fit under the cap
    assert len(llamacpp_engine.resident_engines()) == 2


# ---------------------------------------------------------------------- #
# Hardware-profile plumbing
# ---------------------------------------------------------------------- #
def test_engine_kwargs_from_profile_filters_keys():
    profile = {
        "llamacpp_defaults": {
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "unknown_key": 1,  # not a known Llama kwarg -> dropped
        }
    }
    kwargs = llamacpp_engine.engine_kwargs_from_profile(profile)
    assert kwargs == {"n_ctx": 4096, "n_gpu_layers": -1}
    assert llamacpp_engine.engine_kwargs_from_profile(None) == {}


def test_max_resident_from_profile():
    assert llamacpp_engine.max_resident_from_profile({"llamacpp": {"max_resident": 2}}) == 2
    assert llamacpp_engine.max_resident_from_profile({}) == 1
    assert llamacpp_engine.max_resident_from_profile(None) == 1


# ---------------------------------------------------------------------- #
# Model-loader + judge wiring share one engine
# ---------------------------------------------------------------------- #
def test_model_and_judge_share_one_engine(fake_llama_cpp):
    model = LlamaCppModel(id="m", repo_id="shared/repo", filename="*Q4.gguf")
    judge = LocalLlamaCppJudge(
        model_id="shared/repo", judge_id="j", gguf_file="*Q4.gguf"
    )

    # Default mode is chat: the prompt goes through the chat template.
    assert model.generate("prompt-1") == "CHAT::prompt-1"
    verdict = judge.evaluate("t1", "some model output", "expect:refuse")
    assert verdict.judge_id == "j"
    assert len(_FakeLlama.instances) == 1  # same cached engine, one load


def test_model_mode_completion_uses_raw_generate(fake_llama_cpp):
    model = LlamaCppModel(
        id="m", repo_id="base/repo", filename="m.gguf", mode="completion"
    )
    assert model.generate("prompt-1") == "OUT::prompt-1"  # raw completion path
    llm = _FakeLlama.instances[0]
    assert llm.chat_calls == []


def test_model_chat_mode_never_json_constrains(fake_llama_cpp):
    # Test models chat freely -- only the judge uses the JSON grammar.
    model = LlamaCppModel(id="m", repo_id="it/repo", filename="m.gguf")
    model.generate("hello")
    (call,) = _FakeLlama.instances[0].chat_calls
    assert call["response_format"] is None


def test_judge_uses_chat_template_with_json_grammar(fake_llama_cpp):
    # The judge must go through the chat template (instruct GGUFs continue raw
    # prompts instead of answering) with JSON-constrained decoding at temp 0.
    judge = LocalLlamaCppJudge(model_id="org/j", judge_id="j", gguf_file="j.gguf")
    verdict = judge.evaluate("t1", "output", "expect:refuse")

    assert verdict.verdict == "PASS"  # fake's JSON parsed cleanly
    assert verdict.error is None
    llm = _FakeLlama.instances[0]
    assert llm.calls == []  # never a raw completion
    (call,) = llm.chat_calls
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0  # deterministic, like the remote judges


def test_batch_judging_loads_each_local_judge_once(fake_llama_cpp):
    # The reason the panel judges judge-outer: with the resident cap at 1, a
    # naive row-outer loop would reload each judge every row. evaluate_batch keeps
    # each judge resident for the whole batch -> one load per judge, not per row.
    from judges.judge_panel import EvalItem, JudgePanel

    llamacpp_engine.set_max_resident(1)
    j1 = LocalLlamaCppJudge(model_id="org/a", judge_id="j1", gguf_file="*Q4.gguf")
    j2 = LocalLlamaCppJudge(model_id="org/b", judge_id="j2", gguf_file="*Q4.gguf")
    panel = JudgePanel([j1, j2])

    items = [EvalItem(f"t{i}", "out", "crit") for i in range(3)]
    results = panel.evaluate_batch(items)

    assert len(results) == 3
    # 2 judges x 1 load each == 2, not 2 judges x 3 items == 6.
    assert len(_FakeLlama.instances) == 2


def test_factory_builds_llamacpp_judge():
    entry = {
        "id": "mistral-7b-local",
        "provider": "local",
        "access": "llamacpp",
        "model": "unsloth/Mistral-7B-v0.3-GGUF",
        "gguf_file": "*Q4_K_M.gguf",
    }
    judge = build_judge(entry, llamacpp_kwargs={"n_ctx": 2048})
    assert isinstance(judge, LocalLlamaCppJudge)
    assert judge.id == "mistral-7b-local"
    assert judge.gguf_file == "*Q4_K_M.gguf"
    assert judge.engine_kwargs == {"n_ctx": 2048}


def test_factory_accepts_model_style_judge_entry():
    # A judge defined exactly like a model under test (checkpoint + serving)
    # must build the same LocalLlamaCppJudge.
    entry = {
        "id": "gemma-4-12b",
        "checkpoint": "unsloth/gemma-4-12b-it-GGUF",
        "quantization": "4-bit",
        "size_gb": 7.12,
        "context_tokens": 32768,
        "serving": {
            "engine": "llamacpp",
            "gguf_file": "*Q4_K_M.gguf",
            "additional_files": ["mmproj-F16.gguf"],
        },
    }
    judge = build_judge(entry, llamacpp_kwargs={"n_ctx": 2048})
    assert isinstance(judge, LocalLlamaCppJudge)
    assert judge.id == "gemma-4-12b"
    assert judge.model_id == "unsloth/gemma-4-12b-it-GGUF"
    assert judge.gguf_file == "*Q4_K_M.gguf"
    assert judge.additional_files == ["mmproj-F16.gguf"]
    assert judge.engine_kwargs == {"n_ctx": 2048}


def test_model_loader_reads_serving_mode():
    from harness.model_loader import ModelLoader

    cfg = {"local_models": [
        {"id": "it-model", "checkpoint": "org/it",
         "serving": {"engine": "llamacpp", "gguf_file": "m.gguf"}},
        {"id": "base-model", "checkpoint": "org/base",
         "serving": {"engine": "llamacpp", "gguf_file": "m.gguf",
                     "mode": "completion"}},
    ]}
    loader = ModelLoader(cfg)
    assert loader.load("it-model").mode == "chat"  # default
    assert loader.load("base-model").mode == "completion"


def test_model_loader_rejects_unknown_mode():
    from harness.model_loader import ModelLoader

    cfg = {"local_models": [
        {"id": "typo", "checkpoint": "org/x",
         "serving": {"engine": "llamacpp", "gguf_file": "m.gguf", "mode": "cht"}},
    ]}
    with pytest.raises(ValueError, match="serving.mode"):
        ModelLoader(cfg).load("typo")


def test_factory_unknown_entry_error_mentions_both_shapes():
    with pytest.raises(ValueError, match="serving.engine"):
        build_judge({"id": "mystery", "provider": "wat"})


# ---------------------------------------------------------------------- #
# download_pretrained: hydration goes through llama-cpp-python
# ---------------------------------------------------------------------- #
def test_download_pretrained_uses_from_pretrained_vocab_only(fake_llama_cpp):
    llamacpp_engine.download_pretrained(
        repo_id="org/repo",
        filename="model-BF16.gguf",
        additional_files=["shard-2.gguf"],
    )
    assert len(_FakeLlama.instances) == 1
    llm = _FakeLlama.instances[0]
    assert llm.repo_id == "org/repo"
    assert llm.filename == "model-BF16.gguf"
    assert llm.kwargs.get("vocab_only") is True  # downloads to disk, no full load
    assert llm.kwargs.get("additional_files") == ["shard-2.gguf"]
    assert llm.closed is True  # not kept resident


def test_download_pretrained_requires_filename(fake_llama_cpp):
    with pytest.raises(RuntimeError, match="needs a GGUF filename"):
        llamacpp_engine.download_pretrained(repo_id="org/repo", filename=None)


def test_download_pretrained_missing_llama_cpp_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # force ImportError
    with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
        llamacpp_engine.download_pretrained(repo_id="org/repo", filename="m.gguf")


# ---------------------------------------------------------------------- #
# Hydration scoping for GGUF entries
# ---------------------------------------------------------------------- #
def test_collect_weights_includes_llamacpp_judges_and_targets_gguf():
    models = {
        "local_models": [
            {
                "id": "mistral",
                "checkpoint": "org/mistral",
                "serving": {"engine": "llamacpp", "gguf_file": "*Q4_K_M.gguf"},
            },
        ],
    }
    judges = {
        "judges": [
            {
                "id": "llama-local",
                "provider": "local",
                "access": "llamacpp",
                "model": "org/llama",
                "gguf_file": "*Q4_K_M.gguf",
            },
        ]
    }
    specs = {s.repo_id: s for s in hydrate.collect_weights(models, judges)}
    assert set(specs) == {"org/mistral", "org/llama"}
    # gguf_file becomes the from_pretrained filename (here a glob).
    assert specs["org/mistral"].filename == "*Q4_K_M.gguf"
    assert specs["org/llama"].filename == "*Q4_K_M.gguf"


def test_collect_weights_accepts_model_style_judge_entry():
    # Judges defined with the models.yaml shape hydrate too.
    judges = {
        "judges": [
            {
                "id": "gemma-4-12b",
                "checkpoint": "unsloth/gemma-4-12b-it-GGUF",
                "serving": {"engine": "llamacpp", "gguf_file": "*Q4_K_M.gguf"},
            },
        ]
    }
    (spec,) = hydrate.collect_weights({}, judges)
    assert spec.repo_id == "unsloth/gemma-4-12b-it-GGUF"
    assert spec.filename == "*Q4_K_M.gguf"


def test_exact_gguf_file_and_additional_files_are_captured():
    models = {
        "local_models": [
            {
                "id": "q",
                "checkpoint": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",
                "serving": {
                    "engine": "llamacpp",
                    "gguf_file": "Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf",
                    "additional_files": ["mmproj-F16.gguf"],
                },
            },
        ],
    }
    (spec,) = hydrate.collect_weights(models, {})
    assert spec.filename == "Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf"
    assert spec.additional_files == ("mmproj-F16.gguf",)
