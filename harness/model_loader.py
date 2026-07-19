"""Model Loader for *test* models (PRD v3.1).

This loads the models being EVALUATED (not the judges). Local vLLM models are
loaded in-process through the shared ``harness.vllm_engine`` engine cache — the
same cache the local *judge* models use, so a shared checkpoint is loaded once.

Design (PRD "Decision 2: Test Model Loading"): a single ``ModelLoader.load()``
returns an object exposing ``generate(prompt) -> str``, regardless of backend.
Three backends are supported so Sam can pick without touching call sites:

    * ``ollama``   -- recommended on Apple Silicon; pre-built quantized models.
    * ``vllm``     -- **in-process** vLLM engine (``harness.vllm_engine``); the
      checkpoint is loaded in-process and inference runs in-line, with no
      separately-started server and no HTTP hop.
    * ``replicate``/``api`` -- cloud fallback for models too large for local.

The recommended default is Ollama (PRD "Option A: Ollama-based (Recommended for
Mac)"). BitsAndBytes runtime quantization is intentionally not used.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

import requests
import yaml

from harness.vllm_engine import engine_kwargs_from_profile, get_engine

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TIMEOUT_S = 120  # test-model generations can be long


class GenerativeModel(Protocol):
    """Everything the TestRunner needs from a loaded model."""

    id: str

    def generate(self, prompt: str, **kwargs) -> str: ...


# ---------------------------------------------------------------------- #
# Backend implementations
# ---------------------------------------------------------------------- #
@dataclass
class OllamaModel:
    """Test model served by the local Ollama daemon (http://localhost:11434)."""

    id: str
    tag: str
    host: str = "http://localhost:11434"
    timeout_s: int = _DEFAULT_TIMEOUT_S

    def generate(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.tag,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
                    "temperature": kwargs.get("temperature", _DEFAULT_TEMPERATURE),
                },
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


@dataclass
class VLLMModel:
    """Test model served by an **in-process** vLLM engine (no separate server).

    The checkpoint is loaded in-process via :mod:`harness.vllm_engine` and shared
    process-wide, so ``generate`` is an in-line call — not an HTTP request to a
    ``vllm serve`` process. ``engine_kwargs`` come from the hardware profile.
    """

    id: str
    checkpoint: str
    engine_kwargs: Dict = field(default_factory=dict)

    def generate(self, prompt: str, **kwargs) -> str:
        engine = get_engine(self.checkpoint, self.engine_kwargs)
        return engine.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", _DEFAULT_TEMPERATURE),
        )


@dataclass
class ReplicateModel:
    """Cloud fallback for large models (e.g. Llama 3.1 70B). Scaffold only.

    Filling in the exact Replicate polling protocol is left to Sam, since it
    depends on which cloud provider is chosen (PRD leaves this open).
    """

    id: str
    model: str

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - needs creds
        raise NotImplementedError(
            "Cloud (Replicate) generation is a scaffold. Wire up the provider SDK "
            "and API token before use."
        )


# ---------------------------------------------------------------------- #
# Loader
# ---------------------------------------------------------------------- #
class ModelLoader:
    """Resolve a model id from ``config/models.yaml`` and return a loaded model.

    Args:
        models_config: Parsed models.yaml dict, or a path to it.
        prefer_engine: Which local backend to use for local models when the
            config lists more than one option. ``"ollama"`` (default) or
            ``"vllm"``.
        ensure_daemon: If True and using Ollama, attempt to start the daemon.
        hardware_profile: Parsed ``config/hardware_profile.yaml`` dict, or a path
            to it. Supplies the in-process vLLM engine kwargs (``gpu_memory_
            utilization``, ``max_model_len``, ...). Optional; sensible vLLM
            defaults apply when omitted.
    """

    def __init__(
        self,
        models_config,
        prefer_engine: str = "ollama",
        ensure_daemon: bool = True,
        hardware_profile=None,
    ) -> None:
        self.config = _as_dict(models_config)
        self.prefer_engine = prefer_engine
        self.ensure_daemon = ensure_daemon
        self._engine_kwargs = engine_kwargs_from_profile(_as_dict(hardware_profile))
        self._cache: Dict[str, GenerativeModel] = {}
        self._index = self._build_index()

    def load(self, model_id: str) -> GenerativeModel:
        """Load (and cache) the model with the given ``id``."""
        if model_id in self._cache:
            return self._cache[model_id]

        entry = self._index.get(model_id)
        if entry is None:
            raise KeyError(
                f"Unknown model id {model_id!r}. Known ids: {sorted(self._index)}"
            )

        model = self._instantiate(model_id, entry)
        self._cache[model_id] = model
        return model

    # ------------------------------------------------------------------ #
    def _instantiate(self, model_id: str, entry: Dict) -> GenerativeModel:
        provider = entry.get("provider")
        # Cloud-hosted models under evaluation.
        if provider in {"replicate", "anthropic", "openai"}:
            return ReplicateModel(id=model_id, model=entry.get("model", model_id))

        serving = entry.get("serving", {})
        engine = self.prefer_engine if serving else "ollama"
        if serving.get("engine") and self.prefer_engine not in serving:
            engine = serving["engine"]

        if engine == "ollama":
            tag = serving.get("ollama_tag") or model_id
            if self.ensure_daemon:
                _ensure_ollama_daemon()
                _ollama_pull(tag)
            return OllamaModel(id=model_id, tag=tag)

        if engine == "vllm":
            return VLLMModel(
                id=model_id,
                checkpoint=entry["checkpoint"],
                engine_kwargs=dict(self._engine_kwargs),
            )

        raise ValueError(f"Unsupported serving engine {engine!r} for {model_id!r}")

    def _build_index(self) -> Dict[str, Dict]:
        index: Dict[str, Dict] = {}
        for group in ("local_models", "fine_tuned_models", "remote_models"):
            for entry in self.config.get(group, []) or []:
                index[entry["id"]] = entry
        return index


# ---------------------------------------------------------------------- #
# Ollama daemon helpers
# ---------------------------------------------------------------------- #
def _ensure_ollama_daemon(host: str = "http://localhost:11434", timeout_s: float = 15) -> None:
    """Start `ollama serve` in the background if the daemon isn't responding."""
    if _ollama_up(host):
        return
    logger.info("Ollama daemon not responding; attempting to start it")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`ollama` not found. Install it (`brew install ollama`) or switch "
            "prefer_engine='vllm'."
        ) from exc

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _ollama_up(host):
            return
        time.sleep(1.0)
    raise RuntimeError("Ollama daemon did not become ready in time")


def _ollama_up(host: str) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def _ollama_pull(tag: str) -> None:
    """Pull an Ollama model if not already present (idempotent, streams progress)."""
    logger.info("Ensuring Ollama model is available: %s", tag)
    try:
        subprocess.run(["ollama", "pull", tag], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`ollama pull {tag}` failed") from exc


def _as_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        with open(value, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}
