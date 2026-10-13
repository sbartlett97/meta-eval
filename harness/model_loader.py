"""Model Loader for *test* models (PRD v3.1).

This loads the models being EVALUATED (not the judges). Local GGUF models are
loaded in-process through the shared ``harness.llamacpp_engine`` engine cache —
the same cache the local *judge* models use, so a shared GGUF is loaded once.

Design (PRD "Decision 2: Test Model Loading"): a single ``ModelLoader.load()``
returns an object exposing ``generate(prompt) -> str``, regardless of backend.
Three backends are supported so Sam can pick without touching call sites:

    * ``llamacpp`` -- **in-process** llama.cpp engine (``harness.llamacpp_engine``).
      The default local backend. GGUF-native: points at a *specific* quant file
      inside an HF repo and loads models sequentially (one resident at a time by
      default).
    * ``ollama``   -- local Ollama daemon over HTTP; pre-built quantized models.
    * ``replicate``/``api`` -- cloud fallback for models too large for local.

BitsAndBytes runtime quantization is intentionally not used.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import requests
import yaml

from harness import llamacpp_engine

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TIMEOUT_S = 120  # test-model generations can be long


class GenerativeModel(Protocol):
    """Everything the TestRunner needs from a loaded model."""

    id: str

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str: ...

    def generate_conversation(
        self, turns: List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Run ``turns`` as a real back-and-forth: turn *n* sees the model's own
        reply to turn *n-1* in context, not just the raw text of turn *n-1*.
        Returns one reply per turn, same length/order as ``turns``.
        """
        ...


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

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        payload = {
            "model": self.tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
                "temperature": kwargs.get("temperature", _DEFAULT_TEMPERATURE),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        resp = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout_s
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def generate_conversation(
        self, turns: List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Real multi-turn via Ollama's ``/api/chat`` (accumulates message history)."""
        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        options = {
            "num_predict": kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get("temperature", _DEFAULT_TEMPERATURE),
        }
        replies: List[str] = []
        for turn in turns:
            messages.append({"role": "user", "content": turn})
            resp = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.tag,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            reply = resp.json().get("message", {}).get("content", "")
            messages.append({"role": "assistant", "content": reply})
            replies.append(reply)
        return replies


@dataclass
class LlamaCppModel:
    """Test model served by an **in-process** llama.cpp (GGUF) engine.

    Pinned to one GGUF quant (``gguf_file`` glob) inside ``repo_id`` and loaded
    in-process via :mod:`harness.llamacpp_engine`, which loads models sequentially
    so a test model and the judges never all hold weights at once. ``engine_kwargs``
    come from the hardware profile's ``llamacpp_defaults``.

    ``mode`` controls how prompts are sent (``serving.mode`` in models.yaml):

    * ``"chat"`` (default) — the prompt goes through the GGUF's embedded chat
      template via ``create_chat_completion``. This is what instruction-tuned
      checkpoints (``-it`` / ``-instruct``) need; a raw completion makes them
      *continue* the prompt instead of answering it.
    * ``"completion"`` — raw text completion, for base (non-instruct) checkpoints.
    """

    id: str
    repo_id: str
    filename: Optional[str] = None
    additional_files: List[str] = field(default_factory=list)
    engine_kwargs: Dict = field(default_factory=dict)
    mode: str = "chat"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        engine = llamacpp_engine.get_engine(
            repo_id=self.repo_id,
            filename=self.filename,
            additional_files=self.additional_files,
            engine_kwargs=self.engine_kwargs,
        )
        call = engine.generate if self.mode == "completion" else engine.chat
        return call(
            prompt,
            system_prompt=system_prompt,
            max_tokens=kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", _DEFAULT_TEMPERATURE),
        )

    def generate_conversation(
        self, turns: List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> List[str]:
        engine = llamacpp_engine.get_engine(
            repo_id=self.repo_id,
            filename=self.filename,
            additional_files=self.additional_files,
            engine_kwargs=self.engine_kwargs,
        )
        call = engine.generate_turns if self.mode == "completion" else engine.chat_turns
        return call(
            turns,
            system_prompt=system_prompt,
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

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:  # pragma: no cover - needs creds
        raise NotImplementedError(
            "Cloud (Replicate) generation is a scaffold. Wire up the provider SDK "
            "and API token before use."
        )

    def generate_conversation(
        self, turns: List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> List[str]:  # pragma: no cover - needs creds
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
        prefer_engine: Local backend to use when a model entry does not name one
            in ``serving.engine``. ``"llamacpp"`` (default) or ``"ollama"``.
        ensure_daemon: If True and using Ollama, attempt to start the daemon.
        hardware_profile: Parsed ``config/hardware_profile.yaml`` dict, or a path
            to it. Supplies the in-process llama.cpp engine kwargs (``n_ctx`` /
            ``n_gpu_layers`` ...) and the ``max_resident`` cap. Optional; sensible
            defaults apply when omitted.
    """

    def __init__(
        self,
        models_config,
        prefer_engine: str = "llamacpp",
        ensure_daemon: bool = True,
        hardware_profile=None,
    ) -> None:
        self.config = _as_dict(models_config)
        self.prefer_engine = prefer_engine
        self.ensure_daemon = ensure_daemon
        profile = _as_dict(hardware_profile)
        self._llamacpp_kwargs = llamacpp_engine.engine_kwargs_from_profile(profile)
        # Apply the resident-model cap so llama.cpp loads sequentially.
        llamacpp_engine.set_max_resident(
            llamacpp_engine.max_resident_from_profile(profile)
        )
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

        # The model entry names its engine in ``serving.engine``; the loader's
        # ``prefer_engine`` is only the fallback when it doesn't.
        serving = entry.get("serving", {})
        engine = serving.get("engine") or self.prefer_engine

        if engine == "llamacpp":
            mode = serving.get("mode", "chat")
            if mode not in ("chat", "completion"):
                raise ValueError(
                    f"Unsupported serving.mode {mode!r} for {model_id!r}: "
                    "expected 'chat' (default; applies the chat template) or "
                    "'completion' (raw text completion for base checkpoints)."
                )
            return LlamaCppModel(
                id=model_id,
                repo_id=entry["checkpoint"],
                filename=serving.get("gguf_file"),
                additional_files=list(serving.get("additional_files") or []),
                engine_kwargs=dict(self._llamacpp_kwargs),
                mode=mode,
            )

        if engine == "ollama":
            tag = serving.get("ollama_tag") or model_id
            if self.ensure_daemon:
                _ensure_ollama_daemon()
                _ollama_pull(tag)
            return OllamaModel(id=model_id, tag=tag)

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
            "prefer_engine='llamacpp'."
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
