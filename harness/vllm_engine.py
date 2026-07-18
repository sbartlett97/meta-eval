"""In-process vLLM engine (PRD v3.1 overhaul).

Previously the harness talked to local models over HTTP: a ``vllm serve`` process
had to be started separately (see the now-removed ``vllm_server_manager``) and
every generation was a ``requests.post`` to ``localhost:<port>/v1/completions``.

This module replaces that with **in-line vLLM calls**: the checkpoint is loaded
directly in-process via the vLLM Python API (``vllm.LLM``) and ``generate`` runs
inference in the same process — no server to launch, no ports, no HTTP.

One engine is loaded per checkpoint and cached process-wide (:func:`get_engine`),
so a test model and any judge that share a checkpoint reuse a single loaded copy
of the weights instead of paying the memory cost twice. Loading is lazy: an
engine object can be constructed cheaply (no import of vLLM, no weights) and only
touches the GPU/unified memory the first time :meth:`InProcessVLLM.generate`
is called.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.7

# Subset of ``config/hardware_profile.yaml`` -> ``vllm_defaults`` that maps
# directly onto ``vllm.LLM(...)`` constructor kwargs. ``extra_args`` is
# deliberately excluded: those were CLI flags for ``vllm serve`` and have no
# meaning for the in-process engine constructor.
_ENGINE_KWARG_KEYS = (
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "dtype",
    "enable_prefix_caching",
    "quantization",
    "tokenizer",
)

# Process-wide engine cache keyed by checkpoint id.
_ENGINE_CACHE: Dict[str, "InProcessVLLM"] = {}
_CACHE_LOCK = threading.Lock()


def engine_kwargs_from_profile(
    profile: Optional[Dict], max_model_len: Optional[int] = None
) -> Dict:
    """Extract ``vllm.LLM`` constructor kwargs from a hardware-profile dict.

    Args:
        profile: Parsed ``config/hardware_profile.yaml`` dict (or ``None``).
        max_model_len: Optional explicit override for the served context window.

    Returns:
        A dict of kwargs safe to splat into ``vllm.LLM(model=..., **kwargs)``.
    """
    defaults = {}
    if isinstance(profile, dict):
        defaults = profile.get("vllm_defaults", {}) or {}
    kwargs = {key: defaults[key] for key in _ENGINE_KWARG_KEYS if key in defaults}
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return kwargs


def get_engine(model: str, engine_kwargs: Optional[Dict] = None) -> "InProcessVLLM":
    """Return a process-wide cached engine for ``model``, creating it on first use.

    Engines are keyed by checkpoint **only**. Loading the same weights twice would
    blow the memory budget, so if two callers ask for the same checkpoint with
    different ``engine_kwargs`` they share the engine created first (a warning is
    logged noting the ignored kwargs).
    """
    engine_kwargs = dict(engine_kwargs or {})
    with _CACHE_LOCK:
        existing = _ENGINE_CACHE.get(model)
        if existing is not None:
            if engine_kwargs and engine_kwargs != existing.engine_kwargs:
                logger.warning(
                    "Reusing already-loaded vLLM engine for %s; ignoring differing "
                    "engine_kwargs %s (loaded with %s)",
                    model,
                    engine_kwargs,
                    existing.engine_kwargs,
                )
            return existing
        engine = InProcessVLLM(model=model, engine_kwargs=engine_kwargs)
        _ENGINE_CACHE[model] = engine
        return engine


def reset_engines() -> None:
    """Drop all cached engines. Mainly for tests and explicit teardown."""
    with _CACHE_LOCK:
        _ENGINE_CACHE.clear()


@dataclass
class InProcessVLLM:
    """A single in-process vLLM engine wrapping one checkpoint.

    Construct via :func:`get_engine` rather than directly so the process-wide
    cache is honoured. The heavy ``vllm.LLM`` object is built lazily on the first
    :meth:`generate` call.
    """

    model: str
    engine_kwargs: Dict = field(default_factory=dict)
    _llm: object = field(default=None, init=False, repr=False)
    _sampling_cls: object = field(default=None, init=False, repr=False)
    _load_lock: "threading.Lock" = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        with self._load_lock:
            if self._llm is not None:  # re-check under lock
                return
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "vLLM is not installed but in-process serving needs it. "
                    'Install it with `pip install "vllm>=0.3.0"`, or evaluate an '
                    "Ollama / API-hosted model instead."
                ) from exc
            logger.info(
                "Loading vLLM engine in-process: %s (%s)", self.model, self.engine_kwargs
            )
            self._llm = LLM(model=self.model, **self.engine_kwargs)
            self._sampling_cls = SamplingParams

    def generate(self, prompt: str, **kwargs) -> str:
        """Run a single prompt through the in-process engine and return its text."""
        self._ensure_loaded()
        params = self._sampling_cls(
            max_tokens=kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", _DEFAULT_TEMPERATURE),
        )
        # vLLM returns one RequestOutput per prompt; each carries >=1 completion.
        outputs = self._llm.generate([prompt], params)
        return outputs[0].outputs[0].text
