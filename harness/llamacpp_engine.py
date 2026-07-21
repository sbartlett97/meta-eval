"""In-process llama.cpp engine (GGUF-native, sequential loading).

The local serving backend for the eval harness. Three properties matter for this
workload:

1. **Point at a specific GGUF file.** ``llama.cpp`` is GGUF-native: an engine is
   pinned to *one* quant file, selected by a filename glob
   (e.g. ``"*Q4_K_M.gguf"``). ``Llama.from_pretrained(repo_id, filename=glob)``
   downloads and loads only that file from a HuggingFace repo, rather than
   grabbing the whole checkpoint directory of quants.
2. **Load models sequentially.** This cache holds at most ``max_resident`` models
   in memory at a time. When loading a new model would exceed the cap, the
   least-recently-used engine is unloaded first — its weights are freed via
   ``Llama.close()``. With the default cap of ``1`` you get strict sequential
   loading: only one model is ever resident, so a test model and two judges never
   fight for memory simultaneously.
3. **In-process, no server.** Construct an engine cheaply (no import of
   ``llama_cpp``, no weights), and only touch memory on the first
   :meth:`LlamaCppEngine.generate` call.

An evicted engine keeps its identity in the cache; it simply reloads the next
time it is used. A test model and a judge that reference the same
``(repo_id, filename)`` reuse one engine object (and one loaded copy of the
weights while resident).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.7

# How many models may be resident at once. 1 => strictly sequential loading:
# loading a new model frees the previously-loaded one first. Raise it (via the
# hardware profile) to keep e.g. both local judges resident when memory allows.
_DEFAULT_MAX_RESIDENT = 1

# Subset of ``config/hardware_profile.yaml`` -> ``llamacpp_defaults`` that maps
# onto ``llama_cpp.Llama(...)`` constructor kwargs. Anything not listed here is
# ignored, so the profile can carry harness-only knobs (e.g. ``max_resident``,
# which is a serving policy, not a Llama kwarg) without breaking construction.
_ENGINE_KWARG_KEYS = (
    "n_ctx",
    "n_gpu_layers",
    "n_batch",
    "n_threads",
    "n_threads_batch",
    "seed",
    "rope_freq_base",
    "rope_freq_scale",
    "chat_format",
    "verbose",
)

# Process-wide engine cache. Insertion / access order is LRU: the *front* is the
# least-recently-used engine and the first evicted when the resident cap is hit.
_ENGINE_CACHE: "OrderedDict[str, LlamaCppEngine]" = OrderedDict()
_CACHE_LOCK = threading.RLock()
_max_resident = _DEFAULT_MAX_RESIDENT


def engine_kwargs_from_profile(
    profile: Optional[Dict], n_ctx: Optional[int] = None
) -> Dict:
    """Extract ``llama_cpp.Llama`` constructor kwargs from a hardware profile.

    Args:
        profile: Parsed ``config/hardware_profile.yaml`` dict (or ``None``).
        n_ctx: Optional explicit override for the served context window.

    Returns:
        A dict of kwargs safe to splat into ``Llama(model_path=..., **kwargs)``.
    """
    defaults: Dict = {}
    if isinstance(profile, dict):
        defaults = profile.get("llamacpp_defaults", {}) or {}
    kwargs = {key: defaults[key] for key in _ENGINE_KWARG_KEYS if key in defaults}
    if n_ctx is not None:
        kwargs["n_ctx"] = n_ctx
    return kwargs


def max_resident_from_profile(profile: Optional[Dict]) -> int:
    """Read the resident-model cap from a hardware profile (default ``1``)."""
    if isinstance(profile, dict):
        section = profile.get("llamacpp", {}) or {}
        value = section.get("max_resident")
        if value is not None:
            return max(1, int(value))
    return _DEFAULT_MAX_RESIDENT


def set_max_resident(n: int) -> None:
    """Set how many models may be resident at once (process-wide).

    Lowering the cap does not evict anything immediately; excess engines are
    unloaded lazily the next time a load would exceed the new cap.
    """
    global _max_resident
    with _CACHE_LOCK:
        _max_resident = max(1, int(n))


def get_engine(
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    model_path: Optional[str] = None,
    additional_files: Optional[Sequence[str]] = None,
    engine_kwargs: Optional[Dict] = None,
) -> "LlamaCppEngine":
    """Return a process-wide cached engine, creating it on first use.

    Provide either ``model_path`` (a local ``.gguf`` file) or ``repo_id`` plus a
    ``filename`` naming the GGUF in that HuggingFace repo. As in llama.cpp's
    ``Llama.from_pretrained`` snippet, ``filename`` is normally the **exact** GGUF
    file (e.g. ``"model-BF16.gguf"``); a glob (``"*Q4_K_M.gguf"``) is also
    accepted. ``additional_files`` names extra parts to fetch alongside it (e.g.
    the remaining shards of a split GGUF), matching ``from_pretrained``'s
    parameter of the same name.

    Engines are keyed by ``(repo_id, filename, additional_files)`` / ``model_path``,
    so two callers that reference the same GGUF share one engine (and one loaded
    copy of the weights while it is resident).
    """
    if not repo_id and not model_path:
        raise ValueError("get_engine needs either repo_id or model_path")
    additional_files = list(additional_files or [])
    engine_kwargs = dict(engine_kwargs or {})
    key = _make_key(repo_id, filename, model_path, additional_files)
    with _CACHE_LOCK:
        existing = _ENGINE_CACHE.get(key)
        if existing is not None:
            if engine_kwargs and engine_kwargs != existing.engine_kwargs:
                logger.warning(
                    "Reusing already-registered llama.cpp engine for %s; ignoring "
                    "differing engine_kwargs %s (registered with %s)",
                    key,
                    engine_kwargs,
                    existing.engine_kwargs,
                )
            return existing
        engine = LlamaCppEngine(
            repo_id=repo_id,
            filename=filename,
            model_path=model_path,
            additional_files=additional_files,
            engine_kwargs=engine_kwargs,
            key=key,
        )
        _ENGINE_CACHE[key] = engine
        return engine


def reset_engines() -> None:
    """Unload and drop all cached engines. Mainly for tests / explicit teardown."""
    with _CACHE_LOCK:
        for engine in _ENGINE_CACHE.values():
            engine._unload()
        _ENGINE_CACHE.clear()


def resident_engines() -> "list[LlamaCppEngine]":
    """The engines currently holding weights in memory (for tests / diagnostics)."""
    with _CACHE_LOCK:
        return [e for e in _ENGINE_CACHE.values() if e.loaded]


def _import_llama_cls():
    """Import ``llama_cpp.Llama`` with a clear install message (lazy)."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "llama-cpp-python is not installed but the llamacpp backend needs "
            "it. Install it with `pip install llama-cpp-python` (see its docs "
            "for a Metal/CUDA-accelerated build), or serve this model via "
            "Ollama / an API instead."
        ) from exc
    return Llama


def download_pretrained(
    repo_id: str,
    filename: str,
    additional_files: Optional[Sequence[str]] = None,
) -> None:
    """Download a GGUF (and any ``additional_files``) to the local cache.

    Uses the same ``Llama.from_pretrained`` path that :meth:`LlamaCppEngine.generate`
    uses to load, so hydration and serving fetch bit-for-bit the same file(s)
    through llama-cpp-python — no separate ``huggingface_hub`` download call. The
    model is loaded ``vocab_only`` (just the vocabulary, negligible memory) and
    immediately closed: the point is to land the file on disk, not to serve it.
    ``from_pretrained`` raises if the file/repo can't be found, so a wrong name
    fails loudly here at hydration time.
    """
    if not filename:
        raise RuntimeError(
            f"llama.cpp needs a GGUF filename to fetch {repo_id!r}. Set the exact "
            "GGUF file name via `gguf_file` (models: `serving.gguf_file`, judges: "
            "`gguf_file`) — the value you'd pass to from_pretrained(filename=...)."
        )
    Llama = _import_llama_cls()
    kwargs = {"vocab_only": True, "verbose": False}
    if additional_files:
        kwargs["additional_files"] = list(additional_files)
    logger.info("Hydrating GGUF via llama.cpp: %s :: %s", repo_id, filename)
    llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, **kwargs)
    close = getattr(llm, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning("Error closing hydration handle for %s: %s", repo_id, exc)


def _make_key(
    repo_id: Optional[str],
    filename: Optional[str],
    model_path: Optional[str],
    additional_files: Optional[Sequence[str]] = None,
) -> str:
    if model_path:
        return f"path::{model_path}"
    extra = "|".join(additional_files or [])
    return f"hf::{repo_id}::{filename or ''}::{extra}"


def _evict_to_fit(incoming: "LlamaCppEngine") -> None:
    """Unload LRU engines until loading ``incoming`` stays within the cap.

    Must be called while holding ``_CACHE_LOCK``.
    """
    loaded = [
        e for e in _ENGINE_CACHE.values() if e.loaded and e is not incoming
    ]  # dict order == LRU order (oldest first)
    while len(loaded) >= _max_resident and loaded:
        victim = loaded.pop(0)
        logger.info(
            "llama.cpp resident cap (%d) reached; unloading %s to load %s",
            _max_resident,
            victim.key,
            incoming.key,
        )
        victim._unload()


@dataclass
class LlamaCppEngine:
    """A single in-process llama.cpp engine wrapping one GGUF file.

    Construct via :func:`get_engine` so the process-wide cache + resident cap are
    honoured. The heavy ``llama_cpp.Llama`` object is built lazily on the first
    :meth:`generate` call, and may be transparently unloaded/reloaded as other
    engines are used (see the module docstring on sequential loading).
    """

    repo_id: Optional[str] = None
    filename: Optional[str] = None
    model_path: Optional[str] = None
    additional_files: List[str] = field(default_factory=list)
    engine_kwargs: Dict = field(default_factory=dict)
    key: str = ""
    _llm: object = field(default=None, init=False, repr=False)

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        with _CACHE_LOCK:
            if self._llm is not None:  # re-check under lock
                return
            _evict_to_fit(self)
            self._llm = self._construct()
            _ENGINE_CACHE.move_to_end(self.key)  # most-recently used

    def _construct(self) -> object:
        Llama = _import_llama_cls()
        if self.model_path:
            logger.info("Loading GGUF in-process (llama.cpp): %s", self.model_path)
            return Llama(model_path=self.model_path, **self.engine_kwargs)
        if not self.filename:
            raise RuntimeError(
                f"llama.cpp needs a GGUF filename to load {self.repo_id!r} from "
                "HuggingFace. Set the exact GGUF file name via `gguf_file` "
                "(models: `serving.gguf_file`, judges: `gguf_file`), e.g. "
                "'model-Q4_K_M.gguf' — the same value you'd pass to "
                "Llama.from_pretrained(filename=...)."
            )
        logger.info(
            "Loading GGUF in-process (llama.cpp): %s :: %s", self.repo_id, self.filename
        )
        from_pretrained_kwargs = dict(self.engine_kwargs)
        if self.additional_files:
            from_pretrained_kwargs["additional_files"] = list(self.additional_files)
        return Llama.from_pretrained(
            repo_id=self.repo_id, filename=self.filename, **from_pretrained_kwargs
        )

    def _unload(self) -> None:
        """Free the loaded weights, if any. The engine can reload on next use.

        Called while holding ``_CACHE_LOCK`` (eviction path) or from
        :func:`reset_engines`.
        """
        llm = self._llm
        self._llm = None
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logger.warning("Error closing llama.cpp engine %s: %s", self.key, exc)

    def generate(self, prompt: str, **kwargs) -> str:
        """Run a single prompt through the in-process engine and return its text."""
        self._ensure_loaded()
        with _CACHE_LOCK:
            _ENGINE_CACHE.move_to_end(self.key)  # touch: most-recently used
        result = self._llm.create_completion(
            prompt,
            max_tokens=kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", _DEFAULT_TEMPERATURE),
        )
        return result["choices"][0]["text"]
