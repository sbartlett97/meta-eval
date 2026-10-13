"""Build judge instances from ``config/judges.yaml`` (PRD v3.1).

Decouples the panel from concrete judge classes so judges can be added/removed
by editing YAML (PRD "Easier scaling -- add/remove judges without code changes").
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import yaml

from harness import llamacpp_engine
from judges.base import Judge
from judges.claude_judge import ClaudeJudge
from judges.gpt4_judge import GPT4Judge
from judges.heuristic_judge import HeuristicJudge
from judges.local_llamacpp_judge import LocalLlamaCppJudge


def build_judge(entry: Dict, llamacpp_kwargs: Optional[Dict] = None) -> Judge:
    """Instantiate a single judge from one ``judges.yaml`` entry.

    Local llama.cpp judges accept **the same entry shape as models under test**
    (``checkpoint`` + ``serving.engine: llamacpp`` + ``serving.gguf_file``) as
    well as the flat judge shape (``access: llamacpp`` + ``model`` +
    ``gguf_file``), so an entry can be copied verbatim between models.yaml and
    judges.yaml.

    ``llamacpp_kwargs`` (from the hardware profile) are passed to local llama.cpp
    judges so they load their in-process engine with the right context settings.
    """
    provider = entry.get("provider")
    jid = entry["id"]

    if provider == "anthropic":
        return ClaudeJudge(model=entry.get("model", "claude-sonnet-4-6"), judge_id=jid)
    if provider == "openai":
        return GPT4Judge(model=entry.get("model", "gpt-4o"), judge_id=jid)
    if provider == "deterministic":
        return HeuristicJudge()

    spec = local_llamacpp_spec(entry)
    if spec is not None:
        repo, gguf_file, additional_files = spec
        return LocalLlamaCppJudge(
            model_id=repo,
            judge_id=jid,
            gguf_file=gguf_file,
            additional_files=additional_files,
            engine_kwargs=dict(llamacpp_kwargs or {}),
        )

    raise ValueError(
        f"Cannot build judge for entry: {entry!r}. Expected provider "
        "anthropic/openai/deterministic, or a local llama.cpp judge — either "
        "the model-style shape (checkpoint + serving.engine: llamacpp + "
        "serving.gguf_file) or the flat shape (access: llamacpp + model + "
        "gguf_file)."
    )


def local_llamacpp_spec(entry: Dict) -> Optional[tuple]:
    """Normalize a local llama.cpp entry to ``(repo, gguf_file, additional_files)``.

    Accepts both config shapes — the flat judge shape (``access: llamacpp``,
    ``model``, ``gguf_file``) and the model-style shape (``serving.engine:
    llamacpp``, ``checkpoint``, ``serving.gguf_file``). Returns ``None`` if the
    entry is not a local llama.cpp entry (e.g. an API judge).
    """
    serving = entry.get("serving", {}) or {}
    if entry.get("access") != "llamacpp" and serving.get("engine") != "llamacpp":
        return None
    repo = entry.get("model") or entry.get("checkpoint")
    if not repo:
        return None
    gguf_file = entry.get("gguf_file") or serving.get("gguf_file")
    additional_files = entry.get("additional_files") or serving.get("additional_files")
    return repo, gguf_file, additional_files


def select_judge_entries(
    config,
    max_priority: int | None = None,
    only_enabled: bool = True,
    judge_ids: Optional[Iterable[str]] = None,
) -> List[Dict]:
    """The judges.yaml entries a run uses. Shared by build_judges and hydration
    so both always agree on which judges are in the panel.

    With ``judge_ids``, exactly those entries are returned, in that order, and
    ``max_priority`` / ``only_enabled`` are ignored: naming a judge is an
    explicit request. An unknown id raises ``ValueError``.
    """
    cfg = config if isinstance(config, dict) else _load(config)
    entries = cfg.get("judges", []) or []
    if judge_ids is not None:
        by_id = {e.get("id"): e for e in entries}
        unknown = [j for j in judge_ids if j not in by_id]
        if unknown:
            raise ValueError(
                f"Unknown judge id(s) {unknown}; judges.yaml has {sorted(by_id)}."
            )
        return [by_id[j] for j in judge_ids]
    return [
        e for e in entries
        if (not only_enabled or e.get("enabled", True))
        and (max_priority is None or e.get("priority", 1) <= max_priority)
    ]


def build_judges(
    config,
    max_priority: int | None = None,
    only_enabled: bool = True,
    hardware_profile=None,
    judge_ids: Optional[Iterable[str]] = None,
) -> List[Judge]:
    """Build all judges from a config dict or path to ``judges.yaml``.

    Args:
        config: Parsed judges.yaml dict, or a path to it.
        max_priority: If set, skip judges with ``priority`` greater than this
            (used to run cheaper/faster subsets -- PRD "Decision 3").
        only_enabled: Skip judges with ``enabled: false``.
        hardware_profile: Parsed ``config/hardware_profile.yaml`` dict, or a path
            to it. Supplies in-process llama.cpp engine kwargs for local judges.
        judge_ids: Build exactly these judges (see ``select_judge_entries``).
    """
    profile = (
        hardware_profile if isinstance(hardware_profile, dict) else _maybe_load(hardware_profile)
    )
    # Honour the sequential-loading cap for any local llama.cpp judges.
    llamacpp_engine.set_max_resident(llamacpp_engine.max_resident_from_profile(profile))
    return [
        build_judge(
            entry,
            # Per-judge context window: a judge reads the whole transcript, so
            # it needs room for it (see llamacpp_engine.n_ctx_for).
            llamacpp_kwargs=llamacpp_engine.engine_kwargs_from_profile(
                profile, n_ctx=llamacpp_engine.n_ctx_for(entry, profile)
            ),
        )
        for entry in select_judge_entries(config, max_priority, only_enabled, judge_ids)
    ]


def _load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _maybe_load(value) -> Dict:
    """Load a YAML path into a dict; tolerate ``None`` / a missing file."""
    if not value:
        return {}
    try:
        return _load(value)
    except (OSError, FileNotFoundError):
        return {}
