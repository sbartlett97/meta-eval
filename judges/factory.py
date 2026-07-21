"""Build judge instances from ``config/judges.yaml`` (PRD v3.1).

Decouples the panel from concrete judge classes so judges can be added/removed
by editing YAML (PRD "Easier scaling -- add/remove judges without code changes").
"""

from __future__ import annotations

from typing import Dict, List, Optional

import yaml

from harness import llamacpp_engine
from judges.base import Judge
from judges.claude_judge import ClaudeJudge
from judges.gpt4_judge import GPT4Judge
from judges.heuristic_judge import HeuristicJudge
from judges.local_llamacpp_judge import LocalLlamaCppJudge


def build_judge(entry: Dict, llamacpp_kwargs: Optional[Dict] = None) -> Judge:
    """Instantiate a single judge from one ``judges.yaml`` entry.

    ``llamacpp_kwargs`` (from the hardware profile) are passed to local llama.cpp
    judges so they load their in-process engine with the right context settings.
    """
    provider = entry.get("provider")
    access = entry.get("access")
    jid = entry["id"]

    if provider == "anthropic":
        return ClaudeJudge(model=entry.get("model", "claude-sonnet-4-6"), judge_id=jid)
    if provider == "openai":
        return GPT4Judge(model=entry.get("model", "gpt-4o"), judge_id=jid)
    if provider == "deterministic":
        return HeuristicJudge()
    if provider == "local" and access == "llamacpp":
        return LocalLlamaCppJudge(
            model_id=entry.get("model", ""),
            judge_id=jid,
            gguf_file=entry.get("gguf_file"),
            engine_kwargs=dict(llamacpp_kwargs or {}),
        )

    raise ValueError(f"Cannot build judge for entry: {entry!r}")


def build_judges(
    config,
    max_priority: int | None = None,
    only_enabled: bool = True,
    hardware_profile=None,
) -> List[Judge]:
    """Build all judges from a config dict or path to ``judges.yaml``.

    Args:
        config: Parsed judges.yaml dict, or a path to it.
        max_priority: If set, skip judges with ``priority`` greater than this
            (used to run cheaper/faster subsets -- PRD "Decision 3").
        only_enabled: Skip judges with ``enabled: false``.
        hardware_profile: Parsed ``config/hardware_profile.yaml`` dict, or a path
            to it. Supplies in-process llama.cpp engine kwargs for local judges.
    """
    cfg = config if isinstance(config, dict) else _load(config)
    profile = (
        hardware_profile if isinstance(hardware_profile, dict) else _maybe_load(hardware_profile)
    )
    llamacpp_kwargs = llamacpp_engine.engine_kwargs_from_profile(profile)
    # Honour the sequential-loading cap for any local llama.cpp judges.
    llamacpp_engine.set_max_resident(llamacpp_engine.max_resident_from_profile(profile))
    judges: List[Judge] = []
    for entry in cfg.get("judges", []):
        if only_enabled and not entry.get("enabled", True):
            continue
        if max_priority is not None and entry.get("priority", 1) > max_priority:
            continue
        judges.append(build_judge(entry, llamacpp_kwargs=llamacpp_kwargs))
    return judges


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
