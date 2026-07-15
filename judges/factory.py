"""Build judge instances from ``config/judges.yaml`` (PRD v3.1).

Decouples the panel from concrete judge classes so judges can be added/removed
by editing YAML (PRD "Easier scaling -- add/remove judges without code changes").
"""

from __future__ import annotations

from typing import Dict, List

import yaml

from judges.base import Judge
from judges.claude_judge import ClaudeJudge
from judges.gpt4_judge import GPT4Judge
from judges.heuristic_judge import HeuristicJudge
from judges.llama_local_judge import LlamaLocalJudge
from judges.mistral_local_judge import MistralLocalJudge


def build_judge(entry: Dict) -> Judge:
    """Instantiate a single judge from one ``judges.yaml`` entry."""
    provider = entry.get("provider")
    access = entry.get("access")
    jid = entry["id"]

    if provider == "anthropic":
        return ClaudeJudge(model=entry.get("model", "claude-sonnet-4-6"), judge_id=jid)
    if provider == "openai":
        return GPT4Judge(model=entry.get("model", "gpt-4o"), judge_id=jid)
    if provider == "deterministic":
        return HeuristicJudge()
    if provider == "local" and access == "vllm":
        port = int(entry.get("vllm_port", 8000))
        model = entry.get("model", "")
        # Route by well-known port; fall back to the generic local base class.
        if port == 8001 or "llama" in jid.lower():
            return LlamaLocalJudge(vllm_port=port, model_id=model, judge_id=jid)
        return MistralLocalJudge(vllm_port=port, model_id=model, judge_id=jid)

    raise ValueError(f"Cannot build judge for entry: {entry!r}")


def build_judges(
    config,
    max_priority: int | None = None,
    only_enabled: bool = True,
) -> List[Judge]:
    """Build all judges from a config dict or path to ``judges.yaml``.

    Args:
        config: Parsed judges.yaml dict, or a path to it.
        max_priority: If set, skip judges with ``priority`` greater than this
            (used to run cheaper/faster subsets -- PRD "Decision 3").
        only_enabled: Skip judges with ``enabled: false``.
    """
    cfg = config if isinstance(config, dict) else _load(config)
    judges: List[Judge] = []
    for entry in cfg.get("judges", []):
        if only_enabled and not entry.get("enabled", True):
            continue
        if max_priority is not None and entry.get("priority", 1) > max_priority:
            continue
        judges.append(build_judge(entry))
    return judges


def _load(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
