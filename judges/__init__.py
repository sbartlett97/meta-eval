"""Judges package: unified HTTP/API judge interface (PRD v3.1).

Every judge implements the same ``Judge.evaluate(...) -> Verdict`` contract,
whether it calls a hosted API (Claude/GPT-4), a local vLLM server, or runs a
deterministic heuristic in-process.
"""

from judges.base import Judge, Verdict, JudgeCallRecord
from judges.claude_judge import ClaudeJudge
from judges.gpt4_judge import GPT4Judge
from judges.heuristic_judge import HeuristicJudge
from judges.llama_local_judge import LlamaLocalJudge
from judges.mistral_local_judge import MistralLocalJudge

__all__ = [
    "Judge",
    "Verdict",
    "JudgeCallRecord",
    "ClaudeJudge",
    "GPT4Judge",
    "HeuristicJudge",
    "LlamaLocalJudge",
    "MistralLocalJudge",
]
