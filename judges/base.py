"""Judge base class + verdict types (PRD v3.1).

All judges share one interface so the panel can call Claude, GPT-4, a local
llama.cpp model, or the heuristic baseline through identical code.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Allowed verdict labels. Kept as a module constant so every judge and the
# aggregation layer agree on the vocabulary.
VERDICT_LABELS = ("PASS", "FAIL", "AMBIGUOUS")


@dataclass
class Verdict:
    """A single judge's assessment of one model output."""

    verdict: str  # one of VERDICT_LABELS
    confidence: float  # 0.0 - 1.0
    judge_id: str
    test_id: str
    rationale: str = ""
    # Provenance / cost tracking.
    latency_s: Optional[float] = None
    tokens: Optional[int] = None
    error: Optional[str] = None
    raw: Optional[str] = None  # raw model text, for debugging

    def __post_init__(self) -> None:
        self.verdict = _normalize_label(self.verdict)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "judge_id": self.judge_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "latency_s": self.latency_s,
            "tokens": self.tokens,
            "error": self.error,
        }


@dataclass
class JudgeCallRecord:
    """Lightweight per-call log entry for cost/latency tracking."""

    judge_id: str
    test_id: str
    latency_s: float
    tokens: Optional[int]
    ok: bool


class Judge(ABC):
    """Base class for all judges.

    Subclasses implement ``_evaluate`` (the backend-specific call). The public
    ``evaluate`` wrapper handles timing, error capture, and call-record logging
    so every judge tracks latency/tokens uniformly.
    """

    #: Stable identifier used in configs, results, and aggregation.
    id: str = "base-judge"
    #: Coarse provider tag (anthropic / openai / local / deterministic).
    provider: str = "base"

    def __init__(self) -> None:
        self.call_log: List[JudgeCallRecord] = []

    @abstractmethod
    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        """Backend-specific evaluation. Must return a :class:`Verdict`."""

    def evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        """Evaluate one model output; never raises -- errors become AMBIGUOUS.

        A judge that errors should not abort a whole evaluation run, so failures
        are captured as an AMBIGUOUS verdict with ``error`` set and confidence 0.
        """
        start = time.monotonic()
        try:
            verdict = self._evaluate(test_id, model_output, criteria)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; see docstring
            logger.warning("Judge %s failed on %s: %s", self.id, test_id, exc)
            verdict = Verdict(
                verdict="AMBIGUOUS",
                confidence=0.0,
                judge_id=self.id,
                test_id=test_id,
                error=str(exc),
            )
        if verdict.latency_s is None:
            verdict.latency_s = time.monotonic() - start
        self.call_log.append(
            JudgeCallRecord(
                judge_id=self.id,
                test_id=test_id,
                latency_s=verdict.latency_s,
                tokens=verdict.tokens,
                ok=verdict.error is None,
            )
        )
        return verdict


# ---------------------------------------------------------------------- #
# Helpers shared by verdict construction.
# ---------------------------------------------------------------------- #
def _normalize_label(label: object) -> str:
    text = str(label or "").strip().upper()
    if text in VERDICT_LABELS:
        return text
    # Tolerate common synonyms from free-form model output.
    if text in {"PASSED", "TRUE", "YES", "SAFE", "ALIGNED"}:
        return "PASS"
    if text in {"FAILED", "FALSE", "NO", "UNSAFE", "MISALIGNED"}:
        return "FAIL"
    return "AMBIGUOUS"


def _clamp01(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
