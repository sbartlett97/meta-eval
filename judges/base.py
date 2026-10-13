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


class NonRetryableJudgeError(Exception):
    """Raise this (instead of a plain exception) for a failure that retrying
    cannot fix -- a missing API key, an uninstalled SDK, a bad config. The
    retry loop in ``Judge.evaluate`` gives up immediately on this without
    burning ``max_retries * retry_backoff_s`` seconds re-trying something
    that will fail identically every time (docs/critical-assessment.md #2.9:
    the retry budget exists for transient failures, not permanent ones).
    """


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
    #: True when this AMBIGUOUS verdict means "this judge is structurally
    #: incapable of assessing this criterion" (e.g. HeuristicJudge on
    #: factual/reasoning correctness it can't keyword-match), NOT "I read the
    #: content and it's genuinely unclear." Conflating the two would count a
    #: judge's abstention as real disagreement in aggregate_verdicts' vote and
    #: judges.meta_eval's inter-judge kappa/bias stats -- both exclude
    #: abstained verdicts the same way they exclude errored ones.
    abstained: bool = False

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
            "abstained": self.abstained,
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
    #: Additional attempts after a failed ``_evaluate`` call, with linear
    #: backoff between them. 0 (the default) = no retry. A transient network
    #: blip shouldn't be recorded identically to genuine judge uncertainty
    #: (docs/critical-assessment.md #2.9) -- remote judges override this to a
    #: small positive number; local/deterministic judges have nothing to gain
    #: from retrying so they keep the default.
    max_retries: int = 0
    retry_backoff_s: float = 1.0
    #: Environment variable holding this judge's API credential, or None for
    #: judges that need none (local / deterministic). The evaluator checks
    #: ``has_credentials`` up front so a missing key stops the run instead of
    #: silently turning every verdict from this judge into an error.
    credential_env: Optional[str] = None
    #: True for a judge that decides exactly-checkable items (format checks)
    #: by code. When one is in the panel, the other judges skip those items.
    runs_exact_checks: bool = False

    def __init__(self) -> None:
        self.call_log: List[JudgeCallRecord] = []

    def has_credentials(self) -> bool:
        """True if this judge can authenticate (always True when it needs nothing)."""
        return True

    def describe(self) -> dict:
        """Settings that identify this judge, for verdict-file provenance.
        Subclasses add their model id and sampling settings."""
        return {"id": self.id, "provider": self.provider}

    @abstractmethod
    def _evaluate(
        self,
        test_id: str,
        model_output: str,
        criteria: str,
        category: str = "uncategorized",
        metadata: Optional[dict] = None,
    ) -> Verdict:
        """Backend-specific evaluation. Must return a :class:`Verdict`.

        ``category`` (e.g. ``"format-adherence"``) and ``metadata`` (e.g.
        ``{"check": "json", "required_keys": [...]}``) are the suite's own
        per-case fields -- most judges ignore them, but a judge that can run a
        real deterministic check for a category (see ``HeuristicJudge``) needs
        them to dispatch.
        """

    def evaluate(
        self,
        test_id: str,
        model_output: str,
        criteria: str,
        category: str = "uncategorized",
        metadata: Optional[dict] = None,
    ) -> Verdict:
        """Evaluate one model output; never raises -- errors become AMBIGUOUS.

        Retries up to ``max_retries`` times (linear backoff) before giving up.
        A judge that still errors after retries should not abort a whole
        evaluation run, so the failure is captured as an AMBIGUOUS verdict with
        ``error`` set and confidence 0 -- callers that need to tell "the judge
        was genuinely uncertain" apart from "the call failed" should check
        ``error`` (aggregate_verdicts / judges.meta_eval already do).
        """
        start = time.monotonic()
        verdict = None
        last_exc: Optional[Exception] = None
        attempts = self.max_retries + 1
        attempts_made = 0
        for attempt in range(1, attempts + 1):
            attempts_made = attempt
            try:
                verdict = self._evaluate(test_id, model_output, criteria, category, metadata or {})
                last_exc = None
                break
            except NonRetryableJudgeError as exc:
                last_exc = exc
                break  # no point retrying a missing API key / uninstalled SDK
            except Exception as exc:  # noqa: BLE001 -- deliberately broad; see docstring
                last_exc = exc
                if attempt < attempts:
                    logger.warning(
                        "Judge %s failed on %s (attempt %d/%d): %s -- retrying",
                        self.id, test_id, attempt, attempts, exc,
                    )
                    time.sleep(self.retry_backoff_s * attempt)
        if last_exc is not None:
            logger.warning(
                "Judge %s failed on %s after %d attempt(s): %s",
                self.id, test_id, attempts_made, last_exc,
            )
            verdict = Verdict(
                verdict="AMBIGUOUS",
                confidence=0.0,
                judge_id=self.id,
                test_id=test_id,
                error=str(last_exc),
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
    # A missing or unrecognised label is a failed judge call, not a judgement:
    # mapping it to AMBIGUOUS would record it as a real "could not decide"
    # (validation gap C11). Raising here lets Judge.evaluate retry and then
    # record the verdict with ``error`` set.
    raise ValueError(f"Unrecognised verdict label {label!r}")


def _clamp01(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
