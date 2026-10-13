"""Judge Panel Manager (PRD v3.1).

Runs a panel of judges over model outputs and collects their verdicts. This is
the orchestration layer shown in the architecture diagram ("Judge Panel
Manager"). It does NOT decide the final aggregated verdict -- that (and the
meta-evaluation / inter-judge reliability analysis) is Sam's manual work and
lives in ``analysis/`` (see ``aggregate_verdicts`` stub below).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from judges.base import Judge, Verdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalItem:
    """One (test, model output) pair to judge -- the unit of :meth:`evaluate_batch`."""

    test_id: str
    model_output: str
    criteria: str = ""
    #: Suite category (e.g. "format-adherence"). Lets a judge dispatch on the
    #: kind of check being asked for -- see judges/heuristic_judge.py, which
    #: uses it to run real deterministic checks instead of guessing from prose.
    category: str = "uncategorized"
    #: The test case's free-form ``metadata`` (e.g. ``{"check": "json",
    #: "required_keys": [...]}``). Carries the params a deterministic check
    #: needs -- see judges/format_checks.py -- without a judge having to parse
    #: them back out of the English ``criteria`` prose.
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: The user turn(s) the model answered (one for a single-turn case) and
    #: the system prompt it ran under. A judge that sees only the response
    #: cannot tell whether it fits the request.
    turns: Tuple[str, ...] = ()
    system_prompt: Optional[str] = None


@dataclass
class PanelResult:
    """All judges' verdicts for a single (test, model output) pair."""

    test_id: str
    verdicts: List[Verdict] = field(default_factory=list)

    def by_judge(self) -> Dict[str, Verdict]:
        return {v.judge_id: v for v in self.verdicts}


class JudgePanel:
    """Fan one (test, output) pair out to every judge and gather verdicts."""

    def __init__(self, judges: Sequence[Judge]) -> None:
        if not judges:
            raise ValueError("JudgePanel requires at least one judge")
        self.judges = list(judges)

    def evaluate(
        self,
        test_id: str,
        model_output: str,
        criteria: str,
        category: str = "uncategorized",
        metadata: Dict[str, Any] | None = None,
    ) -> PanelResult:
        """Run every judge on one output. Judge errors are captured, not raised."""
        verdicts = [
            judge.evaluate(test_id, model_output, criteria, category, metadata or {})
            for judge in self.judges
        ]
        return PanelResult(test_id=test_id, verdicts=verdicts)

    def evaluate_batch(self, items: Sequence[EvalItem]) -> List[PanelResult]:
        """Evaluate many outputs **judge-outer, item-inner**.

        Same per-item verdicts as calling :meth:`evaluate` on each item (verdicts
        stay in ``self.judges`` order), but every judge processes the whole batch
        before the next judge runs. That keeps a locally-served judge's weights
        resident for all of its calls instead of reloading per item: with the
        llama.cpp resident cap (see :mod:`harness.llamacpp_engine`) a panel of two
        local judges loads each once per batch, not once per row.
        """
        items = list(items)
        results = [PanelResult(test_id=item.test_id) for item in items]
        for judge in self.judges:
            for item, result in zip(items, results):
                result.verdicts.append(
                    judge.evaluate(
                        item.test_id,
                        item.model_output,
                        item.criteria,
                        item.category,
                        item.metadata,
                    )
                )
        return results

    def cost_summary(self) -> Dict[str, dict]:
        """Aggregate per-judge latency/token stats from call logs."""
        summary: Dict[str, dict] = {}
        for judge in self.judges:
            calls = judge.call_log
            if not calls:
                continue
            summary[judge.id] = {
                "n_calls": len(calls),
                "n_ok": sum(1 for c in calls if c.ok),
                "total_latency_s": round(sum(c.latency_s for c in calls), 3),
                "total_tokens": sum(c.tokens or 0 for c in calls),
            }
        return summary


def aggregate_verdicts(result: PanelResult) -> dict:
    """Per-item majority vote -- a naive placeholder for the real cross-item
    meta-evaluation (inter-judge agreement, bias detection, reliability
    weighting), which lives in judges/meta_eval.py and works over a whole
    batch of these, not one item at a time. Do not treat this as the final
    methodology on its own.

    Two kinds of verdict are excluded from the vote entirely, not just treated
    as an "AMBIGUOUS" ballot:

    * a verdict whose judge call itself failed (``error`` set -- see
      ``Judge.evaluate``) -- counting a transient API failure as a genuine
      "AMBIGUOUS" would silently bias the consensus toward whichever judge
      happened to have a flaky call on this item (docs/critical-assessment.md
      #2.9), and
    * a verdict where the judge structurally couldn't assess the criterion
      (``abstained`` set -- e.g. ``HeuristicJudge`` on factual/reasoning
      correctness it has no keyword signal for). An abstention isn't that
      judge's opinion; counting it as one dilutes what the other judges
      actually decided.
    """
    ok_verdicts = [v for v in result.verdicts if v.error is None and not v.abstained]
    n_errored = sum(1 for v in result.verdicts if v.error is not None)
    n_abstained = sum(1 for v in result.verdicts if v.error is None and v.abstained)

    counts: Dict[str, int] = {}
    conf: Dict[str, float] = {}
    for v in ok_verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        conf[v.verdict] = conf.get(v.verdict, 0.0) + v.confidence
    if not counts:
        return {
            "test_id": result.test_id,
            "consensus": "AMBIGUOUS",
            "agreement": 0.0,
            "n_judges": 0,
            "n_errored": n_errored,
            "n_abstained": n_abstained,
        }

    # Break ties by summed confidence.
    consensus = max(counts, key=lambda label: (counts[label], conf[label]))
    agreement = counts[consensus] / len(ok_verdicts)
    return {
        "test_id": result.test_id,
        "consensus": consensus,
        "agreement": round(agreement, 3),
        "vote_counts": counts,
        "n_judges": len(ok_verdicts),
        "n_errored": n_errored,
        "n_abstained": n_abstained,
    }
