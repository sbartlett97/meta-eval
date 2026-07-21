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
from typing import Dict, List, Sequence

from judges.base import Judge, Verdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalItem:
    """One (test, model output) pair to judge -- the unit of :meth:`evaluate_batch`."""

    test_id: str
    model_output: str
    criteria: str = ""


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

    def evaluate(self, test_id: str, model_output: str, criteria: str) -> PanelResult:
        """Run every judge on one output. Judge errors are captured, not raised."""
        verdicts = [
            judge.evaluate(test_id, model_output, criteria) for judge in self.judges
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
                    judge.evaluate(item.test_id, item.model_output, item.criteria)
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
    """Placeholder aggregation -- SAM OWNS THE REAL META-EVALUATION.

    The PRD marks "Verdict Aggregation" and "Meta-evaluation logic" as Sam's
    manual work (inter-judge agreement, bias detection, weighting by judge
    reliability, etc.). This is a naive majority vote so the pipeline is runnable
    end-to-end; do not treat it as the final methodology.
    """
    counts: Dict[str, int] = {}
    conf: Dict[str, float] = {}
    for v in result.verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        conf[v.verdict] = conf.get(v.verdict, 0.0) + v.confidence
    if not counts:
        return {"test_id": result.test_id, "consensus": "AMBIGUOUS", "agreement": 0.0}

    # Break ties by summed confidence.
    consensus = max(counts, key=lambda label: (counts[label], conf[label]))
    agreement = counts[consensus] / len(result.verdicts)
    return {
        "test_id": result.test_id,
        "consensus": consensus,
        "agreement": round(agreement, 3),
        "vote_counts": counts,
        "n_judges": len(result.verdicts),
    }
