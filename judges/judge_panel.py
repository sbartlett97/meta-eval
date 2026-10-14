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
from judges.declines import is_decline
from judges.items import EvalItem, exact_check_applies  # noqa: F401 -- EvalItem re-exported

logger = logging.getLogger(__name__)


@dataclass
class PanelResult:
    """All judges' verdicts for a single (test, model output) pair."""

    test_id: str
    verdicts: List[Verdict] = field(default_factory=list)

    def by_judge(self) -> Dict[str, Verdict]:
        return {v.judge_id: v for v in self.verdicts}


class JudgePanel:
    """Fan one (test, output) pair out to every judge and gather verdicts."""

    def __init__(self, judges: Sequence[Judge], decline_registry=None) -> None:
        if not judges:
            raise ValueError("JudgePanel requires at least one judge")
        self.judges = list(judges)
        #: judges.declines.DeclineRegistry, or None: content a judge declined
        #: before is never sent to that judge again.
        self.decline_registry = decline_registry
        # Only skip exactly-checkable items when a judge here will decide them.
        self._has_exact_judge = any(j.runs_exact_checks for j in self.judges)

    def evaluate(
        self,
        test_id: str,
        model_output: str,
        criteria: str,
        category: str = "uncategorized",
        metadata: Dict[str, Any] | None = None,
    ) -> PanelResult:
        """Run every judge on one output. Judge errors are captured, not raised."""
        item = EvalItem(test_id, model_output, criteria, category, metadata or {})
        return PanelResult(test_id=test_id, verdicts=[self._judge(j, item) for j in self.judges])

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
            for result, verdict in zip(results, self.judge_all(judge, items)):
                result.verdicts.append(verdict)
        return results

    def judge_all(self, judge: Judge, items: Sequence[EvalItem]) -> List[Verdict]:
        """One judge's verdicts on many items, in item order.

        Items an exact check decides are skipped for the other judges (see
        :meth:`_skip_verdict`). The rest go to ``judge.evaluate_many`` in one
        call, so a batching backend (ClaudeJudge with ``use_batch``) can send
        them as a single batch.
        """
        verdicts: List[Optional[Verdict]] = [self._skip_verdict(judge, item) for item in items]
        todo = [i for i, v in enumerate(verdicts) if v is None]
        for i, verdict in zip(todo, judge.evaluate_many([items[i] for i in todo])):
            verdicts[i] = verdict
            if self.decline_registry is not None and is_decline(verdict):
                self.decline_registry.record(judge.id, items[i], verdict.abstain_reason)
        return verdicts  # type: ignore[return-value]

    def _judge(self, judge: Judge, item: EvalItem) -> Verdict:
        return self.judge_all(judge, [item])[0]

    def _skip_verdict(self, judge: Judge, item: EvalItem) -> Optional[Verdict]:
        """An abstained "skipped" verdict if an exact deterministic check (a
        format check) decides this item and this judge does not run it, or if
        the judge's content policy skips it; else None. Skipped judges neither outvote the exact check nor spend tokens
        on it (validation gap C12)."""
        if (
            self._has_exact_judge
            and not judge.runs_exact_checks
            and exact_check_applies(item)
        ):
            return Verdict(
                verdict="AMBIGUOUS",
                confidence=0.0,
                judge_id=judge.id,
                test_id=item.test_id,
                rationale="Skipped: an exact deterministic check decides this item.",
                abstained=True,
            )
        reason = judge.policy_skip_reason(item)
        if reason is None and self.decline_registry is not None and \
                self.decline_registry.declined(judge.id, item):
            reason = "previously_declined"
        if reason is not None:
            # No call is made: the judge's content policy (judges.yaml `skip:`)
            # keeps this item away from its provider. Kept apart from
            # "judge_declined", which is a real decline by the judge.
            return Verdict(
                verdict="AMBIGUOUS",
                confidence=0.0,
                judge_id=judge.id,
                test_id=item.test_id,
                rationale="Skipped by this judge's content policy.",
                abstained=True,
                abstain_reason=f"skipped_by_policy:{reason}",
            )
        return None

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
    for v in ok_verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    if not counts:
        # No judge gave a real verdict (all errored or abstained). This is
        # missing data, not a judgement: None, so downstream rates drop the
        # item instead of counting it as a non-pass.
        return {
            "test_id": result.test_id,
            "consensus": None,
            "agreement": 0.0,
            "n_judges": 0,
            "n_errored": n_errored,
            "n_abstained": n_abstained,
        }

    # A tie for first place is AMBIGUOUS. Breaking it by summed confidence
    # would favour whichever judge reports higher confidence, and confidence
    # scales differ between judges (validation gap C10).
    top = max(counts.values())
    leaders = [label for label, n in counts.items() if n == top]
    consensus = leaders[0] if len(leaders) == 1 else "AMBIGUOUS"
    agreement = top / len(ok_verdicts)
    return {
        "test_id": result.test_id,
        "consensus": consensus,
        "agreement": round(agreement, 3),
        "vote_counts": counts,
        "n_judges": len(ok_verdicts),
        "n_errored": n_errored,
        "n_abstained": n_abstained,
    }
