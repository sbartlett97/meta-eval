"""Judge meta-evaluation (docs/critical-assessment.md #1/#5/#8).

``judges.judge_panel.aggregate_verdicts`` computes a per-item majority-vote
consensus; it was never meant to answer "are these judges biased relative to
each other" -- nothing in the codebase did, before this module. This is the
actual meta-evaluation the project is named for: pairwise inter-judge
agreement (with chance-corrected Cohen's kappa, not just raw agreement %,
since judges skewed toward one label agree a lot by chance alone), and a
per-category/technique/severity breakdown of each judge's verdict mix so a
judge that's systematically harsher/softer on a particular slice is visible
rather than averaged away.

Works over the same row shape ``harness/evaluator.py`` writes to
``results/model_verdicts.json`` (``{"test_id", "category", ..., "verdicts":
[{"judge_id", "verdict", "error", ...}, ...]}``), so it can run standalone
against a results file or be called inline during ``run_evaluation``.

A verdict with ``error`` set (the judge call itself failed -- see
``judges/base.py``) is **excluded** from every statistic here: counting a
transient API failure as a genuine "AMBIGUOUS" verdict would quietly bias
agreement and bias statistics toward whatever judge happened to have a flaky
day (docs/critical-assessment.md #2.9).
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

VERDICT_LABELS = ("PASS", "FAIL", "AMBIGUOUS")


# ---------------------------------------------------------------------- #
# Row normalization
# ---------------------------------------------------------------------- #
def _ok_verdicts(row: Dict) -> Dict[str, str]:
    """``{judge_id: verdict}`` for this row, dropping any verdict that isn't a
    real judgement: a failed call (``error`` set) or a structural abstention
    (``abstained`` set -- e.g. HeuristicJudge on a criterion it has no
    keyword signal for; see ``Verdict.abstained``'s docstring). Both would
    otherwise register as genuine agreement/disagreement in
    :func:`pairwise_agreement` / :func:`bias_by_field`, which isn't what
    either actually is.
    """
    out: Dict[str, str] = {}
    for v in row.get("verdicts", []):
        if v.get("error") or v.get("abstained"):
            continue
        judge_id = v.get("judge_id")
        verdict = v.get("verdict")
        if judge_id and verdict:
            out[judge_id] = verdict
    return out


def judge_ids(rows: Iterable[Dict]) -> List[str]:
    seen: Dict[str, None] = {}
    for row in rows:
        for v in row.get("verdicts", []):
            if v.get("judge_id"):
                seen.setdefault(v["judge_id"], None)
    return list(seen)


# ---------------------------------------------------------------------- #
# Pairwise agreement + Cohen's kappa
# ---------------------------------------------------------------------- #
def _cohens_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    """Chance-corrected agreement over a list of (label_a, label_b) pairs.

    Standard two-rater kappa: ``(po - pe) / (1 - pe)`` where ``po`` is observed
    agreement and ``pe`` is the agreement expected from each rater's own
    marginal label distribution alone. Returns ``None`` if there are no
    comparable items.
    """
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n

    dist_a: Dict[str, int] = defaultdict(int)
    dist_b: Dict[str, int] = defaultdict(int)
    for a, b in pairs:
        dist_a[a] += 1
        dist_b[b] += 1
    pe = sum((dist_a.get(label, 0) / n) * (dist_b.get(label, 0) / n) for label in VERDICT_LABELS)

    if pe >= 1.0 - 1e-9:
        # Both raters are single-label over this set -- kappa is degenerate;
        # perfect if they picked the same label, zero otherwise.
        return 1.0 if po >= 1.0 - 1e-9 else 0.0
    return round((po - pe) / (1 - pe), 4)


def pairwise_agreement(rows: Iterable[Dict]) -> Dict[str, Dict]:
    """For every pair of judges that appear together in the data: how many
    items they both gave a (non-errored) verdict on, their raw agreement rate,
    and Cohen's kappa. Keyed by ``"judge_a|judge_b"`` (alphabetically sorted
    so each pair appears once).
    """
    rows = list(rows)
    ids = judge_ids(rows)
    per_pair_labels: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    for row in rows:
        verdicts = _ok_verdicts(row)
        for a, b in itertools.combinations(sorted(verdicts), 2):
            per_pair_labels[(a, b)].append((verdicts[a], verdicts[b]))

    report: Dict[str, Dict] = {}
    for a, b in itertools.combinations(sorted(ids), 2):
        pairs = per_pair_labels.get((a, b), [])
        n = len(pairs)
        agreement = (sum(1 for x, y in pairs if x == y) / n) if n else None
        report[f"{a}|{b}"] = {
            "judge_a": a,
            "judge_b": b,
            "n_common": n,
            "agreement": round(agreement, 4) if agreement is not None else None,
            "kappa": _cohens_kappa(pairs),
        }
    return report


def divergence_from_reference(rows: Iterable[Dict], reference_judge_id: str) -> Dict[str, Dict]:
    """Each other judge's agreement/kappa against one fixed reference judge --
    e.g. the deterministic heuristic baseline, or a trusted remote judge.
    A thin view over :func:`pairwise_agreement` for the pairs that involve
    ``reference_judge_id``.
    """
    full = pairwise_agreement(rows)
    out: Dict[str, Dict] = {}
    for key, stats in full.items():
        a, b = stats["judge_a"], stats["judge_b"]
        if reference_judge_id not in (a, b):
            continue
        other = b if a == reference_judge_id else a
        out[other] = {
            "n_common": stats["n_common"],
            "agreement": stats["agreement"],
            "kappa": stats["kappa"],
        }
    return out


# ---------------------------------------------------------------------- #
# Per-judge verdict distribution by slice (category/technique/severity/...)
# ---------------------------------------------------------------------- #
def bias_by_field(rows: Iterable[Dict], field: str) -> Dict[str, Dict[str, Dict]]:
    """``{judge_id: {field_value: {"n":.., "PASS":.., "FAIL":.., "AMBIGUOUS":..,
    "pass_rate":..}}}``. Lets you spot a judge that's systematically softer or
    harsher on one slice of the suite (e.g. always PASSes "jailbreak-
    susceptibility" while another judge splits it) rather than averaging that
    signal away in a single overall pass rate.
    """
    rows = list(rows)
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in rows:
        slice_value = row.get(field) or "uncategorized"
        for judge_id, verdict in _ok_verdicts(row).items():
            counts[judge_id][slice_value][verdict] += 1

    report: Dict[str, Dict[str, Dict]] = {}
    for judge_id, by_slice in counts.items():
        report[judge_id] = {}
        for slice_value, label_counts in by_slice.items():
            n = sum(label_counts.values())
            entry = {label: label_counts.get(label, 0) for label in VERDICT_LABELS}
            entry["n"] = n
            entry["pass_rate"] = round(entry["PASS"] / n, 4) if n else None
            report[judge_id][slice_value] = entry
    return report


# ---------------------------------------------------------------------- #
# Repeat-sampling variance (docs/critical-assessment.md #7)
# ---------------------------------------------------------------------- #
def sample_variance(rows: Iterable[Dict], verdict_source: str = "consensus") -> Dict[str, Dict]:
    """Per (model, test_id) pass-rate + standard error across repeated
    generation samples (``harness.test_runner.TestRunner.run_tests``'
    ``n_samples`` / the CLI's ``--n-samples``). A single non-deterministic
    generation turns "did the model pass" into a noisy one-shot Bernoulli
    draw with no uncertainty estimate; this is that estimate, once more than
    one sample per (model, test_id) exists in the judged rows.

    Keyed by ``"{model}::{test_id}"``. ``n_samples`` is 1 for any row that
    predates ``--n-samples`` -- harmless (``stderr`` is just 0), not an error.
    """
    groups: Dict[str, List[Optional[str]]] = defaultdict(list)
    for row in rows:
        model, test_id = row.get("model"), row.get("test_id")
        if not model or not test_id:
            continue
        groups[f"{model}::{test_id}"].append(_resolve_verdict_source(row, verdict_source))

    report: Dict[str, Dict] = {}
    for key, labels in groups.items():
        n = len(labels)
        passes = sum(1 for l in labels if l == "PASS")
        p = passes / n if n else None
        report[key] = {
            "n_samples": n,
            "pass_rate": round(p, 4) if p is not None else None,
            # Bernoulli standard error of the pass-rate estimate.
            "stderr": round((p * (1 - p) / n) ** 0.5, 4) if p is not None else None,
            "labels": labels,
        }
    return report


def _resolve_verdict_source(row: Dict, verdict_source: str) -> Optional[str]:
    if verdict_source == "consensus":
        return (row.get("consensus") or {}).get("consensus")
    for v in row.get("verdicts", []):
        if v.get("judge_id") == verdict_source and not v.get("error") and not v.get("abstained"):
            return v.get("verdict")
    return None


# ---------------------------------------------------------------------- #
# Top-level report
# ---------------------------------------------------------------------- #
def meta_evaluation_report(
    rows: Iterable[Dict], reference_judge_id: str = "heuristic-baseline"
) -> Dict:
    """The full judge meta-evaluation over a batch of judged rows.

    ``reference_judge_id`` should be a judge guaranteed independent of every
    model under test (the deterministic heuristic by default; a trusted
    remote judge is also a reasonable choice -- see config/judges.yaml's
    docs/critical-assessment.md #2.1 note on judge/model independence).
    """
    rows = list(rows)
    ids = judge_ids(rows)
    report: Dict = {
        "n_items": len(rows),
        "judges": ids,
        "pairwise_agreement": pairwise_agreement(rows),
        "bias_by_category": bias_by_field(rows, "category"),
        "bias_by_technique": bias_by_field(rows, "technique"),
        "bias_by_severity": bias_by_field(rows, "severity"),
    }
    if reference_judge_id in ids:
        report["divergence_from_reference"] = {
            "reference_judge_id": reference_judge_id,
            "judges": divergence_from_reference(rows, reference_judge_id),
        }
    else:
        logger.info(
            "Reference judge %r not present in this batch; skipping divergence-from-reference.",
            reference_judge_id,
        )

    variance = sample_variance(rows)
    if any(v["n_samples"] > 1 for v in variance.values()):
        report["sample_variance"] = variance
    return report
