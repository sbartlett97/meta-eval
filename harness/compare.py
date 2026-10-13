"""Behavioral-invariance comparison (PRD v3.1; docs/critical-assessment.md #1/#7/#11).

The harness is named for measuring whether fine-tuning or quantization changes
model *behavior* relative to a base checkpoint. Nothing else in this codebase
computes that diff -- `harness/evaluator.py` only ever produces independent
per-model verdicts. This module closes that gap: given one or more already-judged
`results/model_verdicts.json`-shaped files and the `base_model_id` linkage in
`config/models.yaml` (see that file's header comment), it reports, per
(variant, base) pair:

  * overall and per-category/technique/severity pass-rate deltas, computed only
    over test ids both models were actually run on (an apples-to-apples set --
    see :func:`_common_test_ids`), and
  * which individual test ids *flipped* verdict between the two models, with
    both verdicts shown.

If either side used ``--n-samples`` > 1, repeated samples of the same test are
collapsed to one verdict via majority vote before comparing -- see
:func:`index_by_model` for why that has to be handled differently from a
plain re-judge.

This is deliberately pure data-crunching over already-written JSON: no network,
no model loading, no judge calls -- so it is fast, free, and safe to test.

Usage::

    python harness/compare.py --verdicts results/model_verdicts.json
    python harness/compare.py --verdicts results/model_verdicts.json \\
        --pairs gemma-4-12B-agentic-fable5:gemma-4-12b
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_MODEL_GROUPS = ("local_models", "fine_tuned_models", "remote_models")


# ---------------------------------------------------------------------- #
# Loading
# ---------------------------------------------------------------------- #
def load_verdict_rows(paths: Iterable[str]) -> List[Dict]:
    """Load and concatenate the ``verdicts`` rows from one or more result files."""
    rows: List[Dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rows.extend(data.get("verdicts", []) if isinstance(data, dict) else data)
    return rows


def index_by_model(
    rows: Iterable[Dict], verdict_source: str = "consensus"
) -> Dict[str, Dict[str, Dict]]:
    """``{model_id: {test_id: row}}`` -- one row per (model, test_id).

    Two different situations can put more than one row under the same
    (model, test_id), and they must NOT be handled the same way:

    * **Re-judging**: concatenating a re-judge file after the original (see
      :func:`load_verdict_rows`). These rows all carry the same
      ``sample_index`` (0, or absent) -- the later file's verdict naturally
      wins, same as before.
    * **Repeated sampling** (``harness.test_runner``'s ``--n-samples`` > 1):
      multiple *distinct* ``sample_index`` values for one (model, test_id).
      Silently keeping whichever sample happened to load last would produce
      a comparison that reports as complete while actually being built on an
      arbitrary single draw -- exactly the "looks right, isn't" failure this
      tool exists to avoid (docs/critical-assessment.md #2.2/#7). These are
      collapsed via majority vote across the samples' ``verdict_source``
      label instead; use ``judges.meta_eval.sample_variance`` for the fuller
      per-sample pass-rate + stderr picture.

    ``verdict_source`` MUST match whatever you'll pass to :func:`compare_pair`
    / :func:`compare_all` afterwards. It's baked into the collapsed row at
    build time here (see :func:`_collapse_samples`) -- indexing on
    ``"consensus"`` and then comparing on a named judge id (or vice versa)
    silently reads a stale single-sample verdict instead of the majority
    across samples, the exact bug this collapsing exists to prevent.
    """
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        model, test_id = row.get("model"), row.get("test_id")
        if model and test_id:
            grouped[(model, test_id)].append(row)

    by_model: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for (model, test_id), group in grouped.items():
        sample_indices = {r.get("sample_index", 0) for r in group}
        if len(sample_indices) > 1:
            by_model[model][test_id] = _collapse_samples(group, verdict_source)
        else:
            by_model[model][test_id] = group[-1]  # re-judge overwrite: latest wins
    return dict(by_model)


def _collapse_samples(group: List[Dict], verdict_source: str) -> Dict:
    """One row per (model, test_id) from several repeated-sample rows, via
    majority vote across their ``verdict_source`` label. Called only when the
    rows actually carry distinct ``sample_index`` values -- see
    :func:`index_by_model`.

    The majority label is written back under BOTH ``consensus.consensus`` and
    (when ``verdict_source`` names a specific judge) that judge's entry in
    ``verdicts`` -- so a later ``_resolve_verdict(collapsed_row,
    verdict_source)`` returns the aggregated answer for whichever source was
    actually used to collapse, not sample 0's raw, un-aggregated call.
    """
    labels = [l for l in (_resolve_verdict(r, verdict_source) for r in group) if l]
    counts = Counter(labels)
    majority = counts.most_common(1)[0][0] if counts else None

    collapsed = dict(group[0])
    collapsed["consensus"] = {
        "consensus": majority,
        "n_samples": len(group),
        "vote_counts": dict(counts),
    }
    if verdict_source != "consensus":
        other_verdicts = [
            v for v in collapsed.get("verdicts", []) if v.get("judge_id") != verdict_source
        ]
        collapsed["verdicts"] = other_verdicts + [
            {"judge_id": verdict_source, "verdict": majority, "error": None}
        ]
    collapsed["sample_index"] = f"aggregated(n={len(group)})"
    logger.info(
        "Collapsed %d samples of %s/%s into one row via majority vote on %s (%s)",
        len(group), collapsed.get("model"), collapsed.get("test_id"), verdict_source, majority,
    )
    return collapsed


def discover_pairs(models_config) -> List[Tuple[str, str]]:
    """``[(variant_id, base_id), ...]`` from every entry with a non-null
    ``base_model_id`` in ``config/models.yaml`` (or an equivalent parsed dict).
    """
    cfg = _as_dict(models_config)
    pairs: List[Tuple[str, str]] = []
    for group in _MODEL_GROUPS:
        for entry in cfg.get(group, []) or []:
            base = entry.get("base_model_id")
            if base:
                pairs.append((entry["id"], base))
    return pairs


# ---------------------------------------------------------------------- #
# Comparison
# ---------------------------------------------------------------------- #
def _resolve_verdict(row: Dict, verdict_source: str) -> Optional[str]:
    """The verdict label to use for one row: the naive consensus by default, or
    one named judge's own verdict (``verdict_source`` = that judge's id).

    A verdict whose call failed (``error`` set) or that's a structural
    abstention (``abstained`` set -- see ``Verdict.abstained``) is not that
    judge's real assessment, so it's skipped -- same treatment as
    ``judges.meta_eval._ok_verdicts``, for the same reason: counting either as
    a genuine PASS/FAIL/AMBIGUOUS would misrepresent what the judge actually
    decided about this output.
    """
    if verdict_source == "consensus":
        return (row.get("consensus") or {}).get("consensus")
    for v in row.get("verdicts", []):
        if v.get("judge_id") == verdict_source and not v.get("error") and not v.get("abstained"):
            return v.get("verdict")
    return None


def _common_test_ids(variant_rows: Dict[str, Dict], base_rows: Dict[str, Dict]) -> List[str]:
    return sorted(set(variant_rows) & set(base_rows))


def _pass_rate(labels: Iterable[str]) -> Optional[float]:
    """Share of PASS labels. Callers pass real verdicts only (no None). A
    genuine AMBIGUOUS -- the judge read the response and could not decide --
    stays in the denominator as a non-pass; compare_pair reports its count."""
    labels = list(labels)
    if not labels:
        return None
    return sum(1 for l in labels if l == "PASS") / len(labels)


def compare_pair(
    variant_id: str,
    base_id: str,
    by_model: Dict[str, Dict[str, Dict]],
    verdict_source: str = "consensus",
) -> Dict:
    """The behavioral-invariance diff between one fine-tuned/quantized variant
    and the base it's linked to. Returns a dict with per-category/technique/
    severity pass-rate deltas and the list of test ids whose verdict flipped.
    """
    variant_rows = by_model.get(variant_id, {})
    base_rows = by_model.get(base_id, {})
    common = _common_test_ids(variant_rows, base_rows)

    result: Dict = {
        "variant": variant_id,
        "base": base_id,
        "verdict_source": verdict_source,
        "n_variant_tests": len(variant_rows),
        "n_base_tests": len(base_rows),
        "n_common_tests": len(common),
    }
    if not common:
        result["warning"] = "no test ids in common -- both models must be judged on the same suite"
        return result

    variant_labels = {t: _resolve_verdict(variant_rows[t], verdict_source) for t in common}
    base_labels = {t: _resolve_verdict(base_rows[t], verdict_source) for t in common}

    # A paired diff needs a real verdict on both sides. A side with none (every
    # judge errored or abstained, or the generation failed) is missing data:
    # counting it as a non-pass would invent a flip.
    paired = [t for t in common if variant_labels[t] is not None and base_labels[t] is not None]
    dropped = [t for t in common if t not in set(paired)]
    result["n_paired_tests"] = len(paired)
    result["n_dropped_no_verdict"] = len(dropped)
    result["dropped_test_ids"] = dropped
    if not paired:
        result["warning"] = "no test has a real verdict on both sides"
        return result

    result["overall"] = {
        "variant_pass_rate": _pass_rate(variant_labels[t] for t in paired),
        "base_pass_rate": _pass_rate(base_labels[t] for t in paired),
        "variant_n_ambiguous": sum(variant_labels[t] == "AMBIGUOUS" for t in paired),
        "base_n_ambiguous": sum(base_labels[t] == "AMBIGUOUS" for t in paired),
    }
    result["overall"]["delta"] = _delta(result["overall"])

    for dimension in ("category", "domain", "technique", "severity"):
        result[dimension] = _breakdown_by(dimension, paired, variant_rows, variant_labels, base_labels)

    flips = []
    for t in paired:
        v, b = variant_labels[t], base_labels[t]
        if v != b:
            flips.append({"test_id": t, "base_verdict": b, "variant_verdict": v})
    result["flipped"] = flips
    result["n_flipped"] = len(flips)

    return result


def _breakdown_by(
    field: str,
    paired: List[str],
    variant_rows: Dict[str, Dict],
    variant_labels: Dict[str, Optional[str]],
    base_labels: Dict[str, Optional[str]],
) -> Dict[str, Dict]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for t in paired:
        groups[variant_rows[t].get(field) or "uncategorized"].append(t)

    breakdown: Dict[str, Dict] = {}
    for key, test_ids in sorted(groups.items()):
        v_rate = _pass_rate(variant_labels[t] for t in test_ids)
        b_rate = _pass_rate(base_labels[t] for t in test_ids)
        entry = {"n": len(test_ids), "variant_pass_rate": v_rate, "base_pass_rate": b_rate}
        entry["delta"] = _delta(entry)
        breakdown[key] = entry
    return breakdown


def _delta(entry: Dict) -> Optional[float]:
    v, b = entry.get("variant_pass_rate"), entry.get("base_pass_rate")
    if v is None or b is None:
        return None
    return round(v - b, 4)


def compare_all(
    by_model: Dict[str, Dict[str, Dict]],
    pairs: Iterable[Tuple[str, str]],
    verdict_source: str = "consensus",
) -> List[Dict]:
    """Run :func:`compare_pair` for every pair, skipping (with a log warning)
    any pair where a side has no judged rows at all -- there's nothing to
    diff, and this shouldn't abort comparing the other pairs.
    """
    results = []
    for variant_id, base_id in pairs:
        if variant_id not in by_model or base_id not in by_model:
            logger.warning(
                "Skipping %s vs %s: no judged rows for %s",
                variant_id,
                base_id,
                variant_id if variant_id not in by_model else base_id,
            )
            continue
        results.append(compare_pair(variant_id, base_id, by_model, verdict_source))
    return results


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _as_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        with open(value, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute behavioral-invariance diffs between fine-tuned/"
        "quantized variants and their base models."
    )
    parser.add_argument(
        "--verdicts", nargs="+", required=True,
        help="One or more results/model_verdicts.json-shaped files.",
    )
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help="Explicit 'variant:base' pairs. Default: every base_model_id link "
        "in --models-config.",
    )
    parser.add_argument(
        "--verdict-source", default="consensus",
        help="'consensus' (default) or a specific judge id to diff on that "
        "judge's own verdict instead of the panel consensus.",
    )
    parser.add_argument("--out", default=None, help="Write JSON here instead of stdout.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rows = load_verdict_rows(args.verdicts)
    # Must match --verdict-source: index_by_model bakes it into any collapsed
    # (repeated-sample) row -- see its docstring.
    by_model = index_by_model(rows, verdict_source=args.verdict_source)

    if args.pairs:
        pairs = [tuple(p.split(":", 1)) for p in args.pairs]
    else:
        pairs = discover_pairs(args.models_config)

    if not pairs:
        logger.warning(
            "No (variant, base) pairs to compare -- set base_model_id in "
            "%s or pass --pairs.", args.models_config,
        )

    results = compare_all(by_model, pairs, verdict_source=args.verdict_source)
    payload = json.dumps({"comparisons": results}, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as sink:
            sink.write(payload)
        logger.info("Wrote %d comparison(s) to %s", len(results), args.out)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
