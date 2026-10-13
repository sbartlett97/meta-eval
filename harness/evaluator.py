"""Evaluation orchestrator / CLI entrypoint (PRD v3.1).

Ties the pieces together:

    0. Hydrate: download every open-weight checkpoint from HuggingFace up front
       (skip with ``--no-hydrate``).
    1. (optional) generate model outputs via ``TestRunner`` if they don't exist.
    2. Build the judge panel from ``config/judges.yaml``.
    3. Run every judge over every model output.
    4. Persist per-judge verdicts + a naive consensus to
       ``results/model_verdicts.json`` (rows under ``verdicts``).

Usage (from the PRD setup section):
    python harness/evaluator.py --models mistral-7b-4bit --judges all
    python harness/evaluator.py --outputs results/model_outputs.json --judges cheap
    python harness/evaluator.py --models mistral-7b-4bit --no-judge   # generate only

Note: the per-item consensus is a naive placeholder majority vote (see
``judges.judge_panel.aggregate_verdicts``). The real cross-item meta-evaluation
-- inter-judge agreement/kappa, per-category/technique/severity bias -- is
computed separately and written to ``results/judge_meta_eval.json``; see
``judges.meta_eval``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Allow running as a script (`python harness/evaluator.py`, as documented in the
# README). Executing a file directly puts its own directory (harness/) on
# sys.path but NOT the project root, so `import harness` / `import judges` would
# fail with ModuleNotFoundError. Add the project root before the first-party
# imports so the documented command works without installing the package.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from harness import provenance
from harness.env import load_env
from harness.hydrate import hydrate_weights
from harness.model_loader import ModelLoader
from harness.test_runner import CheckpointMismatchError, TestRunner
from judges.factory import build_judges
from judges import rubric
from judges.judge_panel import EvalItem, JudgePanel, aggregate_verdicts
from judges.meta_eval import meta_evaluation_report

logger = logging.getLogger(__name__)

# Judge-subset presets (PRD "Decision 3: Judge Placement").
_JUDGE_PRESETS = {
    "all": None,      # every enabled judge
    "cheap": 1,       # priority <= 1 (remote API judges + free heuristic; no local GGUF loads)
    "pilot": 3,       # everything, but named for clarity in pilots
}


class MissingCredentialsError(RuntimeError):
    """An enabled judge has no API credential, so every verdict it gave would be an error."""


def require_credentials(judges) -> None:
    """Raise :class:`MissingCredentialsError` if any judge cannot authenticate.

    Without this, a remote judge with no key fails closed to an AMBIGUOUS
    verdict on every row and the run still exits 0 -- the panel silently loses
    its model-independent anchor.
    """
    missing = [f"{j.id} (needs {j.credential_env})" for j in judges if not j.has_credentials()]
    if missing:
        raise MissingCredentialsError(
            "Judge(s) without credentials: " + ", ".join(missing)
            + ". Set the variable(s) in the environment or in .env, or disable those judges."
        )


def run_evaluation(
    outputs_path: str,
    judges_config: str = "config/judges.yaml",
    verdicts_path: str = "results/model_verdicts.json",
    max_priority: Optional[int] = None,
    hardware_profile: str = "config/hardware_profile.yaml",
    meta_eval_path: Optional[str] = "results/judge_meta_eval.json",
    reference_judge_id: Optional[str] = None,
    judge_ids: Optional[List[str]] = None,
    allow_legacy: bool = False,
    batch: bool = False,
    self_consistency: float = 0.0,
) -> str:
    """Judge previously-generated model outputs; write verdicts. Returns the path.

    ``judge_ids`` selects exactly those judges and overrides ``max_priority``
    (see ``judges.factory.select_judge_entries``). An outputs file that
    predates the current schema raises :class:`LegacyOutputsError` unless
    ``allow_legacy`` is True. ``self_consistency`` (0-1) re-grades that share
    of rows with every non-deterministic judge (see :func:`_regrade_subset`).
    """
    judges = build_judges(
        judges_config, max_priority=max_priority, hardware_profile=hardware_profile,
        judge_ids=judge_ids,
    )
    require_credentials(judges)
    if batch:
        # Judges with a batch API grade the whole run in one batch (about
        # half the price; results can take up to 24 h).
        for judge in judges:
            if judge.supports_batch:
                judge.use_batch = True
    panel = JudgePanel(judges)
    logger.info("Judge panel: %s", [j.id for j in judges])

    os.makedirs(os.path.dirname(verdicts_path) or ".", exist_ok=True)

    # Read all outputs, then judge the batch judge-outer (each judge processes
    # every row before the next judge runs). This keeps a sequentially-loaded
    # local judge resident for all its calls instead of reloading per row.
    rows, outputs_provenance = _read_outputs(outputs_path)
    if not allow_legacy:
        _check_current_schema(outputs_path, rows, outputs_provenance)
    # A failed generation, or a reasoning trace cut off before any answer,
    # has no response to grade. Judging its empty output would record the
    # failure as model behaviour (validation gaps C3, C7), so these rows skip
    # the panel and are kept as missing data instead.
    judged_rows = [row for row in rows if _generation_error(row) is None]
    if len(judged_rows) < len(rows):
        logger.warning(
            "Skipping %d row(s) with no answer to grade; they are recorded with "
            "generation_error set and no verdicts.", len(rows) - len(judged_rows),
        )
    items = [
        EvalItem(
            test_id=row["test_id"],
            # Multi-turn cases are judged on the full exchange (see
            # TestRunner._run_one / _format_transcript), not just the final
            # reply -- otherwise a consistency check has nothing to check.
            model_output=row.get("transcript") or row.get("output", ""),
            criteria=row.get("criteria", ""),
            category=row.get("category", "uncategorized"),
            metadata=row.get("metadata") or {},
            turns=tuple(row.get("turns") or ()),
            system_prompt=row.get("system_prompt"),
            item_type=row.get("item_type") or "",  # "" -> inferred (judges/items.py)
            severity_tier=row.get("severity_tier"),
            reference_answer=row.get("reference_answer"),
        )
        for row in judged_rows
    ]
    untiered = sum(
        1 for item in items
        if item.item_type in ("harmful_request", "jailbreak") and item.severity_tier not in (1, 2, 3)
    )
    if untiered:
        logger.warning(
            "%d harmful item(s) have no severity_tier; LLM judges score them for refusal "
            "only (decision D3). Tag tiers 1-2 in the suite to score harm quality.", untiered,
        )
    results = iter(panel.evaluate_batch(items))
    # The resolved item type of each judged row, for slicing verdicts later.
    item_types = {id(row): item.item_type for row, item in zip(judged_rows, items)}
    regrades = _regrade_subset(panel, judged_rows, items, self_consistency)

    records = [
        {
            "test_id": row["test_id"],
            "model": row.get("model"),
            # 0 when the outputs file predates --n-samples / doesn't set it.
            "sample_index": row.get("sample_index", 0),
            # Carried through from the suite (see harness/test_suite.py) so
            # per-category/technique/severity breakdowns are possible without
            # re-joining against the original suite file.
            "category": row.get("category", "uncategorized"),
            "domain": row.get("domain", ""),
            "technique": row.get("technique", ""),
            "severity": row.get("severity", ""),
            "source": row.get("source", ""),
            "item_type": item_types.get(id(row)),
            "severity_tier": row.get("severity_tier"),
            **_judgement(row, results),
            **({"regrade_verdicts": regrades[id(row)]} if id(row) in regrades else {}),
        }
        for row in rows
    ]
    payload = {
        "verdicts": records,
        "provenance": provenance.collect(
            outputs_path=outputs_path,
            judges_config=judges_config if isinstance(judges_config, str) else None,
            judge_panel=[j.id for j in judges],
            judge_details=[j.describe() for j in judges],
            rubric_version=rubric.RUBRIC_VERSION,
            rubric_sha256=rubric.rubric_sha256(),
            # The suite hash, models and seeds live here, so verdicts can be
            # traced back to exactly what was generated.
            outputs_provenance=outputs_provenance,
        ),
    }
    with open(verdicts_path, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2)
        sink.write("\n")

    logger.info("Wrote %d verdict row(s) to %s", len(records), verdicts_path)
    logger.info("Cost/latency summary: %s", panel.cost_summary())

    # The actual meta-evaluation (docs/critical-assessment.md #1/#8):
    # inter-judge agreement (Cohen's kappa, not just raw agreement -- judges
    # skewed toward one label agree by chance alone) and per-category/
    # technique/severity bias, not just a naive per-item majority vote.
    if meta_eval_path:
        report = meta_evaluation_report(records, reference_judge_id=reference_judge_id)
        os.makedirs(os.path.dirname(meta_eval_path) or ".", exist_ok=True)
        with open(meta_eval_path, "w", encoding="utf-8") as sink:
            json.dump(report, sink, indent=2)
            sink.write("\n")
        logger.info("Wrote judge meta-evaluation to %s", meta_eval_path)
    return verdicts_path


def in_regrade_subset(row: Dict, fraction: float) -> bool:
    """Whether a row is in the self-consistency subset. Chosen by a hash of
    (model, test_id, sample_index), so the same rows are picked on every run
    and the choice cannot depend on the verdicts."""
    if fraction <= 0:
        return False
    key = f"{row.get('model')}|{row.get('test_id')}|{row.get('sample_index', 0)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def _regrade_subset(panel: JudgePanel, judged_rows: List[Dict], items, fraction: float) -> Dict:
    """Grade a fixed subset a second time with each non-deterministic judge
    (task 2.10). Temperature cannot be fixed on current Claude models, so how
    often a judge agrees with itself is measured instead
    (judges.meta_eval.self_consistency). Returns ``{id(row): [verdict dicts]}``."""
    chosen = [(row, item) for row, item in zip(judged_rows, items) if in_regrade_subset(row, fraction)]
    if not chosen:
        return {}
    regrades: Dict[int, List[Dict]] = {id(row): [] for row, _ in chosen}
    for judge in panel.judges:
        if judge.provider == "deterministic":
            continue  # the same input always gives the same verdict
        verdicts = panel.judge_all(judge, [item for _, item in chosen])
        for (row, _), verdict in zip(chosen, verdicts):
            regrades[id(row)].append(verdict.to_dict())
    logger.info("Self-consistency: re-graded %d row(s)", len(chosen))
    return regrades


def _generation_error(row: Dict) -> Optional[str]:
    """Why a row has no answer to grade, or None if it can be judged."""
    if row.get("error"):
        return row["error"]
    if row.get("reasoning_complete") is False:
        return "no final answer: the reasoning trace was cut off (raise serving.max_tokens)"
    return None


def _judgement(row: Dict, results) -> Dict:
    """Verdict fields for one output row; ``results`` yields the panel results
    for the judged rows, in order."""
    error = _generation_error(row)
    if error is not None:
        return {"generation_error": error, "verdicts": [], "consensus": None}
    result = next(results)
    return {
        "generation_error": None,
        "verdicts": [v.to_dict() for v in result.verdicts],
        "consensus": aggregate_verdicts(result),
    }


class LegacyOutputsError(ValueError):
    """The outputs file predates the current schema and should be regenerated."""


#: Row fields every current outputs file has. Older files lack them, and also
#: lack the fixes they stand for: system prompts were dropped, multi-turn cases
#: were collapsed, and judges could not see the request (validation gap C8).
_REQUIRED_ROW_KEYS = ("sample_index", "turns")


def _read_outputs(outputs_path: str):
    """``(rows, provenance)`` from an outputs file. ``provenance`` is None for
    a bare array or a legacy ``.jsonl`` file."""
    if outputs_path.endswith(".jsonl"):
        with open(outputs_path, "r", encoding="utf-8") as src:
            return [json.loads(line) for line in src if line.strip()], None
    with open(outputs_path, "r", encoding="utf-8") as src:
        data = json.load(src)
    if isinstance(data, dict):
        return data.get("outputs", []), data.get("provenance")
    return data, None


def _load_output_rows(outputs_path: str) -> List[Dict]:
    """Load model-output rows from a ``{"outputs": [...]}`` JSON file.

    A bare top-level array is accepted too, and legacy ``.jsonl`` outputs (one
    object per line) still load, dispatched on the file extension.
    """
    return _read_outputs(outputs_path)[0]


def _check_current_schema(outputs_path: str, rows: List[Dict], provenance) -> None:
    """Raise :class:`LegacyOutputsError` unless the file has provenance and
    every row has the current required fields."""
    problems = []
    if not provenance:
        problems.append("no provenance block")
    missing = sorted({k for row in rows for k in _REQUIRED_ROW_KEYS if k not in row})
    if missing:
        problems.append(f"rows without {missing}")
    if problems:
        raise LegacyOutputsError(
            f"{outputs_path} predates the current output schema ({'; '.join(problems)}). "
            "Regenerate it with --models, or pass --allow-legacy to judge it anyway."
        )


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval harness.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model ids to generate outputs for first (from config/models.yaml).",
    )
    parser.add_argument("--test-suite", default="data/test_suite_v1.json")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--judges-config", default="config/judges.yaml")
    parser.add_argument("--hardware-config", default="config/hardware_profile.yaml")
    parser.add_argument(
        "--outputs",
        default="results/model_outputs.json",
        help="Model outputs to judge (generated first if --models is given).",
    )
    parser.add_argument("--verdicts", default="results/model_verdicts.json")
    parser.add_argument(
        "--meta-eval",
        default="results/judge_meta_eval.json",
        help="Where to write the inter-judge agreement/bias report. Pass an "
        "empty string to skip it.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Grade with the provider's batch API where a judge supports it "
        "(Claude): about half the price, results can take up to 24 hours.",
    )
    parser.add_argument(
        "--self-consistency",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="Re-grade this share of rows (e.g. 0.1) with each non-deterministic "
        "judge, to measure how often a judge agrees with itself.",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Judge an outputs file that predates the current schema (no "
        "provenance / sample_index / turns). Its numbers are not comparable "
        "with a current run.",
    )
    parser.add_argument(
        "--reference-judge",
        default=None,
        help="Optional judge id to compute divergence-from-reference against in "
        "the meta-eval report; pick one independent of every model under test.",
    )
    parser.add_argument(
        "--judges",
        default="all",
        choices=sorted(_JUDGE_PRESETS),
        help="Judge subset preset.",
    )
    parser.add_argument(
        "--judge-ids",
        type=lambda v: [x.strip() for x in v.split(",") if x.strip()],
        default=None,
        help="Comma-separated judge ids from judges.yaml (e.g. "
        "heuristic-baseline,claude-sonnet-5-5). Overrides --judges and the "
        "enabled flag.",
    )
    parser.add_argument(
        "--prefer-engine", default="llamacpp", choices=["llamacpp", "ollama"]
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Independent generations per (model, test) at the same sampling "
        "settings, so pass rates carry a variance estimate instead of a "
        "single noisy draw (see judges.meta_eval.sample_variance). Only "
        "applies with --models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for generation; sample i uses seed + i. Recorded on "
        "every output row. Only applies with --models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted generation run from its checkpoint "
        "(<outputs>.partial.jsonl): keep rows that succeeded, re-run errored "
        "and missing ones. Only applies with --models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a new generation run to replace an existing --outputs file.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Only generate model outputs; skip the judge panel. Requires --models.",
    )
    parser.add_argument(
        "--no-hydrate",
        action="store_true",
        help="Skip the startup step that downloads open weights from HuggingFace.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.no_judge and not args.models:
        parser.error("--no-judge only makes sense with --models (nothing to judge, "
                     "nothing to generate otherwise).")

    load_env()
    # Check judge credentials before any download or generation, so a missing
    # key doesn't surface only after hours of generation.
    if not args.no_judge:
        try:
            require_credentials(build_judges(
                args.judges_config,
                max_priority=_JUDGE_PRESETS[args.judges],
                hardware_profile=args.hardware_config,
                judge_ids=args.judge_ids,
            ))
        except (MissingCredentialsError, ValueError) as exc:
            logger.error("%s", exc)
            return 2

    # Hydration: pull the open-weight checkpoints THIS run needs from HF up front
    # so the first in-process llama.cpp load doesn't block mid-run on a multi-GB
    # download. Scoped to the requested models + the judges actually selected, so
    # e.g. a remote-only `--judges cheap` re-judge downloads nothing.
    if not args.no_hydrate:
        hydrate_weights(
            models_config=args.models_config,
            judges_config=args.judges_config,
            model_ids=args.models or [],
            include_judges=not args.no_judge,
            judge_max_priority=_JUDGE_PRESETS[args.judges],
            judge_ids=args.judge_ids,
        )

    if args.models:
        loader = ModelLoader(
            args.models_config,
            prefer_engine=args.prefer_engine,
            hardware_profile=args.hardware_config,
        )
        runner = TestRunner(args.test_suite, loader, results_path=args.outputs)
        # Rows are checkpointed as they finish; see TestRunner.run_tests.
        try:
            runner.run_tests(
                args.models, n_samples=args.n_samples,
                resume=args.resume, overwrite=args.overwrite, seed=args.seed,
            )
        except (FileExistsError, CheckpointMismatchError) as exc:
            logger.error("%s", exc)
            return 2

    if args.no_judge:
        logger.info("Generation-only run (--no-judge); outputs at %s", args.outputs)
        return 0

    try:
        run_evaluation(
            outputs_path=args.outputs,
            judges_config=args.judges_config,
            verdicts_path=args.verdicts,
            max_priority=_JUDGE_PRESETS[args.judges],
            hardware_profile=args.hardware_config,
            meta_eval_path=args.meta_eval or None,
            reference_judge_id=args.reference_judge,
            judge_ids=args.judge_ids,
            allow_legacy=args.allow_legacy,
            batch=args.batch,
            self_consistency=args.self_consistency,
        )
    except LegacyOutputsError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
