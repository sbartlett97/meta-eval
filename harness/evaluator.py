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
    reference_judge_id: str = "heuristic-baseline",
    judge_ids: Optional[List[str]] = None,
) -> str:
    """Judge previously-generated model outputs; write verdicts. Returns the path.

    ``judge_ids`` selects exactly those judges and overrides ``max_priority``
    (see ``judges.factory.select_judge_entries``).
    """
    judges = build_judges(
        judges_config, max_priority=max_priority, hardware_profile=hardware_profile,
        judge_ids=judge_ids,
    )
    require_credentials(judges)
    panel = JudgePanel(judges)
    logger.info("Judge panel: %s", [j.id for j in judges])

    os.makedirs(os.path.dirname(verdicts_path) or ".", exist_ok=True)

    # Read all outputs, then judge the batch judge-outer (each judge processes
    # every row before the next judge runs). This keeps a sequentially-loaded
    # local judge resident for all its calls instead of reloading per row.
    rows = _load_output_rows(outputs_path)
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
        )
        for row in rows
    ]
    results = panel.evaluate_batch(items)

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
            "verdicts": [v.to_dict() for v in result.verdicts],
            "consensus": aggregate_verdicts(result),
        }
        for row, result in zip(rows, results)
    ]
    payload = {
        "verdicts": records,
        "provenance": provenance.collect(
            outputs_path=outputs_path,
            judges_config=judges_config if isinstance(judges_config, str) else None,
            judge_panel=[j.id for j in judges],
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


def _load_output_rows(outputs_path: str) -> List[Dict]:
    """Load model-output rows from a ``{"outputs": [...]}`` JSON file.

    A bare top-level array is accepted too, and legacy ``.jsonl`` outputs (one
    object per line) still load, dispatched on the file extension.
    """
    if outputs_path.endswith(".jsonl"):
        with open(outputs_path, "r", encoding="utf-8") as src:
            return [json.loads(line) for line in src if line.strip()]
    with open(outputs_path, "r", encoding="utf-8") as src:
        data = json.load(src)
    if isinstance(data, dict):
        return data.get("outputs", [])
    return data


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
        "--reference-judge",
        default="heuristic-baseline",
        help="Judge id to compute divergence-from-reference against in the "
        "meta-eval report; should be independent of every model under test.",
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
        "heuristic-baseline,claude-sonnet-4-6). Overrides --judges and the "
        "enabled flag.",
    )
    parser.add_argument(
        "--prefer-engine", default="llamacpp", choices=["llamacpp", "ollama"]
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Independent generations per (model, test) at the same sampling "
        "settings, so pass rates carry a variance estimate instead of a "
        "single noisy draw (see judges.meta_eval.sample_variance). Only "
        "applies with --models.",
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
                resume=args.resume, overwrite=args.overwrite,
            )
        except (FileExistsError, CheckpointMismatchError) as exc:
            logger.error("%s", exc)
            return 2

    if args.no_judge:
        logger.info("Generation-only run (--no-judge); outputs at %s", args.outputs)
        return 0

    run_evaluation(
        outputs_path=args.outputs,
        judges_config=args.judges_config,
        verdicts_path=args.verdicts,
        max_priority=_JUDGE_PRESETS[args.judges],
        hardware_profile=args.hardware_config,
        meta_eval_path=args.meta_eval or None,
        reference_judge_id=args.reference_judge,
        judge_ids=args.judge_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
