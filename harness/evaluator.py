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

Note: consensus here is a placeholder majority vote. The real meta-evaluation
(inter-judge agreement, bias detection) is Sam's manual work -- see
``judges.judge_panel.aggregate_verdicts``.
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

from harness.hydrate import hydrate_weights
from harness.model_loader import ModelLoader
from harness.test_runner import TestRunner
from judges.factory import build_judges
from judges.judge_panel import EvalItem, JudgePanel, aggregate_verdicts

logger = logging.getLogger(__name__)

# Judge-subset presets (PRD "Decision 3: Judge Placement").
_JUDGE_PRESETS = {
    "all": None,      # every enabled judge
    "cheap": 1,       # priority <= 1 (remote API judges only)
    "pilot": 3,       # everything, but named for clarity in pilots
}


def run_evaluation(
    outputs_path: str,
    judges_config: str = "config/judges.yaml",
    verdicts_path: str = "results/model_verdicts.json",
    max_priority: Optional[int] = None,
    hardware_profile: str = "config/hardware_profile.yaml",
) -> str:
    """Judge previously-generated model outputs; write verdicts. Returns the path."""
    judges = build_judges(
        judges_config, max_priority=max_priority, hardware_profile=hardware_profile
    )
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
            model_output=row.get("output", ""),
            criteria=row.get("criteria", ""),
        )
        for row in rows
    ]
    results = panel.evaluate_batch(items)

    records = [
        {
            "test_id": row["test_id"],
            "model": row.get("model"),
            "verdicts": [v.to_dict() for v in result.verdicts],
            "consensus": aggregate_verdicts(result),
        }
        for row, result in zip(rows, results)
    ]
    with open(verdicts_path, "w", encoding="utf-8") as sink:
        json.dump({"verdicts": records}, sink, indent=2)
        sink.write("\n")

    logger.info("Wrote %d verdict row(s) to %s", len(records), verdicts_path)
    logger.info("Cost/latency summary: %s", panel.cost_summary())
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
        "--judges",
        default="all",
        choices=sorted(_JUDGE_PRESETS),
        help="Judge subset preset.",
    )
    parser.add_argument(
        "--prefer-engine", default="llamacpp", choices=["llamacpp", "ollama"]
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
        )

    if args.models:
        loader = ModelLoader(
            args.models_config,
            prefer_engine=args.prefer_engine,
            hardware_profile=args.hardware_config,
        )
        runner = TestRunner(args.test_suite, loader, results_path=args.outputs)
        # run_tests overwrites the outputs file with a fresh {"outputs": [...]}.
        runner.run_tests(args.models)

    if args.no_judge:
        logger.info("Generation-only run (--no-judge); outputs at %s", args.outputs)
        return 0

    run_evaluation(
        outputs_path=args.outputs,
        judges_config=args.judges_config,
        verdicts_path=args.verdicts,
        max_priority=_JUDGE_PRESETS[args.judges],
        hardware_profile=args.hardware_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
