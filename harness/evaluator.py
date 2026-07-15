"""Evaluation orchestrator / CLI entrypoint (PRD v3.1).

Ties the pieces together:

    1. (optional) generate model outputs via ``TestRunner`` if they don't exist.
    2. Build the judge panel from ``config/judges.yaml``.
    3. Run every judge over every model output.
    4. Persist per-judge verdicts + a naive consensus to
       ``results/model_verdicts.jsonl``.

Usage (from the PRD setup section):
    python harness/evaluator.py --models mistral-7b-4bit --judges all
    python harness/evaluator.py --outputs results/model_outputs.jsonl --judges cheap

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

from harness.model_loader import ModelLoader
from harness.test_runner import TestRunner
from judges.factory import build_judges
from judges.judge_panel import JudgePanel, aggregate_verdicts

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
    verdicts_path: str = "results/model_verdicts.jsonl",
    max_priority: Optional[int] = None,
) -> str:
    """Judge previously-generated model outputs; write verdicts. Returns the path."""
    judges = build_judges(judges_config, max_priority=max_priority)
    panel = JudgePanel(judges)
    logger.info("Judge panel: %s", [j.id for j in judges])

    os.makedirs(os.path.dirname(verdicts_path) or ".", exist_ok=True)
    n = 0
    with open(outputs_path, "r", encoding="utf-8") as src, open(
        verdicts_path, "w", encoding="utf-8"
    ) as sink:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result = panel.evaluate(
                test_id=row["test_id"],
                model_output=row.get("output", ""),
                criteria=row.get("criteria", ""),
            )
            record = {
                "test_id": row["test_id"],
                "model": row.get("model"),
                "verdicts": [v.to_dict() for v in result.verdicts],
                "consensus": aggregate_verdicts(result),
            }
            sink.write(json.dumps(record) + "\n")
            n += 1

    logger.info("Wrote %d verdict rows to %s", n, verdicts_path)
    logger.info("Cost/latency summary: %s", panel.cost_summary())
    return verdicts_path


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval harness.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model ids to generate outputs for first (from config/models.yaml).",
    )
    parser.add_argument("--test-suite", default="data/test_suite_v1.jsonl")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--judges-config", default="config/judges.yaml")
    parser.add_argument(
        "--outputs",
        default="results/model_outputs.jsonl",
        help="Model outputs to judge (generated first if --models is given).",
    )
    parser.add_argument("--verdicts", default="results/model_verdicts.jsonl")
    parser.add_argument(
        "--judges",
        default="all",
        choices=sorted(_JUDGE_PRESETS),
        help="Judge subset preset.",
    )
    parser.add_argument("--prefer-engine", default="ollama", choices=["ollama", "vllm"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.models:
        loader = ModelLoader(args.models_config, prefer_engine=args.prefer_engine)
        runner = TestRunner(args.test_suite, loader, results_path=args.outputs)
        # Fresh generation run: start the outputs file clean.
        open(args.outputs, "w").close()
        runner.run_tests(args.models)

    run_evaluation(
        outputs_path=args.outputs,
        judges_config=args.judges_config,
        verdicts_path=args.verdicts,
        max_priority=_JUDGE_PRESETS[args.judges],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
