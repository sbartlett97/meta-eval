"""Test Runner (PRD v3.1).

For each model under evaluation, run every test in the suite and persist the raw
model output. Judging happens downstream (JudgePanel) -- keeping generation and
judging as separate passes means expensive test-model inference is done once and
can be re-judged by different panels without re-running the model.

Outputs are appended to ``results/model_outputs.jsonl`` (one row per test/model).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List, Optional

from harness.test_suite import TestCase, load_test_suite

logger = logging.getLogger(__name__)


class TestRunner:
    def __init__(
        self,
        test_suite_path: str,
        model_loader,
        results_path: str = "results/model_outputs.jsonl",
    ) -> None:
        """
        Args:
            test_suite_path: Path to a ``.jsonl`` suite (see test_suite.py).
            model_loader: A ``ModelLoader``-like object exposing ``load(id)`` ->
                object with ``generate(prompt) -> str``.
            results_path: Where to append output rows.
        """
        self.test_suite: List[TestCase] = load_test_suite(test_suite_path)
        self.model_loader = model_loader
        self.results_path = results_path

    def run_tests(self, models: Iterable[str], gen_kwargs: Optional[dict] = None) -> None:
        """Generate outputs for every (model, test) pair and persist them."""
        gen_kwargs = gen_kwargs or {}
        os.makedirs(os.path.dirname(self.results_path) or ".", exist_ok=True)

        with open(self.results_path, "a", encoding="utf-8") as sink:
            for model_id in models:
                logger.info("Loading model under test: %s", model_id)
                model = self.model_loader.load(model_id)
                for test in self.test_suite:
                    row = self._run_one(model, model_id, test, gen_kwargs)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
        logger.info("Wrote outputs to %s", self.results_path)

    def _run_one(self, model, model_id: str, test: TestCase, gen_kwargs: dict) -> dict:
        try:
            output = model.generate(test.prompt, **gen_kwargs)
            error = None
        except Exception as exc:  # noqa: BLE001 -- one bad test shouldn't abort the run
            logger.warning("Generation failed (%s / %s): %s", model_id, test.id, exc)
            output, error = "", str(exc)
        return {
            "test_id": test.id,
            "model": model_id,
            "category": test.category,
            "criteria": test.criteria,
            "output": output,
            "error": error,
        }
