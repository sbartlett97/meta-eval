"""Test Runner (PRD v3.1).

For each model under evaluation, run every test in the suite and persist the raw
model output. Judging happens downstream (JudgePanel) -- keeping generation and
judging as separate passes means expensive test-model inference is done once and
can be re-judged by different panels without re-running the model.

Outputs are written to ``results/model_outputs.json`` as a single object with the
rows (one per test/model) nested under ``outputs``. Each row carries the
wall-clock generation time (``latency_s``); per-generation and per-model timing
summaries are also logged as the run proceeds.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable, List, Optional

from harness import provenance
from harness.test_suite import TestCase, load_test_suite

logger = logging.getLogger(__name__)


class TestRunner:
    def __init__(
        self,
        test_suite_path: str,
        model_loader,
        results_path: str = "results/model_outputs.json",
    ) -> None:
        """
        Args:
            test_suite_path: Path to a suite file (see test_suite.py).
            model_loader: A ``ModelLoader``-like object exposing ``load(id)`` ->
                object with ``generate(prompt) -> str``.
            results_path: Where to write the outputs JSON (``{"outputs": [...]}``).
        """
        self.test_suite_path = test_suite_path
        self.test_suite: List[TestCase] = load_test_suite(test_suite_path)
        self.model_loader = model_loader
        self.results_path = results_path

    def run_tests(
        self,
        models: Iterable[str],
        gen_kwargs: Optional[dict] = None,
        n_samples: int = 1,
    ) -> None:
        """Generate outputs for every (model, test) pair and persist them.

        Writes ``results_path`` once, overwriting any previous run: a single
        JSON object ``{"outputs": [...], "provenance": {...}}`` with one row
        per (test, model, sample).

        ``n_samples`` > 1 draws that many independent generations per (model,
        test) at the same sampling settings, each tagged with a
        ``sample_index`` -- a single non-deterministic draw otherwise turns
        "did the model pass" into a noisy one-shot Bernoulli estimate with no
        uncertainty measure (docs/critical-assessment.md #7). Downstream,
        ``judges.meta_eval.sample_variance`` reports the per-test pass-rate
        and standard error once the judged verdicts carry >1 sample.
        """
        gen_kwargs = gen_kwargs or {}
        os.makedirs(os.path.dirname(self.results_path) or ".", exist_ok=True)
        models = list(models)

        rows: List[dict] = []
        for model_id in models:
            logger.info("Loading model under test: %s", model_id)
            model = self.model_loader.load(model_id)
            latencies: List[float] = []
            for test in self.test_suite:
                for sample_index in range(n_samples):
                    row = self._run_one(model, model_id, test, gen_kwargs, sample_index)
                    latencies.append(row["latency_s"])
                    rows.append(row)
            _log_latency_summary(model_id, latencies)

        payload = {
            "outputs": rows,
            "provenance": provenance.collect(
                suite_path=self.test_suite_path, models=models, n_samples=n_samples
            ),
        }
        with open(self.results_path, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2)
            sink.write("\n")
        logger.info("Wrote %d output row(s) to %s", len(rows), self.results_path)

    def _run_one(
        self, model, model_id: str, test: TestCase, gen_kwargs: dict, sample_index: int = 0
    ) -> dict:
        start = time.monotonic()
        transcript = None
        try:
            if test.is_multi_turn:
                # Real conversation: turn n sees the model's own reply to turn
                # n-1, not just its raw text (see model_loader.generate_conversation).
                replies = model.generate_conversation(
                    test.turns, system_prompt=test.system_prompt, **gen_kwargs
                )
                transcript = _format_transcript(test.turns, replies)
                output = replies[-1] if replies else ""
            else:
                output = model.generate(
                    test.prompt, system_prompt=test.system_prompt, **gen_kwargs
                )
            error = None
        except Exception as exc:  # noqa: BLE001 -- one bad test shouldn't abort the run
            logger.warning("Generation failed (%s / %s): %s", model_id, test.id, exc)
            output, error = "", str(exc)
        latency_s = time.monotonic() - start
        logger.info(
            "Generated %s / %s in %.2fs%s",
            model_id,
            test.id,
            latency_s,
            " (error)" if error else "",
        )
        return {
            "test_id": test.id,
            "model": model_id,
            "sample_index": sample_index,
            "category": test.category,
            "domain": test.domain,
            "technique": test.technique,
            "severity": test.severity,
            "source": test.source,
            "criteria": test.criteria,
            "system_prompt": test.system_prompt,
            "metadata": test.metadata,
            "output": output,
            # Full turn-by-turn exchange for multi-turn cases (None for
            # single-turn) -- judges score this, not just the final reply, so
            # consistency-across-turns criteria are actually checkable.
            "transcript": transcript,
            "error": error,
            "latency_s": round(latency_s, 3),
        }


def _format_transcript(turns: List[str], replies: List[str]) -> str:
    lines: List[str] = []
    for turn, reply in zip(turns, replies):
        lines.append(f"User: {turn}")
        lines.append(f"Assistant: {reply}")
    return "\n".join(lines)


def _log_latency_summary(model_id: str, latencies: List[float]) -> None:
    """Log total / mean / max generation time for one model's run."""
    if not latencies:
        return
    total = sum(latencies)
    logger.info(
        "Generation timing for %s: %d prompts, total %.2fs, mean %.2fs, max %.2fs",
        model_id,
        len(latencies),
        total,
        total / len(latencies),
        max(latencies),
    )
