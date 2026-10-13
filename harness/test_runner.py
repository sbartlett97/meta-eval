"""Test Runner (PRD v3.1).

For each model under evaluation, run every test in the suite and persist the raw
model output. Judging happens downstream (JudgePanel) -- keeping generation and
judging as separate passes means expensive test-model inference is done once and
can be re-judged by different panels without re-running the model.

Outputs are written to ``results/model_outputs.json`` as a single object with the
rows (one per test/model) nested under ``outputs``. Each row carries the
wall-clock generation time (``latency_s``); per-generation and per-model timing
summaries are also logged as the run proceeds.

While a run is in progress, each row is also appended to
``<results_path>.partial.jsonl`` so a crash loses at most one row; see
``TestRunner.run_tests`` for resume and overwrite rules.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

from harness import provenance
from harness.test_suite import TestCase, load_test_suite

logger = logging.getLogger(__name__)


class CheckpointMismatchError(ValueError):
    """A resume checkpoint was written for a different suite or ``n_samples``."""


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

    @property
    def checkpoint_path(self) -> str:
        """Append-only JSONL checkpoint written while a run is in progress."""
        return self.results_path + ".partial.jsonl"

    def run_tests(
        self,
        models: Iterable[str],
        gen_kwargs: Optional[dict] = None,
        n_samples: int = 1,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Generate outputs for every (model, test) pair and persist them.

        The result is a single JSON object ``{"outputs": [...], "provenance":
        {...}}`` at ``results_path``, one row per (test, model, sample).

        Each row is also appended to ``checkpoint_path`` (and flushed to disk)
        as soon as it is generated, so a crash part-way through a multi-hour
        run loses at most the row in progress. ``resume=True`` reloads that
        checkpoint, skips every (model, test_id, sample_index) that already
        succeeded, and re-runs the ones that errored. The checkpoint is
        deleted only after the final JSON is written.

        To protect finished or interrupted work, a fresh run (``resume=False``)
        refuses to start if a checkpoint exists, or if ``results_path`` exists
        and ``overwrite`` is False.

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
        header = {
            "suite_sha256": provenance.file_sha256(self.test_suite_path),
            "n_samples": n_samples,
        }

        done: Dict[Tuple[str, str, int], dict] = {}
        if resume:
            done = self._load_checkpoint(header)
        elif os.path.exists(self.checkpoint_path):
            raise FileExistsError(
                f"{self.checkpoint_path} holds an interrupted run. Resume it "
                "(--resume) or delete it before starting a new run."
            )
        elif os.path.exists(self.results_path) and not overwrite:
            raise FileExistsError(
                f"{self.results_path} already exists. Pass --overwrite to replace it."
            )
        if not os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "w", encoding="utf-8") as sink:
                sink.write(json.dumps({"header": header}) + "\n")

        rows: List[dict] = []
        with open(self.checkpoint_path, "a", encoding="utf-8") as checkpoint:
            for model_id in models:
                keys = [
                    (model_id, test, i) for test in self.test_suite for i in range(n_samples)
                ]
                if all((m, t.id, i) in done for m, t, i in keys):
                    logger.info("Skipping %s: all rows already in the checkpoint", model_id)
                    rows.extend(done[(m, t.id, i)] for m, t, i in keys)
                    continue
                logger.info("Loading model under test: %s", model_id)
                model = self.model_loader.load(model_id)
                latencies: List[float] = []
                for _, test, sample_index in keys:
                    row = done.get((model_id, test.id, sample_index))
                    if row is None:
                        row = self._run_one(model, model_id, test, gen_kwargs, sample_index)
                        latencies.append(row["latency_s"])
                        checkpoint.write(json.dumps(row) + "\n")
                        checkpoint.flush()
                        os.fsync(checkpoint.fileno())
                    rows.append(row)
                _log_latency_summary(model_id, latencies)

        payload = {
            "outputs": rows,
            "provenance": provenance.collect(
                suite_path=self.test_suite_path, models=models, n_samples=n_samples
            ),
        }
        tmp_path = self.results_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2)
            sink.write("\n")
        os.replace(tmp_path, self.results_path)
        os.remove(self.checkpoint_path)
        logger.info("Wrote %d output row(s) to %s", len(rows), self.results_path)

    def _load_checkpoint(self, header: dict) -> Dict[Tuple[str, str, int], dict]:
        """Successful rows from an interrupted run, keyed by (model, test_id, sample_index).

        Refuses a checkpoint written for a different suite or ``n_samples``:
        mixing rows from two configurations would silently corrupt the run.
        A truncated last line (a crash mid-write) is ignored.
        """
        if not os.path.exists(self.checkpoint_path):
            logger.info("No checkpoint at %s; starting from scratch", self.checkpoint_path)
            return {}
        with open(self.checkpoint_path, encoding="utf-8") as source:
            lines = source.read().splitlines()
        saved = json.loads(lines[0]).get("header") if lines else None
        if saved != header:
            raise CheckpointMismatchError(
                f"{self.checkpoint_path} was written for {saved}, but this run is "
                f"{header}. Delete it to start over."
            )
        done: Dict[Tuple[str, str, int], dict] = {}
        for line in lines[1:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring a truncated checkpoint line")
                continue
            if row.get("error") is None:
                done[(row["model"], row["test_id"], row["sample_index"])] = row
        logger.info("Resuming: %d successful row(s) in the checkpoint", len(done))
        return done

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
            # What the model was asked: every user turn (a single-turn case is
            # one turn) plus the system prompt. Judges need these to grade the
            # response in context (validation gap C1).
            "turns": list(test.turns),
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
