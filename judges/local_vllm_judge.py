"""Base class for local judges backed by an **in-process** vLLM engine.

Mistral and Llama judges differ only in id + default checkpoint, so they share
this implementation. Unlike the earlier design, calls no longer go over HTTP to a
separately-started ``vllm serve`` process: the checkpoint is loaded in-process via
:mod:`harness.vllm_engine` and inference runs in-line. The engine is shared
process-wide, so a judge and a test model that use the same checkpoint reuse a
single loaded copy of the weights.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from harness.vllm_engine import get_engine
from judges.base import Judge, Verdict
from judges.prompts import format_judge_prompt, parse_json_from_response

logger = logging.getLogger(__name__)


class LocalVLLMJudge(Judge):
    """Judge that runs a local checkpoint in-process via vLLM.

    Args:
        model_id: HuggingFace checkpoint id loaded in-process (e.g.
            ``unsloth/Mistral-7B-v0.3-GGUF``).
        judge_id: Stable id for this judge in results/config.
        engine_kwargs: ``vllm.LLM`` constructor kwargs (from the hardware
            profile). Optional; defaults apply when omitted.
        max_tokens: Generation cap for the judge's JSON verdict.
        temperature: Sampling temperature.
    """

    provider = "local"

    def __init__(
        self,
        model_id: str,
        judge_id: str,
        engine_kwargs: Optional[Dict] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.id = judge_id
        self.engine_kwargs = dict(engine_kwargs or {})
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        prompt = format_judge_prompt(test_id, model_output, criteria)
        engine = get_engine(self.model_id, self.engine_kwargs)

        start = time.monotonic()
        text = engine.generate(
            prompt, max_tokens=self.max_tokens, temperature=self.temperature
        )
        latency = time.monotonic() - start

        parsed = parse_json_from_response(text)
        return Verdict(
            verdict=parsed.get("verdict", "AMBIGUOUS"),
            confidence=parsed.get("confidence", 0.0),
            judge_id=self.id,
            test_id=test_id,
            rationale=parsed.get("rationale", ""),
            latency_s=latency,
            tokens=None,
            raw=text,
        )
