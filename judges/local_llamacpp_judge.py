"""Local judge backed by an **in-process** llama.cpp (GGUF) engine.

The llama.cpp counterpart to :mod:`judges.local_vllm_judge`. It loads a specific
GGUF quant in-process via :mod:`harness.llamacpp_engine` and runs inference
in-line — no server, no port. Because the engine cache loads models sequentially
(default: one resident at a time), a llama.cpp judge does not permanently hold
its weights the way an in-process vLLM judge does; they are freed when another
model needs to load.

Unlike the Mistral/Llama vLLM judges, there is no per-model subclass: a GGUF
judge is fully described by its ``(repo_id, gguf_file)`` pair from
``config/judges.yaml``, so one class serves every checkpoint.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from harness.llamacpp_engine import get_engine
from judges.base import Judge, Verdict
from judges.prompts import format_judge_prompt, parse_json_from_response

logger = logging.getLogger(__name__)


class LocalLlamaCppJudge(Judge):
    """Judge that runs a local GGUF checkpoint in-process via llama.cpp.

    Args:
        model_id: HuggingFace repo id holding the GGUF (e.g.
            ``unsloth/Mistral-7B-v0.3-GGUF``).
        judge_id: Stable id for this judge in results/config.
        gguf_file: Filename glob selecting one quant in the repo
            (e.g. ``"*Q4_K_M.gguf"``). Passed straight to
            ``Llama.from_pretrained(filename=...)``.
        engine_kwargs: ``llama_cpp.Llama`` constructor kwargs (from the hardware
            profile). Optional; defaults apply when omitted.
        max_tokens: Generation cap for the judge's JSON verdict.
        temperature: Sampling temperature.
    """

    provider = "local"

    def __init__(
        self,
        model_id: str,
        judge_id: str,
        gguf_file: Optional[str] = None,
        engine_kwargs: Optional[Dict] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.id = judge_id
        self.gguf_file = gguf_file
        self.engine_kwargs = dict(engine_kwargs or {})
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        prompt = format_judge_prompt(test_id, model_output, criteria)
        engine = get_engine(
            repo_id=self.model_id,
            filename=self.gguf_file,
            engine_kwargs=self.engine_kwargs,
        )

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
