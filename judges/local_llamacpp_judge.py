"""Local judge backed by an **in-process** llama.cpp (GGUF) engine.

It loads a specific GGUF quant in-process via :mod:`harness.llamacpp_engine` and
runs inference in-line — no server, no port. Because the engine cache loads
models sequentially (default: one resident at a time), a local judge does not
permanently hold its weights; they are freed when another model needs to load.

There is no per-model subclass: a GGUF judge is fully described by its
``(repo_id, gguf_file)`` pair from ``config/judges.yaml``, so one class serves
every checkpoint.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

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
        gguf_file: The GGUF file to load from the repo. Normally the **exact**
            file name (e.g. ``"model-Q4_K_M.gguf"``), as in llama.cpp's
            ``Llama.from_pretrained`` snippet; a glob (``"*Q4_K_M.gguf"``) is also
            accepted. Passed straight to ``from_pretrained(filename=...)``.
        additional_files: Extra files to fetch alongside it (e.g. the remaining
            shards of a split GGUF).
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
        additional_files: Optional[Sequence[str]] = None,
        engine_kwargs: Optional[Dict] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.id = judge_id
        self.gguf_file = gguf_file
        self.additional_files: List[str] = list(additional_files or [])
        self.engine_kwargs = dict(engine_kwargs or {})
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        prompt = format_judge_prompt(test_id, model_output, criteria)
        engine = get_engine(
            repo_id=self.model_id,
            filename=self.gguf_file,
            additional_files=self.additional_files,
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
