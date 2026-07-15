"""Base class for local judges served via a vLLM OpenAI-compatible server.

Mistral (:8000) and Llama (:8001) differ only in id + port, so they share this
implementation. All calls go over HTTP -- the same interface as the remote
judges (PRD "Judge Calls (All via API)").
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from judges.base import Judge, Verdict
from judges.prompts import format_judge_prompt, parse_json_from_response

logger = logging.getLogger(__name__)


class LocalVLLMJudge(Judge):
    """Judge that calls a local vLLM server's ``/v1/completions`` endpoint.

    Args:
        vllm_port: Port the target model is served on.
        model_id: Checkpoint id passed through to vLLM (``model`` field).
        judge_id: Stable id for this judge in results/config.
        timeout_s: Per-request timeout.
        max_retries: Retries on timeout / connection error.
        retry_backoff_s: Base backoff (linear) between retries.
    """

    provider = "local"

    def __init__(
        self,
        vllm_port: int,
        model_id: str,
        judge_id: str,
        timeout_s: int = 30,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.base_url = f"http://localhost:{vllm_port}"
        self.model_id = model_id
        self.id = judge_id
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        prompt = format_judge_prompt(test_id, model_output, criteria)
        start = time.monotonic()
        payload = self._post_with_retries(prompt)
        latency = time.monotonic() - start

        choice = payload["choices"][0]["text"]
        tokens = payload.get("usage", {}).get("completion_tokens")
        parsed = parse_json_from_response(choice)

        return Verdict(
            verdict=parsed.get("verdict", "AMBIGUOUS"),
            confidence=parsed.get("confidence", 0.0),
            judge_id=self.id,
            test_id=test_id,
            rationale=parsed.get("rationale", ""),
            latency_s=latency,
            tokens=tokens,
            raw=choice,
        )

    def _post_with_retries(self, prompt: str) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model_id,
                        "prompt": prompt,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff_s * (attempt + 1)
                    logger.warning(
                        "%s call failed (%s); retry %d/%d in %.1fs",
                        self.id,
                        exc.__class__.__name__,
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"{self.id}: vLLM call failed after retries") from last_exc
