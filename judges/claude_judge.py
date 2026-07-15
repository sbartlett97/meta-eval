"""Claude judge via the Anthropic API (PRD v3.1).

Reads ``ANTHROPIC_API_KEY`` from the environment. The SDK is imported lazily so
the rest of the harness (and the test suite) works without the dependency
installed.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from judges.base import Judge, Verdict
from judges.prompts import format_judge_prompt, parse_json_from_response


class ClaudeJudge(Judge):
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        judge_id: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.id = judge_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazily constructed

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "anthropic SDK not installed (`pip install anthropic`)."
            ) from exc
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        client = self._ensure_client()
        prompt = format_judge_prompt(test_id, model_output, criteria)

        start = time.monotonic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.monotonic() - start

        text = "".join(block.text for block in resp.content if block.type == "text")
        tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
        parsed = parse_json_from_response(text)

        return Verdict(
            verdict=parsed.get("verdict", "AMBIGUOUS"),
            confidence=parsed.get("confidence", 0.0),
            judge_id=self.id,
            test_id=test_id,
            rationale=parsed.get("rationale", ""),
            latency_s=latency,
            tokens=tokens,
            raw=text,
        )
