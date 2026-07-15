"""GPT-4o judge via the OpenAI API (PRD v3.1).

Reads ``OPENAI_API_KEY`` from the environment. The SDK is imported lazily.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from judges.base import Judge, Verdict
from judges.prompts import format_judge_prompt, parse_json_from_response


class GPT4Judge(Judge):
    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        judge_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.id = judge_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("openai SDK not installed (`pip install openai`).") from exc
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _evaluate(self, test_id: str, model_output: str, criteria: str) -> Verdict:
        client = self._ensure_client()
        prompt = format_judge_prompt(test_id, model_output, criteria)

        start = time.monotonic()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.monotonic() - start

        text = resp.choices[0].message.content or ""
        tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
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
