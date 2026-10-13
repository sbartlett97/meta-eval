"""Claude judge via the Anthropic API, grading with rubric v2 (judges/rubric.py).

Reads ``ANTHROPIC_API_KEY`` from the environment. The SDK is imported lazily so
the rest of the harness (and the test suite) works without the dependency
installed.

Request shape, for current Claude models (Sonnet 5.5 / Opus 5.5):

* No ``temperature``: these models reject a non-default value with HTTP 400.
* ``output_config.effort`` is set explicitly, and ``max_tokens`` leaves room
  for adaptive thinking before the JSON answer.
* ``output_config.format`` constrains the answer to the rubric's JSON schema.
* Server-side refusal **fallbacks are deliberately OFF** (decision D4). A
  fallback silently answers with a different model, which breaks the judge's
  identity and the meta-evaluation. When Claude declines to grade, the verdict
  is recorded as abstained with ``abstain_reason="judge_declined:<category>"``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from judges import rubric
from judges.base import Judge, NonRetryableJudgeError, Verdict
from judges.items import EvalItem

DEFAULT_MODEL = "claude-sonnet-5-5"


class ClaudeJudge(Judge):
    provider = "anthropic"
    credential_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        judge_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        effort: str = "low",
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.id = judge_id
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazily constructed

    def has_credentials(self) -> bool:
        return bool(self._api_key)

    def describe(self) -> dict:
        return {
            **super().describe(),
            "model": self.model,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "temperature": "not sent",
            "fallbacks": "off",
            "rubric_version": rubric.RUBRIC_VERSION,
        }

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise NonRetryableJudgeError(
                "anthropic SDK not installed (`pip install anthropic`)."
            ) from exc
        if not self._api_key:
            raise NonRetryableJudgeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _evaluate(self, item: EvalItem) -> Verdict:
        client = self._ensure_client()
        system, user, schema = rubric.render_prompt(item)

        start = time.monotonic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        )
        latency = time.monotonic() - start

        # Check why the turn ended before reading any content.
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            verdict = rubric.declined_verdict(
                self.id, item, f"judge_declined:{category}",
                getattr(details, "explanation", None) or "",
            )
        elif resp.stop_reason == "max_tokens":
            raise ValueError(
                f"judge stopped at max_tokens={self.max_tokens} before finishing its answer"
            )
        else:
            text = next((b.text for b in resp.content if b.type == "text"), "")
            verdict = rubric.build_verdict(self.id, item, json.loads(text), raw=text)
        verdict.latency_s = latency
        verdict.tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
        return verdict
