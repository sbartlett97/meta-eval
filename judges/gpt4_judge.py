"""OpenAI judges (GPT-4o, GPT-5.x) via the OpenAI API, grading with rubric v2
(judges/rubric.py).

Reads ``OPENAI_API_KEY`` from the environment. The SDK is imported lazily.

The model is a dated snapshot (``gpt-4o-2024-11-20``), not the moving
``gpt-4o`` alias, so the judge cannot change between runs. The answer is
constrained to the rubric's JSON schema (``response_format`` json_schema,
strict). When the model declines to grade, OpenAI returns a ``refusal``
message instead of JSON; that is recorded as abstained with
``abstain_reason="judge_declined"``.

GPT-5.x models are reasoning models: they take ``reasoning_effort`` and
reject a non-default ``temperature``, so their config sets ``temperature:
null`` (not sent) and ``reasoning_effort``. Their ``max_completion_tokens``
also counts reasoning tokens, so it must be much larger than for GPT-4o.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from judges import rubric
from judges.base import Judge, NonRetryableJudgeError, Verdict
from judges.items import EvalItem

DEFAULT_MODEL = "gpt-4o-2024-11-20"


class GPT4Judge(Judge):
    provider = "openai"
    credential_env = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        judge_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = 0.0,
        reasoning_effort: Optional[str] = None,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.id = judge_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def has_credentials(self) -> bool:
        return bool(self._api_key)

    def describe(self) -> dict:
        return {
            **super().describe(),
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "rubric_version": rubric.RUBRIC_VERSION,
        }

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - env dependent
            raise NonRetryableJudgeError("openai SDK not installed (`pip install openai`).") from exc
        if not self._api_key:
            raise NonRetryableJudgeError("OPENAI_API_KEY is not set.")
        self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def _evaluate(self, item: EvalItem) -> Verdict:
        client = self._ensure_client()
        system, user, schema = rubric.render_prompt(item)

        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        start = time.monotonic()
        resp = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            **options,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": rubric.rubric_key(item), "schema": schema, "strict": True},
            },
        )
        latency = time.monotonic() - start

        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            verdict = rubric.declined_verdict(
                self.id, item, "judge_declined", choice.message.refusal
            )
        elif choice.finish_reason == "length":
            raise ValueError(
                f"judge stopped at max_tokens={self.max_tokens} before finishing its answer"
            )
        else:
            text = choice.message.content or ""
            verdict = rubric.build_verdict(self.id, item, json.loads(text), raw=text)
        verdict.latency_s = latency
        verdict.tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
        return verdict
