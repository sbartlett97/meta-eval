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
import logging
import os
import time
from typing import List, Optional, Sequence

from judges import rubric
from judges.base import Judge, NonRetryableJudgeError, Verdict
from judges.items import EvalItem

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"


class ClaudeJudge(Judge):
    provider = "anthropic"
    credential_env = "ANTHROPIC_API_KEY"
    supports_batch = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        judge_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        effort: str = "low",
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        use_batch: bool = False,
        batch_poll_s: float = 30.0,
        batch_max_wait_s: float = 24 * 3600,
    ) -> None:
        super().__init__()
        self.model = model
        self.id = judge_id
        self.max_tokens = max_tokens
        self.effort = effort
        self.use_batch = use_batch
        self.batch_poll_s = batch_poll_s
        self.batch_max_wait_s = batch_max_wait_s
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
            "use_batch": self.use_batch,
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

    def _request(self, item: EvalItem) -> dict:
        """Messages API parameters for one item (shared by both paths)."""
        system, user, schema = rubric.render_prompt(item)
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        }

    def _verdict_from_message(self, item: EvalItem, resp) -> Verdict:
        """Turn one Message into a Verdict. Checks why the turn ended before
        reading any content."""
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
        verdict.tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
        return verdict

    def _evaluate(self, item: EvalItem) -> Verdict:
        client = self._ensure_client()
        start = time.monotonic()
        resp = client.messages.create(**self._request(item))
        verdict = self._verdict_from_message(item, resp)
        verdict.latency_s = time.monotonic() - start
        return verdict

    def evaluate_many(self, items: Sequence[EvalItem]) -> List[Verdict]:
        """With ``use_batch``, grade all items in one Message Batch (about half
        the price, results within 24 h). Otherwise one call per item.

        Results are matched by ``custom_id``, never by position (a batch
        returns them in any order). A request that errored, expired or was
        cancelled in the batch, or whose answer is invalid, is re-graded with
        a normal call, which has the usual retries and error recording.
        """
        items = list(items)
        if not self.use_batch or not items:
            return super().evaluate_many(items)
        try:
            client = self._ensure_client()
        except NonRetryableJudgeError:
            return super().evaluate_many(items)  # records the error on every item

        batch = client.messages.batches.create(requests=[
            {"custom_id": f"item-{n}", "params": self._request(item)}
            for n, item in enumerate(items)
        ])
        logger.info("Judge %s: submitted batch %s (%d requests)", self.id, batch.id, len(items))
        start = time.monotonic()
        while batch.processing_status != "ended":
            if time.monotonic() - start > self.batch_max_wait_s:
                raise TimeoutError(
                    f"Batch {batch.id} did not end within {self.batch_max_wait_s}s; "
                    "its results stay retrievable for 29 days."
                )
            time.sleep(self.batch_poll_s)
            batch = client.messages.batches.retrieve(batch.id)
        per_item_s = (time.monotonic() - start) / len(items)
        results = {r.custom_id: r.result for r in client.messages.batches.results(batch.id)}

        verdicts: List[Optional[Verdict]] = [None] * len(items)
        for n, item in enumerate(items):
            result = results.get(f"item-{n}")
            if result is None or result.type != "succeeded":
                continue
            try:
                verdict = self._verdict_from_message(item, result.message)
            except Exception as exc:  # noqa: BLE001 -- re-graded below
                logger.warning("Judge %s: invalid batch answer for %s (%s); re-grading",
                               self.id, item.test_id, exc)
                continue
            verdict.latency_s = per_item_s
            self._log_call(verdict)
            verdicts[n] = verdict
        missing = [n for n, v in enumerate(verdicts) if v is None]
        if missing:
            logger.warning("Judge %s: re-grading %d item(s) outside the batch",
                           self.id, len(missing))
        for n in missing:
            verdicts[n] = self.evaluate(items[n])
        return verdicts  # type: ignore[return-value]
