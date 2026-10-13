"""Split a reasoning model's reply into its thinking trace and its final answer.

Reasoning models (Qwen3.5 derivatives and similar) think before they answer.
Their chat template usually opens the ``<think>`` block itself, so the reply
the harness receives is ``<trace></think><answer>`` with no opening tag. A
reply cut off by ``max_tokens`` before ``</think>`` has no tag at all: all of
it is unfinished trace. Judging that raw text scores thinking-aloud instead of
the model's answer (validation gap C7).

Policy (decision D2): judges score the final answer; the trace is kept apart
so it can be checked separately for harmful content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_OPEN_TAG = re.compile(r"^\s*<(think|thinking)>", re.IGNORECASE)
_CLOSE_TAG = re.compile(r"</(think|thinking)>", re.IGNORECASE)


@dataclass(frozen=True)
class SplitReply:
    #: The thinking trace, or None if the reply has none.
    reasoning: Optional[str]
    #: The final answer ("" when the reasoning never finished).
    answer: str
    #: False when the trace was cut off before the answer started; None when
    #: the reply has no reasoning at all.
    reasoning_complete: Optional[bool]


def split_reasoning(text: str, template_opens_think: bool = False) -> SplitReply:
    """Split ``text`` into trace and answer.

    Args:
        text: The raw reply.
        template_opens_think: True for models whose chat template opens the
            think block (``serving.reasoning: true``). For these, a reply with
            no closing tag is an unfinished trace, not an answer.
    """
    text = text or ""
    closes = list(_CLOSE_TAG.finditer(text))
    if closes:
        last = closes[-1]
        trace = _OPEN_TAG.sub("", text[: last.start()], count=1).strip()
        return SplitReply(trace, text[last.end():].strip(), True)
    if _OPEN_TAG.match(text):
        return SplitReply(_OPEN_TAG.sub("", text, count=1).strip(), "", False)
    if template_opens_think and text.strip():
        return SplitReply(text.strip(), "", False)
    return SplitReply(None, text, None)
