"""Criterion helpers and JSON parsing shared by the judges.

The judge prompt itself is rubric v2 (judges/rubric.py).
"""

from __future__ import annotations

import json
import re
from typing import Optional

# The suite's criteria strings are prefixed with a machine-readable directive
# for judges/heuristic_judge.py's deterministic dispatch (`expect:PASS ...` /
# `expect:FAIL ...`; the legacy `expect:refuse` / `expect:comply` spelling is
# also produced by hand-written suites). That token is NOT meant for the LLM
# judges: left in, "expect:FAIL The model must refuse ..." sits directly next
# to the instruction to output PASS/FAIL, telling the judge what to answer
# before it's read the output. Strip it before rendering into a prompt --
# HeuristicJudge reads the *raw* (undstripped) criteria string separately, so
# this has no effect on the deterministic dispatch.
_DIRECTIVE_PREFIX_RE = re.compile(r"^\s*expect:\s*(pass|fail|refuse|comply)\s*", re.IGNORECASE)


def strip_expect_directive(criteria: str) -> str:
    """The criterion without its leading expect: directive (judges never see it)."""
    return _DIRECTIVE_PREFIX_RE.sub("", criteria or "", count=1).strip()


def parse_json_from_response(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Tolerates the common failure modes of small local models: markdown code
    fences, leading/trailing prose, and single-object-with-trailing-junk.

    Raises:
        ValueError: if no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")

    candidate = _strip_code_fences(text)

    # Fast path: the whole thing is JSON.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first balanced {...} block.
    block = _first_json_object(candidate)
    if block is not None:
        return json.loads(block)

    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def _strip_code_fences(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced brace-delimited substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
