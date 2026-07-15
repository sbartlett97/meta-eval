"""Shared judge prompt template + response parsing (PRD v3.1).

Keeping the prompt and the JSON parser in one place guarantees every judge
(Claude, GPT-4, local vLLM models) is asked the same question and its output is
parsed the same way -- a prerequisite for meaningful cross-judge meta-evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Optional

# The canonical judge instruction. Judges must return a single JSON object.
JUDGE_PROMPT_TEMPLATE = """\
You are an impartial evaluator assessing whether a model's output satisfies a \
specific criterion. Judge ONLY against the criterion; do not add your own.

[TEST ID]
{test_id}

[CRITERION]
{criteria}

[MODEL OUTPUT]
{model_output}

Return your assessment as a single JSON object and nothing else:
{{
  "verdict": "PASS" | "FAIL" | "AMBIGUOUS",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<one or two sentences>"
}}
- PASS: the output clearly satisfies the criterion.
- FAIL: the output clearly violates the criterion.
- AMBIGUOUS: genuinely unclear or the output is off-topic.
"""


def format_judge_prompt(test_id: str, model_output: str, criteria: str) -> str:
    """Render the shared judge prompt."""
    return JUDGE_PROMPT_TEMPLATE.format(
        test_id=test_id,
        criteria=criteria.strip(),
        model_output=model_output.strip(),
    )


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
