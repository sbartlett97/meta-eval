"""Deterministic, network-free checks for mechanically-verifiable criteria.

docs/critical-assessment.md #2.6: sending "is this valid JSON" through an LLM
judge costs real money/latency and adds a real chance of a wrong verdict, for
a question Python can answer exactly and for free. These checks back
``HeuristicJudge``'s ``format-adherence`` dispatch (see
``judges/heuristic_judge.py``): for a case whose ``metadata.check`` names one
of these, this *is* the check, not a heuristic proxy for one.

Every function returns ``(ok, note)`` -- ``note`` is folded into the verdict's
``rationale`` so a human reading results/model_verdicts.json can see exactly
what was checked without re-running anything.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
from typing import Iterable, List, Tuple


def _strip_code_fences(text: str) -> str:
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    return (fenced.group(1) if fenced else text).strip()


def check_json(text: str, required_keys: Iterable[str] = ()) -> Tuple[bool, str]:
    candidate = _strip_code_fences(text)
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return False, "output is not valid JSON"
    if not isinstance(obj, dict):
        return False, "valid JSON but not a JSON object"
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return False, f"valid JSON object but missing key(s): {missing}"
    return True, "valid JSON object with all required keys"


# Sentence boundary: punctuation followed by whitespace and a capital/quote/digit.
# Not a full NLP sentence tokenizer -- good enough for the short, plain-prose
# outputs this check is applied to; genuinely ambiguous prose should go to an
# LLM judge instead (set metadata.check to something else, or omit it).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def check_sentence_count(text: str, n: int) -> Tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return False, "empty output"
    sentences = [s for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]
    ok = len(sentences) == n
    return ok, f"found {len(sentences)} sentence(s), expected {n}"


def check_markdown_structure(text: str, min_bullets: int = 2) -> Tuple[bool, str]:
    has_h1 = bool(re.search(r"(?m)^#\s+\S", text))
    has_h2 = bool(re.search(r"(?m)^##\s+\S", text))
    bullets = len(re.findall(r"(?m)^\s*[-*]\s+\S", text))
    ok = has_h1 and has_h2 and bullets >= min_bullets
    return ok, f"h1={has_h1} h2={has_h2} bullets={bullets} (need >= {min_bullets})"


def check_csv(text: str, headers: List[str]) -> Tuple[bool, str]:
    candidate = _strip_code_fences(text)
    try:
        rows = [r for r in csv.reader(io.StringIO(candidate)) if any(c.strip() for c in r)]
    except csv.Error as exc:
        return False, f"not parseable as CSV: {exc}"
    if not rows:
        return False, "no CSV rows found"
    actual_header = [c.strip() for c in rows[0]]
    ok = actual_header == list(headers) and len(rows) > 1
    return ok, f"header={actual_header} (want {list(headers)}), {len(rows) - 1} data row(s)"


def check_python_syntax(text: str, must_contain: str = "") -> Tuple[bool, str]:
    candidate = _strip_code_fences(text)
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        return False, f"invalid Python syntax: {exc}"
    if must_contain and must_contain not in candidate:
        return False, f"valid Python but missing required text {must_contain!r}"
    suffix = f" containing {must_contain!r}" if must_contain else ""
    return True, "valid Python syntax" + suffix


def check_sections_present(
    text: str, required: Iterable[str], case_sensitive: bool = False
) -> Tuple[bool, str]:
    haystack = text if case_sensitive else text.lower()
    missing = [r for r in required if (r if case_sensitive else r.lower()) not in haystack]
    return (not missing), (
        "all required section(s) present" if not missing else f"missing section(s): {missing}"
    )


def check_email_structure(text: str, min_paragraphs: int = 3) -> Tuple[bool, str]:
    has_subject = bool(re.search(r"(?im)^subject\s*:", text))
    has_to = bool(re.search(r"(?im)^to\s*:", text))
    body_split = re.split(r"(?im)^body\s*:", text, maxsplit=1)
    body_text = body_split[1] if len(body_split) > 1 else text
    paragraphs = [p for p in re.split(r"\n\s*\n", body_text.strip()) if p.strip()]
    ok = has_subject and has_to and len(paragraphs) >= min_paragraphs
    return ok, f"subject={has_subject} to={has_to} paragraphs={len(paragraphs)} (need >= {min_paragraphs})"


#: metadata.check -> (fn, kwargs-from-metadata). Each fn's kwargs are read out
#: of the test case's ``metadata`` dict by HeuristicJudge; see
#: data/test_suite_v1.json's format-adherence entries for the shape.
CHECKS = {
    "json": lambda text, meta: check_json(text, meta.get("required_keys", ())),
    "sentence_count": lambda text, meta: check_sentence_count(text, meta.get("n", 1)),
    "markdown": lambda text, meta: check_markdown_structure(text, meta.get("min_bullets", 2)),
    "csv": lambda text, meta: check_csv(text, meta.get("headers", [])),
    "python": lambda text, meta: check_python_syntax(text, meta.get("must_contain", "")),
    "sections": lambda text, meta: check_sections_present(text, meta.get("required", [])),
    "email": lambda text, meta: check_email_structure(text, meta.get("min_paragraphs", 3)),
}
