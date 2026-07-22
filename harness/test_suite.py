"""Test-suite schema + loader (PRD v3.1).

The *content* of the 15-test behavioral-invariance suite is Sam's manual work
(PRD: "Design 15-test suite (your manual work)"). This module defines the schema
and a loader so the runner has a stable contract to build against; a tiny example
suite lives at ``data/test_suite_v1.json``.

The suite is a single JSON object with the cases nested under ``evals`` (easy to
hand-edit)::

    {
      "evals": [
        {
          "id": "refuse-weapon-synthesis-01",   # unique test id
          "prompt": "...",                        # sent to the model under test
          "criteria": "expect:refuse ...",        # judged against (see prompts.py)
          "category": "safety-refusal",           # optional grouping
          "metadata": { ... }                      # optional, free-form
        }
      ]
    }

Any other top-level keys (e.g. ``_comment``) are ignored, so you can keep notes
in the file. A bare top-level JSON array is also accepted. Legacy ``.jsonl``
suites (one JSON object per line, ``#`` comments allowed) still load, dispatched
on the file extension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple


@dataclass
class TestCase:
    id: str
    prompt: str
    criteria: str
    category: str = "uncategorized"
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: Dict) -> "TestCase":
        missing = [k for k in ("id", "prompt", "criteria") if k not in obj]
        if missing:
            raise ValueError(f"Test case missing required fields {missing}: {obj}")
        return cls(
            id=obj["id"],
            prompt=obj["prompt"],
            criteria=obj["criteria"],
            category=obj.get("category", "uncategorized"),
            metadata=obj.get("metadata", {}),
        )


def load_test_suite(path: str) -> List[TestCase]:
    """Load a test suite from a ``.json`` (``evals``-wrapped) or ``.jsonl`` file.

    Raises:
        ValueError: on malformed JSON, a missing ``evals`` array, duplicate ids,
            or a case missing required fields (with the offending entry located).
    """
    entries = _read_jsonl(path) if path.endswith(".jsonl") else _read_json(path)

    tests: List[TestCase] = []
    seen: set[str] = set()
    for label, obj in entries:
        try:
            case = TestCase.from_dict(obj)
        except ValueError as exc:
            raise ValueError(f"{path} ({label}): {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{path} ({label}): duplicate test id {case.id!r}")
        seen.add(case.id)
        tests.append(case)
    return tests


def _read_json(path: str) -> List[Tuple[str, Dict]]:
    """Parse a JSON suite: a ``{"evals": [...]}`` object or a bare array."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if isinstance(data, dict):
        if "evals" not in data:
            raise ValueError(f"{path}: expected a top-level 'evals' array")
        data = data["evals"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: 'evals' must be a JSON array of test cases")

    entries: List[Tuple[str, Dict]] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(
                f"{path} (evals[{i}]): expected an object, got {type(obj).__name__}"
            )
        entries.append((f"evals[{i}]", obj))
    return entries


def _read_jsonl(path: str) -> Iterator[Tuple[str, Dict]]:
    """Parse a legacy ``.jsonl`` suite: one object per line; ``#`` lines skipped."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            yield f"line {lineno}", obj
