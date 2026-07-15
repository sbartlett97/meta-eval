"""Test-suite schema + loader (PRD v3.1).

The *content* of the 15-test behavioral-invariance suite is Sam's manual work
(PRD: "Design 15-test suite (your manual work)"). This module defines the schema
and a loader so the runner has a stable contract to build against; a tiny example
suite lives at ``data/test_suite_v1.jsonl``.

Each line of a suite file is a JSON object:
    {
      "id": "refuse-weapon-synthesis-01",   # unique test id
      "prompt": "...",                        # sent to the model under test
      "criteria": "expect:refuse ...",        # judged against (see prompts.py)
      "category": "safety-refusal",           # optional grouping
      "metadata": { ... }                     # optional, free-form
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List


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
    """Load a ``.jsonl`` test suite. Blank lines and ``#`` comments are skipped.

    Raises:
        ValueError: on duplicate ids or malformed rows (with the line number).
    """
    tests: List[TestCase] = []
    seen: set[str] = set()
    for lineno, raw in _iter_rows(path):
        try:
            case = TestCase.from_dict(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate test id {case.id!r}")
        seen.add(case.id)
        tests.append(case)
    return tests


def _iter_rows(path: str) -> Iterator[tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield lineno, stripped
