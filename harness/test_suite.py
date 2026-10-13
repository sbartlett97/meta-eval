"""Test-suite schema + loader (PRD v3.1).

The suite (``data/test_suite_v1.json``) is a single JSON object with the cases
nested under ``evals`` (easy to hand-edit)::

    {
      "evals": [
        {
          "id": "refuse-weapon-synthesis-01",   # unique test id
          "prompt": "...",                        # sent to the model under test
          "system_prompt": "...",                 # optional; sent as the system
                                                    # turn (see model_loader.py /
                                                    # llamacpp_engine.py)
          "criteria": "expect:FAIL ...",          # judged against (see prompts.py)
          "category": "safety-refusal",           # grouping used by judges/
                                                    # heuristic_judge.py for
                                                    # category-aware checks
          "domain": "...",                        # optional provenance fields,
          "technique": "...",                     # preserved through to
          "severity": "...",                      # results/*.json so per-slice
          "source": "...",                        # analysis is possible later
          "metadata": { ... }                      # optional, free-form
        }
      ]
    }

A case that needs a real multi-turn exchange (does the model stay consistent
across a follow-up, rather than seeing both halves as one static prompt) uses
``turns`` instead of ``prompt``::

    {
      "id": "reasoning-consistency-06",
      "turns": ["part A prompt", "the follow-up prompt"],
      "criteria": "...",
      ...
    }

``turns`` and ``prompt`` are never both required: a single-turn case's
``prompt`` is internally just ``turns == [prompt]`` (see ``TestCase.prompt``,
kept for backward compatibility with anything reading it directly).

Any other top-level keys (e.g. ``_comment``) are ignored, so you can keep notes
in the file. A bare top-level JSON array is also accepted. Legacy ``.jsonl``
suites (one JSON object per line, ``#`` comments allowed) still load, dispatched
on the file extension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass
class TestCase:
    id: str
    criteria: str
    turns: List[str] = field(default_factory=list)
    category: str = "uncategorized"
    system_prompt: Optional[str] = None
    domain: str = ""
    technique: str = ""
    severity: str = ""
    source: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """The first (or only) turn. Kept for single-turn call sites/back-compat."""
        return self.turns[0] if self.turns else ""

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1

    @classmethod
    def from_dict(cls, obj: Dict) -> "TestCase":
        missing = [k for k in ("id", "criteria") if k not in obj]
        if missing:
            raise ValueError(f"Test case missing required fields {missing}: {obj}")

        turns_in = obj.get("turns")
        if turns_in is not None:
            if not isinstance(turns_in, list) or not turns_in or not all(
                isinstance(t, str) and t for t in turns_in
            ):
                raise ValueError(
                    f"`turns` must be a non-empty list of non-empty strings: {obj}"
                )
            turns = list(turns_in)
        elif "prompt" in obj:
            turns = [obj["prompt"]]
        else:
            raise ValueError(
                f"Test case missing required field 'prompt' or 'turns': {obj}"
            )

        return cls(
            id=obj["id"],
            turns=turns,
            criteria=obj["criteria"],
            category=obj.get("category", "uncategorized"),
            system_prompt=obj.get("system_prompt") or None,
            domain=obj.get("domain", ""),
            technique=obj.get("technique", ""),
            severity=obj.get("severity", ""),
            source=obj.get("source", ""),
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
