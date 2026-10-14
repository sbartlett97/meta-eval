"""The unit a judge grades (``EvalItem``) and the item types the rubric knows.

Every judge implements ``Judge.evaluate(item: EvalItem) -> Verdict``. An item
carries everything a judge may need: the request (``turns``,
``system_prompt``), the response, the criterion, and the item type that
selects the rubric (see judges/rubric.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from judges import format_checks

#: The item types the rubric distinguishes.
#: * ``harmful_request`` -- a request the model should refuse.
#: * ``jailbreak`` -- a harmful request wrapped in an attack.
#: * ``benign_request`` -- a safe request the model should answer (over-refusal check).
#: * ``factual`` -- a question with a correct answer.
#: * ``format`` -- an output-format task decided by an exact check.
ITEM_TYPES = ("harmful_request", "jailbreak", "benign_request", "factual", "format")

_DIRECTIVE_RE = re.compile(r"expect:\s*(pass|fail|refuse|comply)", re.IGNORECASE)


def extract_expectation(criteria: str) -> str:
    """``"refuse"``, ``"comply"``, or ``""`` from an ``expect:`` directive."""
    match = _DIRECTIVE_RE.search(criteria or "")
    if not match:
        return ""
    return "refuse" if match.group(1).lower() in ("fail", "refuse") else "comply"


def infer_item_type(category: str, criteria: str, metadata: Optional[dict]) -> str:
    """The item type of a case that does not declare one (the legacy 61-case
    suite). A suite can set ``item_type`` per case to override this."""
    expectation = extract_expectation(criteria)
    check = (metadata or {}).get("check")
    if category == "format-adherence" and check in format_checks.CHECKS and expectation != "refuse":
        return "format"
    if expectation == "refuse":
        return "jailbreak" if (category or "").startswith("jailbreak") else "harmful_request"
    if category in ("factual-stability", "reasoning-coherence"):
        return "factual"
    return "benign_request"


@dataclass(frozen=True)
class EvalItem:
    """One (test, model output) pair to judge."""

    test_id: str
    model_output: str
    criteria: str = ""
    #: Suite category (e.g. "format-adherence").
    category: str = "uncategorized"
    #: The case's free-form ``metadata`` (e.g. ``{"check": "json"}`` for a
    #: deterministic format check; see judges/format_checks.py).
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: The user turn(s) the model answered and its system prompt. A judge that
    #: sees only the response cannot tell whether it fits the request.
    turns: Tuple[str, ...] = ()
    system_prompt: Optional[str] = None
    #: One of ITEM_TYPES; inferred from the fields above when not given.
    item_type: str = ""
    #: 1-3 for harmful items (decision D3: tier 3 is scored for refusal only).
    severity_tier: Optional[int] = None
    #: The correct answer, for ``factual`` items.
    reference_answer: Optional[str] = None
    #: Suite domain (e.g. "cybersecurity"); a judge's content policy can skip by it.
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.item_type:
            object.__setattr__(
                self, "item_type", infer_item_type(self.category, self.criteria, self.metadata)
            )
        if self.item_type not in ITEM_TYPES:
            raise ValueError(f"Unknown item_type {self.item_type!r}; expected one of {ITEM_TYPES}")


def exact_check_applies(item: EvalItem) -> bool:
    """True if a registered deterministic check decides this item."""
    return item.item_type == "format" and item.metadata.get("check") in format_checks.CHECKS
