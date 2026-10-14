"""Decline memory: never send a judge content it has already declined.

When a cloud judge declines to grade an item (``abstain_reason`` starting
``judge_declined``), the decline is recorded in a local JSONL file, keyed by
the judge id and a hash of the content the judge was sent (request turns,
system prompt and response). Later runs skip that judge on the same content
without a call. Repeated requests that a provider declines put the API
account at risk, and suite labels (severity tier, domain) cannot be trusted
to catch all of them: in the 2.12 gold check, four cyber items labelled
``general`` / ``code`` reached Claude and were declined again.

The file holds only hashes and ids, never the content itself.

Backfill from earlier verdict files::

    python judges/declines.py --outputs results/gold/sample.json \\
        results/gold/verdicts.json results/gold/verdicts_v2_1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges.items import EvalItem  # noqa: E402

DECLINED_PREFIX = "judge_declined"
DEFAULT_PATH = "results/judge_declines.jsonl"


def content_hash(item: EvalItem) -> str:
    """Hash of what a judge is sent about ``item`` (not the rubric text, so a
    rubric change does not make declined content look new)."""
    payload = json.dumps(
        {"turns": list(item.turns), "system_prompt": item.system_prompt,
         "response": item.model_output},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_decline(verdict) -> bool:
    reason = getattr(verdict, "abstain_reason", None) or ""
    return reason.startswith(DECLINED_PREFIX)


class DeclineRegistry:
    """The set of ``(judge_id, content_hash)`` pairs a judge has declined,
    backed by an append-only JSONL file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._seen: Set[Tuple[str, str]] = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._seen.add((rec["judge_id"], rec["content_sha256"]))

    def __len__(self) -> int:
        return len(self._seen)

    def declined(self, judge_id: str, item: EvalItem) -> bool:
        return (judge_id, content_hash(item)) in self._seen

    def record(self, judge_id: str, item: EvalItem, reason: str) -> bool:
        """Record one decline. Returns False if it was already known."""
        key = (judge_id, content_hash(item))
        if key in self._seen:
            return False
        self._seen.add(key)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "judge_id": judge_id,
                "content_sha256": key[1],
                "test_id": item.test_id,
                "reason": reason,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        return True


def _row_item(row: Dict) -> EvalItem:
    return EvalItem(
        test_id=row["test_id"],
        model_output=row.get("transcript") or row.get("output", ""),
        criteria=row.get("criteria", ""),
        category=row.get("category", "uncategorized"),
        metadata=row.get("metadata") or {},
        turns=tuple(row.get("turns") or ()),
        system_prompt=row.get("system_prompt"),
        item_type=row.get("item_type") or "",
    )


def backfill(registry: DeclineRegistry, output_rows: Iterable[Dict],
             verdict_rows: Iterable[Dict]) -> int:
    """Record the declines in ``verdict_rows``, joining each to its output row
    (by model, test_id, sample_index) to hash the content. Returns how many
    new declines were recorded."""
    def key(row):
        return (row.get("model"), row["test_id"], row.get("sample_index", 0))

    outputs = {key(r): r for r in output_rows}
    added = 0
    for row in verdict_rows:
        out = outputs.get(key(row))
        for v in row.get("verdicts") or []:
            reason = v.get("abstain_reason") or ""
            if out is not None and reason.startswith(DECLINED_PREFIX):
                added += registry.record(v["judge_id"], _row_item(out), reason)
    return added


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Record judge declines from verdict files.")
    parser.add_argument("verdicts", nargs="+")
    parser.add_argument("--outputs", required=True, help="the outputs file the verdicts came from")
    parser.add_argument("--registry", default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    with open(args.outputs, encoding="utf-8") as fh:
        outputs = json.load(fh)["outputs"]
    registry = DeclineRegistry(args.registry)
    added = 0
    for path in args.verdicts:
        with open(path, encoding="utf-8") as fh:
            added += backfill(registry, outputs, json.load(fh)["verdicts"])
    print(f"Recorded {added} new decline(s); {len(registry)} in {args.registry}.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
