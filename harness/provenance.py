"""Run provenance for results/*.json (docs/critical-assessment.md #10).

A result file only self-describes if it also records what produced it. Before
this, ``model_outputs.json`` / ``model_verdicts.json`` recorded only
``test_id`` / ``model`` per row -- nothing said which suite revision, harness
commit, or judge panel generated the file, so a file found months later
couldn't be trusted or reproduced (was it the suite before or after a case
got fixed?).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def git_commit() -> Optional[str]:
    """The harness's current commit hash, or ``None`` if it can't be determined
    (not a git checkout, ``git`` unavailable, ...). Best-effort and silent on
    failure -- provenance is a nice-to-have, never something that should abort
    a run.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- best-effort; any failure just means "unknown"
        return None


def file_sha256(path: str) -> Optional[str]:
    """SHA-256 of a file's bytes, or ``None`` if it can't be read."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def collect(suite_path: Optional[str] = None, **extra) -> Dict:
    """Provenance metadata for a results file.

    Always includes a generation timestamp and the harness's git commit.
    ``suite_path``, if given, also records the suite file's content hash --
    the suite is edited in place (see data/test_suite_v1.json), so the path
    alone doesn't distinguish "the suite as it was when this file was
    generated" from "the suite as it is now". Any other keyword becomes a
    run-specific provenance field (models used, judge panel, etc.).
    """
    provenance: Dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_git_commit": git_commit(),
    }
    if suite_path:
        provenance["suite_path"] = suite_path
        provenance["suite_sha256"] = file_sha256(suite_path)
    provenance.update(extra)
    return provenance
