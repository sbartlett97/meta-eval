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
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
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


def git_dirty() -> Optional[bool]:
    """True if tracked files have uncommitted changes (the commit hash alone
    then does not describe the code that ran); None if unknown."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(result.stdout.strip())
    except Exception:  # noqa: BLE001 -- best-effort
        return None


#: Distributions whose versions change results: the serving backend, the
#: judge SDKs, and the HF client that resolves weights.
_TRACKED_PACKAGES = (
    "llama_cpp_python", "anthropic", "openai", "huggingface_hub", "requests", "PyYAML",
)


def environment() -> Dict:
    """Python, platform, hardware and package versions. Package versions are
    read from installed metadata, so nothing heavy is imported."""
    versions = {}
    for dist in _TRACKED_PACKAGES:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    memory_gb = None
    try:
        memory_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "packages": versions,
    }


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def weights_info(path: Optional[str]) -> Optional[Dict]:
    """Identify a weights file without hashing gigabytes.

    The HuggingFace cache stores each LFS file as ``blobs/<sha256>``, with the
    snapshot path a symlink to it, so the real path's name is the file hash.
    For a file outside the cache, ``sha256`` is None.
    """
    if not path:
        return None
    real = os.path.realpath(path)
    name = os.path.basename(real)
    try:
        size = os.path.getsize(real)
    except OSError:
        size = None
    return {
        "file": os.path.basename(path),
        "size_bytes": size,
        "sha256": name if _SHA256_RE.match(name) else None,
    }


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
        "harness_git_dirty": git_dirty(),
        "environment": environment(),
    }
    if suite_path:
        provenance["suite_path"] = suite_path
        provenance["suite_sha256"] = file_sha256(suite_path)
    provenance.update(extra)
    return provenance
