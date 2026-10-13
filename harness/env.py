"""Load API keys from the repo's ``.env`` file into the environment.

The remote judges read ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` and hydration
reads ``HF_TOKEN`` from ``os.environ``. Keeping them in a gitignored ``.env``
is the usual setup, but nothing read that file, so a run without the keys
exported looked successful while every remote verdict was an error. The CLIs
call :func:`load_env` first to close that gap.

Variables already set in the environment win over ``.env`` values. If
``python-dotenv`` is not installed this is a no-op, so the import-light core
(and the test suite) still works without it.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")


def load_env(path: Optional[str] = None) -> bool:
    """Load ``path`` (default: ``<repo>/.env``) without overriding set variables.

    Returns True if a file was loaded, False if it is missing or python-dotenv
    is not installed.
    """
    path = path or DEFAULT_ENV_PATH
    if not os.path.isfile(path):
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(
            "%s exists but python-dotenv is not installed (`pip install python-dotenv`); "
            "its variables were NOT loaded.", path,
        )
        return False
    load_dotenv(path, override=False)
    return True
