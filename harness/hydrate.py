"""Weight hydration (PRD v3.1).

When the harness starts we "hydrate" it: the open-weight checkpoints the run
needs are pre-downloaded from the HuggingFace Hub into the local HF cache.
Because generation now loads vLLM engines in-process (see
:mod:`harness.vllm_engine`), the first ``generate`` call would otherwise block on
a multi-gigabyte download mid-run. Hydrating up front makes that cost explicit,
lets it happen once, and surfaces auth errors on gated repos (e.g. Llama-2)
immediately instead of deep inside an evaluation.

``snapshot_download`` is idempotent: already-cached repos are verified and
skipped, so hydrating on every start is cheap after the first run.

What counts as an "open weight" here: any config entry that is served locally and
carries a HuggingFace checkpoint id — the ``local_models`` and (non-cloud)
``fine_tuned_models`` in ``config/models.yaml`` plus the ``access: vllm`` judges
in ``config/judges.yaml``. Cloud/API models (Anthropic, OpenAI, Replicate) have
no local weights and are skipped.

Hydration is **scoped to the run**: :func:`collect_weights` takes filters so the
evaluator only pulls the models it will actually generate with and the judges it
will actually build (so a remote-only ``--judges cheap`` re-judge downloads
nothing). Called with no filters it collects everything — that's what the
standalone ``python harness/hydrate.py`` CLI does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Providers whose weights are NOT downloaded from HF (hosted / cloud-served).
_CLOUD_PROVIDERS = {"anthropic", "openai", "replicate", "api"}


@dataclass(frozen=True)
class WeightSpec:
    """One open-weight repo to hydrate from the HuggingFace Hub."""

    repo_id: str
    #: Where it came from, for logging (e.g. "local_models:mistral-7b-4bit").
    source: str
    #: Optional HF glob(s) to restrict the download (e.g. a single GGUF quant).
    allow_patterns: Optional[List[str]] = None


def collect_weights(
    models_config: Optional[Dict] = None,
    judges_config: Optional[Dict] = None,
    model_ids: Optional[Iterable[str]] = None,
    include_judges: bool = True,
    judge_max_priority: Optional[int] = None,
    only_enabled_judges: bool = True,
) -> List[WeightSpec]:
    """Collect the unique open-weight HF repos a run needs.

    With no filters (the standalone-CLI case) this returns every open weight in
    the configs. The evaluator passes filters so it only hydrates what the run
    will actually load.

    Args:
        models_config: Parsed ``models.yaml`` dict, or a path to it.
        judges_config: Parsed ``judges.yaml`` dict, or a path to it.
        model_ids: If ``None``, include every local model. If an iterable
            (possibly empty), include only models whose ``id`` is in it — pass
            ``[]`` for a run that generates nothing (e.g. re-judging).
        include_judges: If False, skip all judge weights (e.g. ``--no-judge``).
        judge_max_priority: If set, skip judges with ``priority`` greater than
            this — mirrors the same filter :func:`judges.factory.build_judges`
            applies, so only the judges that will actually be built are hydrated.
        only_enabled_judges: Skip judges with ``enabled: false``.

    Returns:
        De-duplicated :class:`WeightSpec` list (first source wins on the label).
    """
    specs: "Dict[str, WeightSpec]" = {}
    wanted = None if model_ids is None else set(model_ids)

    models = _as_dict(models_config)
    for group in ("local_models", "fine_tuned_models"):
        for entry in models.get(group, []) or []:
            if entry.get("provider") in _CLOUD_PROVIDERS:
                continue  # served in the cloud; no local weights
            if wanted is not None and entry.get("id") not in wanted:
                continue  # not requested for this run
            repo = entry.get("checkpoint")
            if not repo:
                continue
            _add(specs, WeightSpec(
                repo_id=repo,
                source=f"{group}:{entry.get('id', repo)}",
                allow_patterns=entry.get("hf_allow_patterns"),
            ))

    if include_judges:
        judges = _as_dict(judges_config)
        for entry in judges.get("judges", []) or []:
            if entry.get("access") != "vllm":
                continue  # only in-process vLLM judges have local weights
            if only_enabled_judges and not entry.get("enabled", True):
                continue
            if judge_max_priority is not None and entry.get("priority", 1) > judge_max_priority:
                continue
            repo = entry.get("model")
            if not repo:
                continue
            _add(specs, WeightSpec(
                repo_id=repo,
                source=f"judges:{entry.get('id', repo)}",
                allow_patterns=entry.get("hf_allow_patterns"),
            ))

    return list(specs.values())


def hydrate_weights(
    models_config: Optional[Dict] = None,
    judges_config: Optional[Dict] = None,
    token: Optional[str] = None,
    dry_run: bool = False,
    model_ids: Optional[Iterable[str]] = None,
    include_judges: bool = True,
    judge_max_priority: Optional[int] = None,
    only_enabled_judges: bool = True,
) -> List[str]:
    """Download the open-weight HF checkpoints a run needs.

    Args:
        models_config: Parsed ``models.yaml`` dict, or a path to it.
        judges_config: Parsed ``judges.yaml`` dict, or a path to it.
        token: HF access token for gated repos. Falls back to the
            ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` env vars picked up by
            ``huggingface_hub`` when omitted.
        dry_run: If True, log what would be downloaded but do not download.
        model_ids, include_judges, judge_max_priority, only_enabled_judges:
            Run-scoping filters forwarded to :func:`collect_weights`.

    Returns:
        The list of repo ids that were hydrated (or would be, on a dry run).
    """
    specs = collect_weights(
        models_config,
        judges_config,
        model_ids=model_ids,
        include_judges=include_judges,
        judge_max_priority=judge_max_priority,
        only_enabled_judges=only_enabled_judges,
    )
    if not specs:
        logger.info("Hydration: nothing to download for this run.")
        return []

    logger.info(
        "Hydrating %d open-weight checkpoint(s) from HuggingFace: %s",
        len(specs),
        ", ".join(s.repo_id for s in specs),
    )
    if dry_run:
        for spec in specs:
            logger.info("  [dry-run] would download %s (%s)", spec.repo_id, spec.source)
        return [s.repo_id for s in specs]

    snapshot_download = _import_snapshot_download()
    hydrated: List[str] = []
    for spec in specs:
        logger.info("Downloading %s (%s) ...", spec.repo_id, spec.source)
        snapshot_download(
            repo_id=spec.repo_id,
            allow_patterns=spec.allow_patterns,
            token=token,
        )
        hydrated.append(spec.repo_id)
    logger.info("Hydration complete: %d checkpoint(s) ready.", len(hydrated))
    return hydrated


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _add(specs: "Dict[str, WeightSpec]", spec: WeightSpec) -> None:
    specs.setdefault(spec.repo_id, spec)


def _import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "huggingface_hub is required to hydrate weights "
            "(`pip install huggingface_hub`), or pass --no-hydrate to skip it."
        ) from exc
    return snapshot_download


def _as_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        with open(value, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


# ---------------------------------------------------------------------- #
# CLI: `python harness/hydrate.py [--dry-run]`
# ---------------------------------------------------------------------- #
def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    import os
    import sys

    # Allow `python harness/hydrate.py` without installing the package.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    parser = argparse.ArgumentParser(description="Download open weights from HuggingFace.")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--judges-config", default="config/judges.yaml")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict model hydration to these ids (default: all local models).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List repos without downloading.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    hydrate_weights(
        models_config=args.models_config,
        judges_config=args.judges_config,
        model_ids=args.models,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
