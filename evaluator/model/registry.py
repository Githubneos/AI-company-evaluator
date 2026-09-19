"""Where models live: candidates that training writes, production that serving reads.

Training overwrites ``artifacts/models/<target>`` on every run. Serving must
therefore never read from there, or a retrain goes live the moment it finishes
and the promotion gate (spec 8.3) decides nothing. Serving reads
``artifacts/production/<target>``, which only `scripts.promote` writes, and
only for a candidate that passed the gate.

Each target's directory holds the booster, its metadata, the out-of-fold
predictions, and whatever analysis reports have been run against it
(baselines, ablations, volatility benchmarks). Promotion copies the directory
whole, so a promoted model's evidence travels with it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from evaluator.config import ARTIFACT_DIR

log = logging.getLogger(__name__)

CANDIDATE_DIR = Path(ARTIFACT_DIR) / "models"
PRODUCTION_DIR = Path(ARTIFACT_DIR) / "production"

CANDIDATE, PRODUCTION = "candidate", "production"
#: Analysis reports that live beside a model and travel with it on promotion.
REPORTS = ("baselines", "ablations", "vol_benchmarks")
MODEL_FILES = ("model.txt", "metadata.json", "oos_predictions.npz")


def stage_dir(stage: str = PRODUCTION) -> Path:
    if stage == PRODUCTION:
        return PRODUCTION_DIR
    if stage == CANDIDATE:
        return CANDIDATE_DIR
    raise ValueError(f"unknown stage {stage!r}; use {PRODUCTION!r} or {CANDIDATE!r}")


def model_dir(target: str, stage: str = PRODUCTION) -> Path:
    return stage_dir(stage) / target


def available_targets(stage: str = PRODUCTION) -> list[str]:
    root = stage_dir(stage)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "model.txt").exists())


def load_metadata(target: str, stage: str = PRODUCTION) -> dict | None:
    path = model_dir(target, stage) / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_report(target: str, kind: str, stage: str = PRODUCTION) -> dict | None:
    """One analysis report for a model, or None if it has not been run."""
    if kind not in REPORTS:
        raise ValueError(f"unknown report {kind!r}; known: {', '.join(REPORTS)}")
    path = model_dir(target, stage) / f"{kind}.json"
    return json.loads(path.read_text()) if path.exists() else None


__all__ = [
    "CANDIDATE",
    "CANDIDATE_DIR",
    "MODEL_FILES",
    "PRODUCTION",
    "PRODUCTION_DIR",
    "REPORTS",
    "available_targets",
    "load_metadata",
    "load_report",
    "model_dir",
    "stage_dir",
]
