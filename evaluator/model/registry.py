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
RETIRED_DIR = Path(ARTIFACT_DIR) / "retired"

#: Panels whose models may not serve, whatever they scored.
#:
#: A retired panel is one later found to be wrong about its own data. Skill
#: measured on it is not evidence, because the thing being measured was
#: contaminated -- and an inflated incumbent then *wins* the promotion gate
#: against its own clean replacement, which is exactly what happened here. So
#: this is a disqualification, not a tie-break: it is checked before any score.
#:
#: Keyed by whatever identifies the panel in a model's metadata. Models trained
#: before `panel_id` existed record only `schema`, which is why the schema hash
#: is the key for the one panel retired so far.
RETIRED_PANELS = {
    "3db13f4b530b": (
        "Panel 3db13f4b530b (44 features, 503 names) contained 621,211 rows -- 25% of it -- "
        "dated before the company had joined the index. The date_added filter was not taking "
        "effect, so every metric measured on it was computed partly on look-ahead rows. "
        "Rebuilt as e8fe94da6605 with a working membership filter."
    ),
}


def is_retired(metadata: dict | None) -> str | None:
    """Why this model's training panel was retired, or None if it was not."""
    if not metadata:
        return None
    panel = metadata.get("panel") or {}
    for key in (panel.get("panel_id"), metadata.get("schema"), panel.get("schema")):
        if key and key in RETIRED_PANELS:
            return RETIRED_PANELS[key]
    return None


def panel_of(metadata: dict | None) -> str | None:
    """The panel a model was trained on: its id, or its schema for older models."""
    if not metadata:
        return None
    panel = metadata.get("panel") or {}
    return panel.get("panel_id") or metadata.get("schema")


#: Analysis reports that live beside a model and travel with it on promotion.
REPORTS = ("baselines", "ablations", "vol_benchmarks", "deployability")
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
    "RETIRED_DIR",
    "RETIRED_PANELS",
    "MODEL_FILES",
    "PRODUCTION",
    "PRODUCTION_DIR",
    "REPORTS",
    "available_targets",
    "is_retired",
    "load_metadata",
    "load_report",
    "model_dir",
    "panel_of",
    "stage_dir",
]
