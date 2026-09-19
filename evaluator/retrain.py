"""Retraining cadence and drift-triggered refresh (spec 6.3).

Three triggers, in the order they should be consulted:

  scheduled   Quarterly full retrain on the expanded panel.
  incremental Monthly, via LightGBM `init_model` continuation -- cheaper, but it
              only extends the existing trees rather than rethinking them, so it
              cannot recover from a genuine regime change.
  drift       Out-of-cycle when PSI crosses 0.2 on any feature (spec 6.3).

The drift trigger exists because the calendar does not know when the market
changed. A model can be three weeks into a perfectly healthy quarter and already
be scoring inputs that no longer resemble anything it was trained on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evaluator.config import ARTIFACT_DIR, default_targets
from evaluator.io import atomic_write_json
from evaluator.monitoring import PSI_RETRAIN_THRESHOLD, feature_drift, should_retrain

log = logging.getLogger(__name__)

STATE_PATH = Path(ARTIFACT_DIR) / "retrain_state.json"
FULL_RETRAIN_DAYS = 91
INCREMENTAL_DAYS = 30
LIVE_WINDOW_DAYS = 60


@dataclass
class RetrainDecision:
    should_run: bool
    mode: str  # "full" | "incremental" | "none"
    reason: str
    drift: list[dict]

    def as_dict(self) -> dict:
        return {
            "should_run": self.should_run,
            "mode": self.mode,
            "reason": self.reason,
            "drift": self.drift[:10],
        }


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def record_retrain(mode: str) -> None:
    state = _load_state()
    now = datetime.now(UTC).isoformat()
    state["last_" + mode] = now
    if mode == "full":
        state["last_incremental"] = now  # a full retrain subsumes an incremental
    atomic_write_json(state, STATE_PATH)


def _days_since(state: dict, key: str) -> float | None:
    stamp = state.get(key)
    if not stamp:
        return None
    return (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds() / 86400


def decide(panel_frame: pd.DataFrame | None = None, feature_names: list[str] | None = None) -> RetrainDecision:
    """Should a retrain run now, and of which kind?"""
    state = _load_state()
    drift_rows: list[dict] = []

    if panel_frame is not None and feature_names:
        cutoff = panel_frame["date"].max() - pd.Timedelta(days=LIVE_WINDOW_DAYS)
        reference = panel_frame[panel_frame["date"] < cutoff]
        live = panel_frame[panel_frame["date"] >= cutoff]
        if len(reference) > 1000 and len(live) > 200:
            drift = feature_drift(reference[feature_names], live[feature_names], feature_names)
            drift_rows = [d.as_dict() for d in drift]
            if should_retrain(drift):
                worst = drift[0]
                return RetrainDecision(
                    True,
                    "full",
                    f"PSI {worst.psi:.3f} on {worst.feature} exceeds "
                    f"{PSI_RETRAIN_THRESHOLD}; distribution has moved away from training.",
                    drift_rows,
                )

    since_full = _days_since(state, "last_full")
    if since_full is None:
        return RetrainDecision(True, "full", "No recorded training run.", drift_rows)
    if since_full >= FULL_RETRAIN_DAYS:
        return RetrainDecision(
            True, "full", f"{since_full:.0f} days since last full retrain.", drift_rows
        )

    since_incremental = _days_since(state, "last_incremental")
    if since_incremental is None or since_incremental >= INCREMENTAL_DAYS:
        return RetrainDecision(
            True, "incremental", f"{since_incremental or 0:.0f} days since last update.", drift_rows
        )

    return RetrainDecision(
        False,
        "none",
        f"Last full retrain {since_full:.0f}d ago, incremental {since_incremental:.0f}d ago; "
        "no drift trigger.",
        drift_rows,
    )


def run(decision: RetrainDecision | None = None, targets=None) -> dict:
    """Execute the decided retrain.

    Incremental continuation uses LightGBM's `init_model`, which appends trees to
    the existing booster. It is cheaper than a full refit but strictly additive,
    so it is not a substitute for the quarterly rebuild.
    """
    from evaluator.features.store import load_panel
    from evaluator.model.train import train_all

    panel = load_panel()
    decision = decision or decide(panel.frame, panel.feature_names)
    if not decision.should_run:
        return {"ran": False, **decision.as_dict()}

    targets = targets or default_targets()
    log.info("retraining (%s): %s", decision.mode, decision.reason)
    # Walk-forward evaluation inside train_all is always refit from scratch.
    # Only the final serving fit is continued from the prior booster for an
    # incremental run, so a model trained after a validation fold can never
    # influence that fold's score.
    if decision.mode == "incremental":
        try:
            results = train_all(targets, panel=panel, incremental=True)
        except RuntimeError as exc:
            log.warning("incremental retrain unavailable; falling back to full: %s", exc)
            decision = RetrainDecision(
                True,
                "full",
                f"Incremental retrain unavailable ({exc}); ran a full retrain.",
                decision.drift,
            )
            results = train_all(targets, panel=panel)
    else:
        results = train_all(targets, panel=panel)
    record_retrain(decision.mode)

    return {
        "ran": True,
        **decision.as_dict(),
        "targets": {
            name: meta["validation"]["pooled_out_of_sample"]["brier_skill"]
            for name, meta in results.items()
        },
    }
