"""Drift and system-health monitoring (spec 6.3, 8.2).

Population Stability Index is the drift measure the spec asks for. Bins come
from the *training* distribution and are then applied to live data, which is the
only ordering that answers the question being asked: how far has the world moved
from what this model learned. Re-binning on the live data would compare each
period against itself and report no drift, forever.

Conventional reading, and the thresholds used here:
  PSI < 0.10   stable
  0.10 - 0.25  moderate shift, worth watching
  PSI > 0.25   significant shift; spec 6.3 triggers an out-of-cycle retrain at 0.2

Feedback-loop health is included because a feedback loop that has quietly
stopped resolving outcomes looks exactly like one where nothing has gone wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluator.feedback.store import DB_PATH, connect

log = logging.getLogger(__name__)

PSI_RETRAIN_THRESHOLD = 0.2
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
EPS = 1e-6


@dataclass
class DriftResult:
    feature: str
    psi: float
    verdict: str

    def as_dict(self) -> dict:
        return {"feature": self.feature, "psi": round(self.psi, 4), "verdict": self.verdict}


def _verdict(psi: float) -> str:
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference (training) and a live sample.

    Bin edges are quantiles of `expected`. NaNs are treated as their own bucket
    rather than dropped: a feature that has silently started arriving empty is
    exactly the kind of drift this is meant to catch, and dropping the NaNs
    would hide it.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    expected_missing = float(np.isnan(expected).mean())
    actual_missing = float(np.isnan(actual).mean())

    expected_valid = expected[~np.isnan(expected)]
    actual_valid = actual[~np.isnan(actual)]
    if expected_valid.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected_valid, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = np.histogram(expected_valid, bins=edges)[0] / max(len(expected_valid), 1)
    actual_share = np.histogram(actual_valid, bins=edges)[0] / max(len(actual_valid), 1)

    # Scale the present-value shares by how much of each sample is non-missing,
    # then append the missing bucket so both vectors still sum to 1.
    expected_share = np.append(expected_share * (1 - expected_missing), expected_missing)
    actual_share = np.append(actual_share * (1 - actual_missing), actual_missing)

    expected_share = np.clip(expected_share, EPS, None)
    actual_share = np.clip(actual_share, EPS, None)

    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def feature_drift(
    reference: pd.DataFrame, live: pd.DataFrame, features: list[str] | None = None
) -> list[DriftResult]:
    features = features or [c for c in reference.columns if c in live.columns]
    results = []
    for feature in features:
        psi = population_stability_index(
            reference[feature].to_numpy(), live[feature].to_numpy()
        )
        if not np.isnan(psi):
            results.append(DriftResult(feature, psi, _verdict(psi)))
    return sorted(results, key=lambda r: -r.psi)


def should_retrain(drift: list[DriftResult], threshold: float = PSI_RETRAIN_THRESHOLD) -> bool:
    """Spec 6.3: any feature above the threshold triggers an out-of-cycle retrain."""
    return any(result.psi >= threshold for result in drift)


def feedback_health(db_path=DB_PATH) -> dict:
    """Is the feedback loop actually closing? (spec 8.2)"""
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL"
        ).fetchone()[0]
        tagged = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE post_mortem_tag IS NOT NULL"
        ).fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE error_type = 'incorrect'"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT as_of, resolved_at FROM predictions WHERE resolved_at IS NOT NULL"
        ).fetchall()

    latencies = []
    for row in rows:
        try:
            latencies.append(
                (pd.Timestamp(row["resolved_at"]).tz_localize(None) - pd.Timestamp(row["as_of"])).days
            )
        except (ValueError, TypeError):
            continue

    return {
        "predictions_total": int(total),
        "predictions_resolved": int(resolved),
        "resolution_rate": round(resolved / total, 4) if total else None,
        "post_mortem_coverage": round(tagged / errors, 4) if errors else None,
        "mean_days_to_resolution": round(float(np.mean(latencies)), 2) if latencies else None,
        "healthy": bool(total == 0 or resolved / total > 0.5),
    }


def prediction_distribution(db_path=DB_PATH, window_days: int = 90) -> dict:
    """Prediction volume and score distribution over time (spec 8.2).

    A model degrading silently often shows up here first: the probabilities
    drift toward the priors, or one class stops being predicted at all, well
    before enough outcomes resolve to move the accuracy numbers.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT as_of, predicted_class, proba_drop, proba_neutral, proba_spike FROM predictions"
        ).fetchall()

    if not rows:
        return {"predictions": 0, "note": "No predictions logged yet."}

    frame = pd.DataFrame([dict(r) for r in rows])
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    recent = frame[frame["as_of"] >= frame["as_of"].max() - pd.Timedelta(days=window_days)]

    return {
        "predictions": int(len(frame)),
        "recent_window_days": window_days,
        "recent_predictions": int(len(recent)),
        "class_share": recent["predicted_class"].value_counts(normalize=True).round(4).to_dict(),
        "mean_probabilities": {
            "DROP": round(float(recent["proba_drop"].mean()), 4),
            "NEUTRAL": round(float(recent["proba_neutral"].mean()), 4),
            "SPIKE": round(float(recent["proba_spike"].mean()), 4),
        },
    }


def system_report(reference: pd.DataFrame | None = None, live: pd.DataFrame | None = None) -> dict:
    from evaluator.feedback.store import live_skill

    report = {
        "feedback_loop": feedback_health(),
        "predictions": prediction_distribution(),
        # The backtest is a claim about the past; this is the record of what the
        # system actually predicted and how it turned out.
        "live_skill": live_skill(),
    }
    if reference is not None and live is not None:
        drift = feature_drift(reference, live)
        report["drift"] = {
            "worst": [d.as_dict() for d in drift[:10]],
            "significant_count": sum(1 for d in drift if d.verdict == "significant"),
            "retrain_recommended": should_retrain(drift),
        }
    return report
