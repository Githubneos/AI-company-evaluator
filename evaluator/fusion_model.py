"""Optional learned fusion meta-model (spec 5).

The spec offers a shallow model over `[gbm_score, sentiment_score, interaction
terms] -> outcome`, to learn things like "when the GBM and sentiment disagree,
which one is usually right". This implements it as logistic regression under the
same walk-forward discipline as Phase 3.

It is not the default, for a reason worth stating plainly: fitting a combiner
over signals that are individually weak mostly produces a more confident weak
signal. `should_prefer` therefore requires the meta-model to beat the GBM alone
out of sample by a real margin before the fusion layer will use it -- and on the
current data, with sentiment available only for recent dates, that bar is rarely
cleared. Reporting the components separately is the honest default.

The historical sentiment problem is structural: free news feeds return only
recent headlines, so a meta-model trained on history has no sentiment column to
learn from. Until a historical news archive exists, this trains on GBM outputs
plus price context and is a placeholder for the real thing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluator.config import ARTIFACT_DIR, TargetSpec, ValidationConfig
from evaluator.io import atomic_write_bytes, atomic_write_json
from evaluator.metrics import evaluate
from evaluator.validation import PurgedWalkForward

log = logging.getLogger(__name__)

FUSION_DIR = Path(ARTIFACT_DIR) / "fusion"
MIN_IMPROVEMENT = 0.005


@dataclass
class FusionModel:
    model: LogisticRegression
    scaler: StandardScaler
    columns: list[str]
    metrics: dict

    def predict_proba(self, features: dict) -> np.ndarray:
        row = np.array([[features.get(c, 0.0) or 0.0 for c in self.columns]], dtype=float)
        return self.model.predict_proba(self.scaler.transform(row))[0]


def _design_matrix(base_proba: np.ndarray, sentiment: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    """GBM probabilities, sentiment if present, and their interaction."""
    columns = [f"gbm_p{i}" for i in range(base_proba.shape[1])]
    blocks = [base_proba]

    if sentiment is not None:
        sentiment = np.nan_to_num(sentiment.reshape(-1, 1), nan=0.0)
        positive = base_proba[:, -1:].copy()
        blocks.extend([sentiment, sentiment * positive])
        columns.extend(["sentiment", "sentiment_x_gbm"])

    return np.hstack(blocks), columns


def train_fusion(
    base_proba: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    spec: TargetSpec,
    sentiment: np.ndarray | None = None,
    validation: ValidationConfig | None = None,
) -> FusionModel:
    """Fit the meta-model with the same purged walk-forward discipline as the GBM."""
    validation = validation or ValidationConfig(n_splits=3, test_days=252, embargo_days=10)
    X, columns = _design_matrix(base_proba, sentiment)

    splitter = PurgedWalkForward(
        n_splits=validation.n_splits,
        test_size=validation.test_days,
        purge=spec.horizon_days,
        embargo=validation.embargo_days,
    )

    oos_true, oos_meta, oos_base, oos_priors = [], [], [], []
    for train_idx, test_idx in splitter.split_panel(dates):
        scaler = StandardScaler().fit(X[train_idx])
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(scaler.transform(X[train_idx]), y[train_idx])

        oos_meta.append(model.predict_proba(scaler.transform(X[test_idx])))
        oos_true.append(y[test_idx])
        oos_base.append(base_proba[test_idx])
        priors = np.bincount(y[train_idx], minlength=spec.n_classes).astype(float)
        priors /= priors.sum()
        oos_priors.append(np.tile(priors, (len(test_idx), 1)))

    all_true = np.concatenate(oos_true)
    all_priors = np.vstack(oos_priors)

    meta_metrics = evaluate(all_true, np.vstack(oos_meta), all_priors)
    base_metrics = evaluate(all_true, np.vstack(oos_base), all_priors)

    scaler = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(X), y)

    return FusionModel(
        model=final,
        scaler=scaler,
        columns=columns,
        metrics={
            "meta": meta_metrics,
            "gbm_alone": base_metrics,
            "improvement": (meta_metrics["brier_skill"] or 0) - (base_metrics["brier_skill"] or 0),
            "has_sentiment": sentiment is not None,
        },
    )


def should_prefer(model: FusionModel) -> bool:
    """Only use the meta-model when it clearly beats the GBM alone.

    A combiner that merely matches its inputs adds a layer of opacity for no
    gain, and one that is marginally better is probably fitting fold noise.
    """
    return model.metrics["improvement"] >= MIN_IMPROVEMENT


def save(model: FusionModel, target: str) -> None:
    import joblib

    payload = {"model": model.model, "scaler": model.scaler, "columns": model.columns}
    atomic_write_bytes(lambda tmp: joblib.dump(payload, tmp), FUSION_DIR / f"{target}.joblib")
    atomic_write_json(model.metrics, FUSION_DIR / f"{target}.json", default=str)


def load(target: str) -> FusionModel | None:
    import joblib

    path = FUSION_DIR / f"{target}.joblib"
    if not path.exists():
        return None
    blob = joblib.load(path)
    metrics = json.loads((FUSION_DIR / f"{target}.json").read_text())
    return FusionModel(blob["model"], blob["scaler"], blob["columns"], metrics)
