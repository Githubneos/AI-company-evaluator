"""Scoring for calibrated predictions, 2-class or 3-class.

Every metric is reported alongside the same metric for a constant prediction of
the training-set class priors. On a target this noisy, a model can post a
respectable-looking log loss purely by learning that most windows are
unremarkable, so the absolute number says very little and the comparison
against the prior says almost everything. Skill scores below are positive only
when the model beats that baseline.

Brier score and calibration matter more here than ranking metrics do: the
downstream LLM evaluator states a confidence to a human, so the probabilities
have to mean what they say, not merely sort correctly. See docs/SPEC.md 8.1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

N_CLASSES = 3


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier: mean over samples of the squared error summed over classes."""
    return float(np.mean(np.sum((proba - _one_hot(y_true, proba.shape[1])) ** 2, axis=1)))


def macro_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    """AUC, or None when a fold is missing a class entirely."""
    n_classes = proba.shape[1]
    if len(np.unique(y_true)) < n_classes:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def average_precision(y_true: np.ndarray, proba: np.ndarray, positive_class: int) -> float | None:
    """Area under the precision-recall curve for one class.

    Spec 8.1 prefers this to ROC-AUC under class imbalance: it ignores the large
    easy negative class and reports only how well the rare positives are ranked.
    """
    actual = (y_true == positive_class).astype(int)
    if actual.sum() == 0 or actual.sum() == len(actual):
        return None
    return float(average_precision_score(actual, proba[:, positive_class]))


def precision_at_k(
    y_true: np.ndarray, proba: np.ndarray, positive_class: int, k_fraction: float = 0.05
) -> float | None:
    """Precision among the highest-confidence predictions (spec 8.1).

    Users act on the strongest signals, not on every row, so top-k precision is
    closer to realised experience than a metric averaged over all predictions.
    """
    k = max(int(len(y_true) * k_fraction), 1)
    if k > len(y_true):
        return None
    top = np.argsort(-proba[:, positive_class])[:k]
    return float(np.mean(y_true[top] == positive_class))


def calibration_bins(
    y_true: np.ndarray, proba: np.ndarray, positive_class: int, n_bins: int = 10
) -> list[dict]:
    """Reliability curve for one class: predicted vs observed frequency per bin."""
    p = proba[:, positive_class]
    actual = (y_true == positive_class).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        bins.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": int(sel.sum()),
                "mean_predicted": float(p[sel].mean()),
                "observed_frequency": float(actual[sel].mean()),
            }
        )
    return bins


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, positive_class: int) -> float:
    """Bin-count-weighted mean gap between predicted and observed frequency.

    One number for "when it says 30%, does it happen 30% of the time".
    """
    bins = calibration_bins(y_true, proba, positive_class)
    if not bins:
        return float("nan")
    total = sum(b["n"] for b in bins)
    return float(
        sum(b["n"] * abs(b["mean_predicted"] - b["observed_frequency"]) for b in bins) / total
    )


def evaluate(y_true: np.ndarray, proba: np.ndarray, priors: np.ndarray) -> dict:
    n_classes = proba.shape[1]
    labels = list(range(n_classes))
    priors = np.asarray(priors, dtype=float)
    # A walk-forward report may carry one baseline distribution per row, each
    # learned from that row's preceding training fold.  Accepting that matrix
    # keeps pooled skill strictly out-of-sample rather than comparing early
    # folds to class frequencies learned from their future outcomes.
    baseline = priors if priors.ndim == 2 else np.tile(priors, (len(y_true), 1))
    if baseline.shape != proba.shape:
        raise ValueError("priors must be one class distribution or one per prediction")

    model_ll = float(log_loss(y_true, proba, labels=labels))
    base_ll = float(log_loss(y_true, baseline, labels=labels))
    model_brier = brier_score(y_true, proba)
    base_brier = brier_score(y_true, baseline)

    # For 3-class direction the interesting rare class is SPIKE(2)/DROP(0); for
    # binary magnitude it is LARGE_MOVE(1). In both cases the last index is a
    # "something happened" class.
    positive = n_classes - 1

    return {
        "n": int(len(y_true)),
        "log_loss": model_ll,
        "log_loss_baseline": base_ll,
        "log_loss_skill": float(1.0 - model_ll / base_ll) if base_ll > 0 else None,
        "brier": model_brier,
        "brier_baseline": base_brier,
        "brier_skill": float(1.0 - model_brier / base_brier) if base_brier > 0 else None,
        "macro_auc": macro_auc(y_true, proba),
        "average_precision": average_precision(y_true, proba, positive),
        "precision_at_5pct": precision_at_k(y_true, proba, positive, 0.05),
        "base_rate_positive": float(np.mean(y_true == positive)),
        "calibration_error": expected_calibration_error(y_true, proba, positive),
        "accuracy": float(np.mean(np.argmax(proba, axis=1) == y_true)),
        "accuracy_baseline": float(np.mean(y_true == np.argmax(baseline, axis=1))),
        "class_distribution": {str(c): float(np.mean(y_true == c)) for c in labels},
    }
