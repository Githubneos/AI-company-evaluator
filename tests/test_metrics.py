"""Metric baselines must respect each walk-forward fold's information set."""

import numpy as np
import pytest

from evaluator.metrics import evaluate


def test_per_prediction_priors_are_used_for_pooled_baseline():
    y = np.array([0, 1])
    proba = np.array([[0.8, 0.2], [0.2, 0.8]])
    priors = np.array([[0.9, 0.1], [0.1, 0.9]])

    result = evaluate(y, proba, priors)
    expected_baseline = np.mean(
        np.sum((priors - np.eye(2)[y]) ** 2, axis=1)
    )
    assert result["brier_baseline"] == pytest.approx(expected_baseline)
    assert result["accuracy_baseline"] == pytest.approx(1.0)


def test_rejects_a_baseline_matrix_with_the_wrong_shape():
    with pytest.raises(ValueError, match="priors"):
        evaluate(np.array([0, 1]), np.full((2, 2), 0.5), np.array([[0.5, 0.5]]))
