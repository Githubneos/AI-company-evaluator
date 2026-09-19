"""Drift detection and promotion-gate tests (spec 6.3, 8.2, 8.3).

Two behaviours matter most here:

PSI bins must come from the training distribution. Re-binning on live data
compares each period against itself and reports stability forever, which is the
quiet way a drift monitor becomes decorative.

The promotion gate must refuse a candidate that improves on average while
regressing in a single regime. Spec 8.3 exists because that trade is invisible
to aggregate metrics and is exactly backwards: the regimes a model gets worse in
are usually the volatile ones where being right matters most.
"""

import numpy as np
import pandas as pd

from evaluator.model.promotion import compare
from evaluator.monitoring import (
    PSI_RETRAIN_THRESHOLD,
    feature_drift,
    population_stability_index,
    should_retrain,
)


class TestPSI:
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=5000)
        assert population_stability_index(sample, sample) < 0.01

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 5000)
        shifted = rng.normal(2.5, 1, 5000)
        assert population_stability_index(reference, shifted) > PSI_RETRAIN_THRESHOLD

    def test_small_shift_stays_below_the_retrain_threshold(self):
        rng = np.random.default_rng(1)
        assert population_stability_index(rng.normal(0, 1, 20000), rng.normal(0.05, 1, 20000)) < 0.1

    def test_a_feature_that_goes_all_nan_is_flagged(self):
        """A silently-empty feature is drift, not missing data to be dropped."""
        rng = np.random.default_rng(2)
        reference = rng.normal(size=2000)
        broken = np.full(2000, np.nan)
        assert population_stability_index(reference, broken) > PSI_RETRAIN_THRESHOLD

    def test_empty_input_is_nan_not_an_exception(self):
        assert np.isnan(population_stability_index(np.array([]), np.array([1.0])))

    def test_constant_reference_does_not_divide_by_zero(self):
        assert population_stability_index(np.ones(100), np.ones(100)) == 0.0


class TestFeatureDrift:
    def test_drift_is_sorted_worst_first(self):
        rng = np.random.default_rng(3)
        reference = pd.DataFrame({"stable": rng.normal(size=3000), "moved": rng.normal(size=3000)})
        live = pd.DataFrame(
            {"stable": rng.normal(size=3000), "moved": rng.normal(3.0, 1, 3000)}
        )
        drift = feature_drift(reference, live)
        assert drift[0].feature == "moved"
        assert drift[0].verdict == "significant"
        assert should_retrain(drift) is True

    def test_no_drift_does_not_trigger_retrain(self):
        rng = np.random.default_rng(4)
        reference = pd.DataFrame({"a": rng.normal(size=8000)})
        live = pd.DataFrame({"a": rng.normal(size=8000)})
        assert should_retrain(feature_drift(reference, live)) is False


def _meta(skill, regimes=None):
    return {
        "validation": {
            "pooled_out_of_sample": {"brier_skill": skill},
            "by_regime": regimes or {},
        }
    }


class TestPromotionGate:
    def test_candidate_without_skill_is_never_promoted(self):
        verdict = compare(_meta(-0.01), None)
        assert verdict["promote"] is False
        assert "no measured skill" in verdict["reason"]

    def test_candidate_without_skill_refused_even_against_a_worse_incumbent(self):
        """Beating a bad model is not evidence of being a useful one."""
        verdict = compare(_meta(-0.01), _meta(-0.05))
        assert verdict["promote"] is False

    def test_first_model_with_skill_is_promoted(self):
        assert compare(_meta(0.03), None)["promote"] is True

    def test_candidate_must_beat_the_incumbent(self):
        assert compare(_meta(0.02), _meta(0.03))["promote"] is False

    def test_regime_regression_blocks_an_aggregate_improvement(self):
        incumbent = _meta(0.02, {"gfc": {"brier_skill": 0.05}, "recovery": {"brier_skill": 0.01}})
        candidate = _meta(0.04, {"gfc": {"brier_skill": -0.02}, "recovery": {"brier_skill": 0.06}})

        verdict = compare(candidate, incumbent)
        assert verdict["promote"] is False
        assert "gfc" in verdict["reason"]
        assert verdict["regressions"][0]["regime"] == "gfc"

    def test_uniform_improvement_is_promoted(self):
        incumbent = _meta(0.02, {"gfc": {"brier_skill": 0.01}, "recovery": {"brier_skill": 0.02}})
        candidate = _meta(0.04, {"gfc": {"brier_skill": 0.03}, "recovery": {"brier_skill": 0.05}})
        assert compare(candidate, incumbent)["promote"] is True

    def test_tiny_regime_wobble_is_tolerated(self):
        """Within-noise movement should not block a genuine improvement."""
        incumbent = _meta(0.02, {"gfc": {"brier_skill": 0.0200}})
        candidate = _meta(0.05, {"gfc": {"brier_skill": 0.0180}})
        assert compare(candidate, incumbent)["promote"] is True
