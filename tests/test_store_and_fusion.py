"""Feature-store schema guard and fusion payload shape (spec 2.5, 5).

Two properties:

A model must refuse to score features that do not satisfy the schema it was
trained on. Silently reindexing mismatched columns is how a model ends up
scoring inputs that mean something different from what it learned -- with no
error, and no way to notice from the output.

The fusion payload must represent absent signals as `available: false` with a
reason, never as a zero. A sentiment score of 0.0 and a dead news feed are
different states, and the reasoning layer downstream will faithfully explain
whichever one it is handed.
"""

import numpy as np
import pandas as pd
import pytest

from evaluator.features.store import align_to_schema, schema_hash
from evaluator.fusion import _model_quality, _quality_verdict


class TestSchemaGuard:
    def test_hash_is_order_independent(self):
        assert schema_hash(["a", "b", "c"]) == schema_hash(["c", "a", "b"])

    def test_hash_changes_when_a_feature_is_added(self):
        assert schema_hash(["a", "b"]) != schema_hash(["a", "b", "c"])

    def test_matching_columns_are_aligned_and_reordered(self):
        row = pd.DataFrame([{"b": 2.0, "a": 1.0}])
        names = ["a", "b"]
        aligned = align_to_schema(row, names, schema_hash(names))
        assert list(aligned.columns) == names
        assert aligned.iloc[0]["a"] == 1.0

    def test_extra_serving_columns_are_dropped_not_fatal(self):
        """Serving may compute more than the model needs; that is fine."""
        row = pd.DataFrame([{"a": 1.0, "b": 2.0, "unused": 9.0}])
        aligned = align_to_schema(row, ["a", "b"], schema_hash(["a", "b"]))
        assert list(aligned.columns) == ["a", "b"]

    def test_missing_required_feature_is_refused(self):
        row = pd.DataFrame([{"a": 1.0}])
        with pytest.raises(ValueError, match="do not satisfy training schema"):
            align_to_schema(row, ["a", "b"], schema_hash(["a", "b"]))

    def test_infinities_are_converted_to_nan(self):
        row = pd.DataFrame([{"a": np.inf, "b": 1.0}])
        aligned = align_to_schema(row, ["a", "b"], schema_hash(["a", "b"]))
        assert pd.isna(aligned.iloc[0]["a"])


class TestQualityVerdict:
    def test_negative_skill_is_stated_bluntly(self):
        verdict = _quality_verdict(-0.01, 0.51)
        assert "does NOT beat" in verdict
        assert "must not be presented as a signal" in verdict

    def test_marginal_skill_is_called_noise(self):
        assert "noise" in _quality_verdict(0.002, 0.51)

    def test_real_skill_is_reported_as_such(self):
        verdict = _quality_verdict(0.05, 0.62)
        assert "beats base-rate prediction" in verdict

    def test_unmeasurable_skill_is_not_silently_positive(self):
        assert "could not be measured" in _quality_verdict(None, None)

    def test_best_target_drives_the_overall_verdict(self):
        targets = {
            "direction_5d": {"skill": {"brier_skill": -0.01, "macro_auc": 0.50}},
            "magnitude_5d": {"skill": {"brier_skill": 0.04, "macro_auc": 0.61}},
        }
        quality = _model_quality(targets)
        assert quality["best_target"] == "magnitude_5d"
        assert quality["any_target_has_skill"] is True
        assert "does NOT beat" in quality["per_target"]["direction_5d"]["interpretation"]

    def test_all_targets_without_skill_reports_no_skill(self):
        targets = {
            "direction_5d": {"skill": {"brier_skill": -0.01, "macro_auc": 0.50}},
            "magnitude_5d": {"skill": {"brier_skill": 0.001, "macro_auc": 0.51}},
        }
        quality = _model_quality(targets)
        assert quality["any_target_has_skill"] is False
