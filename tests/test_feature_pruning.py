"""Training on a subset of features, and what a pruned model records."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import evaluator.model.train as train
from evaluator.config import TargetSpec, ValidationConfig
from evaluator.features.store import FeaturePanel, align_to_schema, schema_hash
from scripts.train import resolve_features

SPEC = TargetSpec(1, "magnitude")
FEATURES = ["vol_20", "vol_60", "vol_ratio_20_60", "atr_14_pct", "ret_5", "vix", "gross_margin"]


def _panel(seed: int = 11) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=900)
    frame = pd.DataFrame([(d, f"T{i}") for d in dates for i in range(10)], columns=["date", "ticker"])
    n = len(frame)
    frame = frame.assign(**{c: rng.normal(size=n).astype("float32") for c in FEATURES})
    logit = 1.4 * frame["ret_5"] - 1.0
    frame["label_magnitude_1d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return FeaturePanel(frame, FEATURES, ["label_magnitude_1d"], schema_hash(FEATURES))


def _cfg(**kwargs) -> train.PanelTrainConfig:
    return train.PanelTrainConfig(
        validation=ValidationConfig(n_splits=3, test_days=150, embargo_days=5),
        max_train_rows=6000, num_boost_round=150, early_stopping_rounds=20, min_data_in_leaf=50,
        **kwargs,
    )


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "_log_to_mlflow", lambda *a, **k: None)
    return tmp_path


def test_a_pruned_model_trains_on_and_records_only_its_features(model_dir):
    keep = ["vol_20", "vol_60", "ret_5"]

    train.train_target(_panel(), SPEC, _cfg(features=keep))

    metadata = json.loads((model_dir / SPEC.name / "metadata.json").read_text())
    assert metadata["feature_names"] == keep
    assert metadata["panel_feature_names"] == FEATURES  # what was on offer
    booster = train.lgb.Booster(model_file=str(model_dir / SPEC.name / "model.txt"))
    assert booster.feature_name() == keep  # the model cannot see the rest


def test_requesting_a_feature_the_panel_lacks_is_an_error(model_dir):
    with pytest.raises(ValueError, match="missing from the panel"):
        train.train_target(_panel(), SPEC, _cfg(features=["vol_20", "not_a_feature"]))


def test_dropping_the_signal_feature_costs_skill(model_dir):
    """A pruning that removes what matters must show up as lost skill."""
    with_signal = train.train_target(_panel(), SPEC, _cfg(features=["vol_20", "ret_5"]))
    without = train.train_target(_panel(), SPEC, _cfg(features=["vol_20", "vix"]))

    keep = with_signal["validation"]["pooled_out_of_sample"]["brier_skill"]
    lost = without["validation"]["pooled_out_of_sample"]["brier_skill"]
    assert keep > 0.05 and lost < 0.01 and keep > lost


def test_serving_reindexes_a_wider_row_onto_a_pruned_model(model_dir):
    keep = ["vol_20", "ret_5"]
    train.train_target(_panel(), SPEC, _cfg(features=keep))
    metadata = json.loads((model_dir / SPEC.name / "metadata.json").read_text())
    row = pd.DataFrame({c: [0.5] for c in FEATURES})  # serving builds every feature

    aligned = align_to_schema(row, metadata["feature_names"], metadata["schema"])

    assert list(aligned.columns) == keep


def test_serving_refuses_a_row_missing_a_pruned_models_feature(model_dir):
    keep = ["vol_20", "ret_5"]
    train.train_target(_panel(), SPEC, _cfg(features=keep))
    metadata = json.loads((model_dir / SPEC.name / "metadata.json").read_text())
    row = pd.DataFrame({"vol_20": [0.5]})  # ret_5 absent

    with pytest.raises(ValueError, match="ret_5"):
        align_to_schema(row, metadata["feature_names"], metadata["schema"])


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("volatility", ["vol_20", "vol_60", "vol_ratio_20_60", "atr_14_pct"]),
        ("volatility,ret_5", ["vol_20", "vol_60", "vol_ratio_20_60", "atr_14_pct", "ret_5"]),
        ("macro", ["vix"]),
    ],
)
def test_group_names_expand_to_features(requested, expected):
    got = resolve_features(requested, FEATURES)

    assert sorted(got) == sorted(expected)
    assert got == [f for f in FEATURES if f in set(got)]  # panel order preserved


def test_an_unknown_group_is_refused():
    with pytest.raises(SystemExit, match="unknown feature or group"):
        resolve_features("not_a_group", FEATURES)


def test_resolution_is_importable_without_running_the_cli():
    assert "scripts.train" in sys.modules
