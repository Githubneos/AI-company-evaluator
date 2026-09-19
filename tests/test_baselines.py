"""Baseline ablations: same folds as the model, refused otherwise, and honest attribution."""

import numpy as np
import pandas as pd
import pytest

import evaluator.model.baselines as baselines
import evaluator.model.train as train
from evaluator.config import TargetSpec, ValidationConfig
from evaluator.features.store import FeaturePanel

SPEC = TargetSpec(1, "magnitude")
FEATURES = [
    *baselines.BASELINE_FEATURES["simple"],
    "ret_5",
    "rsi_14",
]


def _panel(driver: str, seed: int = 3) -> FeaturePanel:
    """A date-ordered synthetic panel whose label depends on one feature only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=1500)
    tickers = [f"T{i}" for i in range(12)]
    frame = pd.DataFrame(
        [(d, t) for d in dates for t in tickers], columns=["date", "ticker"]
    )
    n = len(frame)
    frame = frame.assign(**{c: rng.normal(size=n).astype("float32") for c in FEATURES})
    logit = 1.5 * frame[driver] - 1.0
    frame["label_magnitude_1d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return FeaturePanel(frame, FEATURES, ["label_magnitude_1d"], "synthetic")


def _cfg(n_splits: int = 5) -> train.PanelTrainConfig:
    return train.PanelTrainConfig(
        validation=ValidationConfig(n_splits=n_splits, test_days=150, embargo_days=5),
        max_train_rows=9000,
        num_boost_round=200,
        early_stopping_rounds=20,
        min_data_in_leaf=50,
    )


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(baselines, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "_log_to_mlflow", lambda *a, **k: None)
    return tmp_path


def test_signal_the_baselines_already_capture_earns_no_credit(model_dir):
    panel = _panel("vol_ratio_20_60")
    train.train_target(panel, SPEC, _cfg())

    report = baselines.evaluate_baselines(panel, SPEC, _cfg())
    inc = report["incremental"][f"vs_{report['reference']}"]

    assert report["reference"] in ("vol", "simple")  # a baseline holding the driver wins
    assert report["pooled"]["simple"]["brier_skill"] > 0.1  # the baseline finds the signal
    assert abs(inc["pooled"]) < 0.02  # ...so the full model adds ~nothing over it
    assert "Adds measurable skill" not in report["verdict"]
    assert (model_dir / SPEC.name / "baselines.json").exists()


def test_signal_outside_the_baselines_is_credited_to_the_model(model_dir):
    panel = _panel("ret_5")  # not a baseline feature
    train.train_target(panel, SPEC, _cfg())

    report = baselines.evaluate_baselines(panel, SPEC, _cfg())
    inc = report["incremental"][f"vs_{report['reference']}"]

    assert abs(report["pooled"]["simple"]["brier_skill"]) < 0.02
    assert inc["pooled"] > 0.1
    assert inc["ci90"][0] > 0 and inc["fold_wins"] == inc["n_folds"]
    assert report["verdict"].startswith("Adds measurable skill")
    assert set(report["by_regime"]) and all(v["incremental"] > 0 for v in report["by_regime"].values())


def test_different_folds_are_refused_not_compared(model_dir):
    panel = _panel("vol_ratio_20_60")
    train.train_target(panel, SPEC, _cfg(n_splits=3))

    with pytest.raises(baselines.BaselineMismatch, match="folds"):
        baselines.evaluate_baselines(panel, SPEC, _cfg(n_splits=4))


def test_changed_panel_is_refused(model_dir):
    train.train_target(_panel("vol_ratio_20_60", seed=3), SPEC, _cfg())

    with pytest.raises(baselines.BaselineMismatch, match="panel has changed"):
        baselines.evaluate_baselines(_panel("vol_ratio_20_60", seed=4), SPEC, _cfg())


def test_missing_baseline_feature_is_a_clear_error(model_dir):
    panel = _panel("ret_5")
    train.train_target(panel, SPEC, _cfg())
    narrow = FeaturePanel(
        panel.frame.drop(columns=["vix"]), [f for f in FEATURES if f != "vix"],
        panel.label_names, panel.schema,
    )

    with pytest.raises(ValueError, match="vix"):
        baselines.evaluate_baselines(narrow, SPEC, _cfg())


def test_paired_fold_stats_are_deterministic_and_pooled_by_rows():
    n = np.array([100, 300, 200, 400])
    model = np.array([0.30, 0.28, 0.33, 0.29])
    ref = np.array([0.31, 0.29, 0.32, 0.30])

    a = baselines.paired_fold_stats(n, model, ref)
    b = baselines.paired_fold_stats(n, model, ref)

    assert a == b
    assert a["pooled"] == pytest.approx(1 - (n * model).sum() / (n * ref).sum())
    assert a["fold_wins"] == 3 and a["n_folds"] == 4
    assert a["ci90"][0] <= a["pooled"] <= a["ci90"][1]


def test_headline_is_against_the_strongest_baseline(model_dir):
    panel = _panel("vol_ratio_20_60")
    train.train_target(panel, SPEC, _cfg())

    report = baselines.evaluate_baselines(panel, SPEC, _cfg())
    best = max(("vol", "earnings", "simple"), key=lambda k: report["pooled"][k]["brier_skill"])

    assert report["reference"] == best
    assert set(report["incremental"]) == {"vs_vol", "vs_earnings", "vs_simple"}
    assert all("ci90" in v for v in report["incremental"].values())


def test_verdict_never_calls_a_zero_edge_skill():
    stats = {"pooled": 0.001, "ci90": [-0.004, 0.006], "fold_wins": 9, "n_folds": 17}
    text = baselines.verdict(0.04, 0.035, stats, "simple")
    assert "not distinguishable from zero" in text
    assert "88% of its skill" in text

    inconsistent = baselines.verdict(
        0.0435, 0.0391, {"pooled": 0.0045, "ci90": [0.0006, 0.0086], "fold_wins": 11, "n_folds": 17}, "vol"
    )
    assert inconsistent.startswith("Small but real edge") and "11/17" in inconsistent
    assert "not distinguishable" not in inconsistent

    worse = baselines.verdict(
        0.0158, 0.0253, {"pooled": -0.01, "ci90": [-0.02, 0.0], "fold_wins": 3, "n_folds": 17}, "vol"
    )
    assert worse.startswith("Worse than the best simple baseline (volatility-only)")
    assert "scores higher (+0.0253 vs +0.0158)" in worse
