"""Feature-group ablations: exact partition, and credit goes to the group that carries the signal."""

import json

import numpy as np
import pandas as pd
import pytest

import evaluator.model.ablations as ablations
import evaluator.model.baselines as baselines
import evaluator.model.train as train
from evaluator.config import TargetSpec, ValidationConfig
from evaluator.features.store import FeaturePanel

SPEC = TargetSpec(1, "magnitude")
FEATURES = [
    *baselines.VOL_FEATURES,
    "ret_5",  # technical
    "sector_rel_ret_5",  # sector
    "days_since_earnings_result",  # events
    "vix", "vix_chg_20",  # macro
    "gross_margin",  # fundamentals
]


def _panel(driver: str | None, seed: int = 5) -> FeaturePanel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=1500)
    frame = pd.DataFrame([(d, f"T{i}") for d in dates for i in range(12)], columns=["date", "ticker"])
    n = len(frame)
    frame = frame.assign(**{c: rng.normal(size=n).astype("float32") for c in FEATURES})
    logit = (1.5 * frame[driver] if driver else 0.0) - 1.0
    frame["label_magnitude_1d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return FeaturePanel(frame, FEATURES, ["label_magnitude_1d"], "synthetic")


def _cfg() -> train.PanelTrainConfig:
    return train.PanelTrainConfig(
        validation=ValidationConfig(n_splits=5, test_days=150, embargo_days=5),
        max_train_rows=9000, num_boost_round=200, early_stopping_rounds=20, min_data_in_leaf=50,
    )


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    for module in (train, baselines, ablations):
        monkeypatch.setattr(module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "_log_to_mlflow", lambda *a, **k: None)
    return tmp_path


def test_real_feature_set_is_partitioned_exactly():
    names = [
        "ret_5", "ret_20", "ret_60", "ret_252", "vol_20", "vol_60", "vol_ratio_20_60", "atr_14_pct",
        "volume_z_60", "rsi_14", "macd_hist_pct", "bb_position", "drawdown_from_252h", "runup_from_252l",
        "beta_60", "excess_ret_5", "excess_ret_20", "excess_ret_60", "sector_rel_ret_5", "sector_rel_ret_20",
        "sector_rel_ret_60", "sector_vol_20", "vol_vs_sector", "days_since_earnings_result",
        "days_since_executive_change", "days_since_ma_announced", "days_since_regulatory_action",
        "days_since_fraud_accounting", "days_since_auditor_change", "days_since_disclosure",
        "days_since_periodic_report", "days_since_any_event", "event_density_90d", "recession", "vix",
        "vix_chg_20", "curve_slope", "curve_slope_chg_60", "revenue_growth_yoy", "gross_margin",
        "net_margin", "debt_to_equity", "current_ratio", "earnings_surprise_streak",
    ]
    groups = ablations.feature_groups(names)

    assert sorted(f for cols in groups.values() for f in cols) == sorted(names)
    assert {k: len(v) for k, v in groups.items()} == {
        "technical": 14, "volatility": 4, "sector": 5, "events": 10, "macro": 5, "fundamentals": 6,
    }


def test_unknown_or_duplicate_features_are_refused():
    with pytest.raises(ValueError, match="no ablation group"):
        ablations.group_of("mystery_feature")
    with pytest.raises(ValueError, match="duplicate"):
        ablations.feature_groups(["vol_20", "vol_20"])


def test_sets_include_vix_level_ablation_and_all():
    sets = ablations.ablation_sets(FEATURES)

    assert sets["vol"] == list(baselines.VOL_FEATURES)
    assert "vix" not in sets["vol+macro-vix_level"] and "vix_chg_20" in sets["vol+macro-vix_level"]
    assert sets["all"] == FEATURES
    assert set(sets) == {
        "vol", "vol+technical", "vol+sector", "vol+events", "vol+macro", "vol+fundamentals",
        "vol+macro-vix_level", "all",
    }


def test_credit_goes_to_the_group_that_carries_the_signal(model_dir):
    panel = _panel("ret_5")  # a technical feature
    train.train_target(panel, SPEC, _cfg())

    report = ablations.evaluate_ablations(panel, SPEC, _cfg())
    sets = report["sets"]

    assert sets["vol+technical"]["verdict"] == "earns its place"
    assert sets["vol+technical"]["edge_vs_vol"]["ci90"][0] > 0
    for other in ("vol+sector", "vol+events", "vol+macro", "vol+fundamentals"):
        assert sets[other]["verdict"] != "earns its place", other
    assert report["groups_that_earn_their_place"] == ["technical"]
    assert report["recommended_features"] == [*baselines.VOL_FEATURES, "ret_5"]
    assert json.loads((model_dir / SPEC.name / "ablations.json").read_text())["target"] == SPEC.name


def test_no_group_earns_credit_when_volatility_carries_everything(model_dir):
    panel = _panel("vol_ratio_20_60")
    train.train_target(panel, SPEC, _cfg())

    report = ablations.evaluate_ablations(panel, SPEC, _cfg())

    assert report["groups_that_earn_their_place"] == []
    assert report["recommended_features"] == list(baselines.VOL_FEATURES)
    assert all(abs(r["edge_vs_vol"]["pooled"]) < 0.02 for r in report["sets"].values())


def test_classify_thresholds():
    base = {"n_folds": 10}
    assert ablations.classify({**base, "ci": [0.001, 0.01], "pooled": 0.005, "fold_wins": 7}) == "earns its place"
    assert ablations.classify({**base, "ci": [0.001, 0.01], "pooled": 0.005, "fold_wins": 6}) == (
        "small, inconsistent gain"
    )
    # Significant but too small to justify a feature group.
    assert ablations.classify({**base, "ci": [0.0001, 0.001], "pooled": 0.0008, "fold_wins": 9}) == (
        "small, inconsistent gain"
    )
    assert ablations.classify({**base, "ci": [-0.02, -0.001], "pooled": -0.01, "fold_wins": 1}) == "hurts"
    assert ablations.classify({**base, "ci": [-0.01, 0.01], "pooled": 0.0, "fold_wins": 5}) == "no measurable effect"


@pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
def test_noise_groups_never_earn_their_place_across_seeds(model_dir, seed):
    """The false positive that motivated MIN_EDGE and the Bonferroni interval."""
    panel = _panel("ret_5", seed=seed)
    train.train_target(panel, SPEC, _cfg())

    report = ablations.evaluate_ablations(panel, SPEC, _cfg())

    assert report["groups_that_earn_their_place"] == ["technical"], seed
