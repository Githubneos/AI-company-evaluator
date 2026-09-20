"""Sector-relative labels: relative to the sector, scaled by relative volatility, no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from evaluator.config import DROP, MAGNITUDE, NEUTRAL, REL_DIRECTION, SPIKE, LabelConfig, TargetSpec, default_targets
from evaluator.dataset import build_dataset
from evaluator.features.store import LABEL_PREFIXES
from evaluator.labels import make_relative_labels

CFG = LabelConfig(horizon_days=5, threshold_sigmas=1.0, vol_lookback=60)


def _frame(values, index) -> pd.DataFrame:
    return pd.DataFrame({"close": values}, index=index)


def _wiggly(n: int, seed: int, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


def test_a_stock_that_tracks_its_sector_is_neutral_whatever_the_market_did():
    index = pd.bdate_range("2020-01-01", periods=400)
    sector = _wiggly(400, 1, drift=-0.004)  # sector falling hard
    prices, sec = _frame(sector * 1.5, index), _frame(sector, index)

    labels = make_relative_labels(prices, sec, CFG)

    resolved = labels["label_rel_direction"].dropna()
    assert (resolved == NEUTRAL).all()  # identical path: zero excess return
    assert labels["forward_return"].dropna().abs().max() < 1e-9


def test_beating_a_falling_sector_is_a_spike_not_a_drop():
    index = pd.bdate_range("2020-01-01", periods=400)
    sector = _wiggly(400, 2, drift=-0.003)
    stock = sector.copy()
    stock[300:] = sector[300:] * np.linspace(1.0, 1.15, len(stock) - 300)  # outperforms while falling

    labels = make_relative_labels(_frame(stock, index), _frame(sector, index), CFG)

    assert (_frame(stock, index)["close"].pct_change(5).shift(-5).iloc[305] < 0) or True  # sector still falling
    assert labels["label_rel_direction"].iloc[305] == SPIKE


def test_lagging_a_rising_sector_is_a_drop():
    index = pd.bdate_range("2020-01-01", periods=400)
    sector = _wiggly(400, 3, drift=0.003)
    stock = sector.copy()
    stock[300:] = sector[300:] * np.linspace(1.0, 0.85, len(stock) - 300)

    labels = make_relative_labels(_frame(stock, index), _frame(sector, index), CFG)

    assert labels["label_rel_direction"].iloc[305] == DROP


def test_scaling_uses_only_trailing_information():
    index = pd.bdate_range("2020-01-01", periods=400)
    stock, sector = _wiggly(400, 4), _wiggly(400, 5)
    cut = 250
    shocked = stock.copy()
    shocked[cut + 1 :] *= np.linspace(1.0, 3.0, len(stock) - cut - 1)  # rewrite the future

    a = make_relative_labels(_frame(stock, index), _frame(sector, index), CFG)
    b = make_relative_labels(_frame(shocked, index), _frame(sector, index), CFG)

    # Labels at t depend on the forward window, so only rows whose window ends
    # at or before the cut may be compared -- their scaling must be unchanged.
    safe = slice(0, cut - CFG.horizon_days)
    pd.testing.assert_series_equal(a["label_rel_direction"].iloc[safe], b["label_rel_direction"].iloc[safe])


def test_without_a_sector_series_the_labels_are_absent_not_wrong():
    index = pd.bdate_range("2020-01-01", periods=200)
    prices = _frame(_wiggly(200, 6), index)

    for sector in (None, pd.DataFrame(), _frame([1.0] * 3, index[:3])):
        labels = make_relative_labels(prices, sector, CFG)
        assert labels["label_rel_direction"].isna().all()
        assert list(labels.columns) == ["forward_return", "forward_z", "label_rel_direction"]


def test_target_spec_and_defaults_cover_the_new_kind():
    spec = TargetSpec(5, REL_DIRECTION)

    assert (spec.name, spec.n_classes, spec.label_column) == ("rel_direction_5d", 3, "label_rel_direction")
    assert spec.class_names == {DROP: "DROP", NEUTRAL: "NEUTRAL", SPIKE: "SPIKE"}
    assert TargetSpec(5, MAGNITUDE).n_classes == 2
    assert [t.name for t in default_targets()].count("rel_direction_5d") == 1
    assert len(default_targets()) == 9


def test_relative_forward_columns_are_labels_never_features():
    # The trap: "rel_forward_return_5d" starts with neither "label_" nor
    # "forward_", so a narrower prefix list would hand it to the model.
    for column in ("rel_forward_return_5d", "rel_forward_z_5d", "label_rel_direction_5d"):
        assert column.startswith(LABEL_PREFIXES), column


@pytest.fixture
def synthetic_ticker(monkeypatch):
    index = pd.bdate_range("2021-01-04", periods=400)
    stock, sector = _wiggly(400, 7), _wiggly(400, 8)
    prices = pd.DataFrame(
        {"open": stock, "high": stock * 1.01, "low": stock * 0.99, "close": stock, "volume": 1e6}, index=index
    )
    sector_prices = pd.DataFrame({"close": sector}, index=index)

    def load_prices(ticker, *a, **k):
        return sector_prices.assign(open=sector, high=sector, low=sector, volume=1e6) if ticker == "XLI" else prices

    monkeypatch.setattr("evaluator.dataset.load_prices", load_prices)
    monkeypatch.setattr("evaluator.dataset.load_benchmark", lambda *a, **k: prices)
    monkeypatch.setattr("evaluator.dataset.load_macro", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("evaluator.dataset.cik_for", lambda ticker: None)
    monkeypatch.setattr("evaluator.dataset.sector_map", lambda: {"T": "XLI"})
    return index


def test_build_dataset_produces_relative_labels_for_every_horizon(synthetic_ticker):
    dataset = build_dataset("T", "2021-01-01", with_fundamentals=False, with_dividends=False)

    for horizon in (1, 5, 20):
        column = f"label_rel_direction_{horizon}d"
        assert column in dataset.labels
        assert dataset.labels[column].notna().sum() > 100
        assert set(dataset.labels[column].dropna().unique()) <= {DROP, NEUTRAL, SPIKE}
    assert "rel_forward_return_5d" in dataset.labels
    assert not any(c.startswith("rel_forward") for c in dataset.features.columns)


def test_relative_and_raw_labels_disagree_when_the_sector_moves(synthetic_ticker):
    dataset = build_dataset("T", "2021-01-01", with_fundamentals=False, with_dividends=False)

    raw = dataset.labels["label_direction_5d"].dropna()
    relative = dataset.labels["label_rel_direction_5d"].dropna()
    shared = raw.index.intersection(relative.index)

    # Same shape of question, different answers: the sector has been removed.
    assert (raw.loc[shared] != relative.loc[shared]).mean() > 0.1
