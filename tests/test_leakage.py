"""Lookahead guards.

These are the tests worth keeping green above all others: a leak here does not
crash anything, it just makes every downstream metric quietly wrong.
"""

import numpy as np
import pandas as pd
import pytest

from evaluator.config import DROP, NEUTRAL, SPIKE, LabelConfig
from evaluator.features.build import build_features
from evaluator.labels import make_labels


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=900, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, len(dates))),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, len(dates)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, len(dates)))),
            "close": close,
            "volume": rng.integers(1_000_000, 9_000_000, len(dates)).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def benchmark(prices) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 400 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, len(prices))))
    return pd.DataFrame({"close": close}, index=prices.index)


@pytest.fixture
def sector(prices) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    close = 80 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(prices))))
    return pd.DataFrame({"close": close}, index=prices.index)


@pytest.fixture
def events(prices) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    picks = np.sort(rng.choice(len(prices), size=40, replace=False))
    types = rng.choice(
        ["EARNINGS_RESULT", "EXECUTIVE_CHANGE", "MA_ANNOUNCED", "DISCLOSURE"], size=40
    )
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(40)],
            "ticker": "TEST",
            "event_date": prices.index[picks],
            "event_type": types,
            "event_subtype": "X",
            "source": "8-K",
            "confidence": 1.0,
        }
    )


def test_features_do_not_change_when_future_bars_are_appended(
    prices, benchmark, sector, events
):
    """Every feature at time t must be a function of data at or before t.

    Computing on a truncated history and on the full history has to give
    identical values for the overlapping dates -- otherwise something in the
    pipeline is reading forward. Run with every optional input wired up, so
    event, sector, and macro features are covered too and a leak cannot hide in
    an argument the test forgot to pass.
    """
    cutoff = 700
    full = build_features(prices, benchmark=benchmark, sector=sector, events=events)
    truncated = build_features(
        prices.iloc[:cutoff],
        benchmark=benchmark.iloc[:cutoff],
        sector=sector.iloc[:cutoff],
        events=events,  # full event history: future events must still be ignored
    )

    overlap = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[overlap],
        truncated,
        check_exact=False,
        rtol=1e-9,
    )


def test_event_features_ignore_events_dated_after_the_row(prices, events):
    """Passing the complete event table must not change history.

    This is the specific failure the merge_asof direction guards against: an
    as-of join that looked forward would make every row aware of its next
    earnings release.
    """
    cutoff_date = prices.index[500]
    past_only = events[events["event_date"] <= cutoff_date]

    with_future = build_features(prices, events=events).loc[:cutoff_date]
    without_future = build_features(prices, events=past_only).loc[:cutoff_date]

    event_columns = [c for c in with_future.columns if "event" in c or "days_since" in c]
    assert event_columns
    pd.testing.assert_frame_equal(
        with_future[event_columns], without_future[event_columns], rtol=1e-9
    )


def test_days_since_event_is_never_negative(prices, events):
    features = build_features(prices, events=events)
    for column in [c for c in features.columns if c.startswith("days_since_")]:
        assert (features[column] >= 0).all(), column


def test_missing_event_history_is_a_sentinel_not_a_nan(prices):
    """"Never happened" must be distinguishable from "no data"."""
    empty = pd.DataFrame(columns=["event_date", "event_type"])
    features = build_features(prices, events=empty)
    since = features["days_since_earnings_result"]
    assert since.notna().all()
    assert (since == 9999.0).all()


def test_forward_return_matches_the_realised_future_move(prices):
    cfg = LabelConfig(horizon_days=5)
    labels = make_labels(prices, cfg)

    close = prices["close"]
    t = 300
    expected = close.iloc[t + cfg.horizon_days] / close.iloc[t] - 1.0
    assert labels["forward_return"].iloc[t] == pytest.approx(expected)


def test_final_rows_have_no_label(prices):
    """The last `horizon_days` rows cannot be labelled yet -- they are the rows
    to score, not to train on."""
    cfg = LabelConfig(horizon_days=5)
    labels = make_labels(prices, cfg)
    assert labels["label"].iloc[-cfg.horizon_days :].isna().all()
    assert labels["label"].iloc[-cfg.horizon_days - 1] in {DROP, NEUTRAL, SPIKE}


def test_label_scaling_uses_trailing_not_forward_volatility(prices):
    """Perturbing bars strictly after t must not change the z-score scaling at t.

    The forward return itself does change, so compare the implied scale.
    """
    cfg = LabelConfig(horizon_days=5, vol_lookback=60)
    t = 500

    original = make_labels(prices, cfg)
    scale_before = original["forward_return"].iloc[t] / original["forward_z"].iloc[t]

    shocked = prices.copy()
    shocked.iloc[t + cfg.horizon_days + 1 :, shocked.columns.get_loc("close")] *= 3.0
    after = make_labels(shocked, cfg)
    scale_after = after["forward_return"].iloc[t] / after["forward_z"].iloc[t]

    assert scale_before == pytest.approx(scale_after, rel=1e-9)


def test_labels_are_symmetric_around_the_threshold(prices):
    cfg = LabelConfig(horizon_days=5, threshold_sigmas=1.0)
    labels = make_labels(prices, cfg).dropna(subset=["label"])

    assert (labels.loc[labels["label"] == SPIKE, "forward_z"] >= 1.0).all()
    assert (labels.loc[labels["label"] == DROP, "forward_z"] <= -1.0).all()
    assert (labels.loc[labels["label"] == NEUTRAL, "forward_z"].abs() < 1.0).all()
