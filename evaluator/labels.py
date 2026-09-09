"""Forward-window, volatility-scaled 3-class labels.

The scaling uses *trailing* volatility known at time t. Using realised
volatility over the forward window instead would leak the answer into the
question: the label would be normalised by the very move it is trying to
describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluator.config import DROP, LARGE_MOVE, NEUTRAL, QUIET, SPIKE, LabelConfig
from evaluator.features.build import EPS, realized_vol


def label_scale(prices: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """Expected move size over the horizon under a random walk, known at time t.

    Exposed separately because scoring needs it for the most recent bar, where
    the forward return does not exist yet: it is what later converts a realised
    outcome into the same z-score the model was trained against.
    """
    daily_returns = prices["close"].pct_change()
    return realized_vol(daily_returns, cfg.vol_lookback) * np.sqrt(cfg.horizon_days)


def make_labels(prices: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """Return per-date forward return, its volatility z-score, and a 3-class label.

    Rows where the forward window extends past the end of the data are NaN and
    must be dropped before training -- but they are exactly the rows to score
    at serving time.
    """
    close = prices["close"]

    forward_return = close.shift(-cfg.horizon_days) / close - 1.0

    scale = label_scale(prices, cfg)
    z = forward_return / (scale + EPS)

    known = z.notna()

    direction = pd.Series(np.nan, index=close.index, dtype="float64")
    direction[known & (z >= cfg.threshold_sigmas)] = SPIKE
    direction[known & (z <= -cfg.threshold_sigmas)] = DROP
    direction[known & (z.abs() < cfg.threshold_sigmas)] = NEUTRAL

    # Magnitude collapses DROP and SPIKE: "something big happened", regardless
    # of sign. Strictly easier than direction, and the autocorrelation of
    # volatility means it is the target most likely to carry real signal.
    magnitude = pd.Series(np.nan, index=close.index, dtype="float64")
    magnitude[known & (z.abs() >= cfg.threshold_sigmas)] = LARGE_MOVE
    magnitude[known & (z.abs() < cfg.threshold_sigmas)] = QUIET

    return pd.DataFrame(
        {
            "forward_return": forward_return,
            "forward_z": z,
            "label_direction": direction,
            "label_magnitude": magnitude,
            # Retained so single-target callers keep working.
            "label": direction,
        },
        index=close.index,
    )


def make_labels_for(prices: pd.DataFrame, targets) -> dict[str, pd.DataFrame]:
    """Label frames keyed by `TargetSpec.name`, one per model to be trained."""
    return {spec.name: make_labels(prices, spec.label_config()) for spec in targets}
