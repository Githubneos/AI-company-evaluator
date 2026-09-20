from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

MARKET_BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
YIELD_10Y_TICKER = "^TNX"
YIELD_3M_TICKER = "^IRX"

DROP, NEUTRAL, SPIKE = 0, 1, 2
CLASS_NAMES = {DROP: "DROP", NEUTRAL: "NEUTRAL", SPIKE: "SPIKE"}

QUIET, LARGE_MOVE = 0, 1
MAGNITUDE_NAMES = {QUIET: "QUIET", LARGE_MOVE: "LARGE_MOVE"}

#: Spec 1.4 asks for several windows because they are different problems: a
#: 1-day label is dominated by earnings gaps, a 20-day label by drift.
HORIZONS = (1, 5, 20)

DIRECTION, MAGNITUDE = "direction", "magnitude"
#: Direction of the move *relative to the stock's sector ETF*. Raw direction is
#: dominated by whatever the market and the sector did that week, which is the
#: part a stock-specific model has least hope of forecasting -- and the part
#: that sank the direction models in the 2022-23 rate-hike regime.
REL_DIRECTION = "rel_direction"


@dataclass(frozen=True)
class LabelConfig:
    """Forward-window label definition.

    The move threshold is volatility-scaled rather than a fixed percentage: a 5%
    move means something very different for a utility than for a biotech, and a
    fixed threshold makes the positive class almost entirely a proxy for the
    stock's unconditional volatility.
    """

    horizon_days: int = 5
    threshold_sigmas: float = 1.0
    vol_lookback: int = 60


@dataclass(frozen=True)
class TargetSpec:
    """One trained model: a horizon crossed with a target family.

    `direction` (3-class drop/neutral/spike) is what the spec asks for.
    `magnitude` (binary large-move) is added because volatility is strongly
    autocorrelated while short-horizon direction is close to a martingale --
    if anything in this system has real skill, it is most likely here.
    """

    horizon_days: int
    kind: str = DIRECTION
    threshold_sigmas: float = 1.0
    vol_lookback: int = 60

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.horizon_days}d"

    @property
    def n_classes(self) -> int:
        return 2 if self.kind == MAGNITUDE else 3

    @property
    def label_column(self) -> str:
        return f"label_{self.kind}"

    @property
    def class_names(self) -> dict[int, str]:
        return MAGNITUDE_NAMES if self.kind == MAGNITUDE else CLASS_NAMES

    def label_config(self) -> LabelConfig:
        return LabelConfig(
            horizon_days=self.horizon_days,
            threshold_sigmas=self.threshold_sigmas,
            vol_lookback=self.vol_lookback,
        )


def default_targets() -> list[TargetSpec]:
    """Nine models: {direction, magnitude, rel_direction} x {1, 5, 20} days."""
    return [
        TargetSpec(horizon_days=h, kind=kind)
        for kind in (DIRECTION, MAGNITUDE, REL_DIRECTION)
        for h in HORIZONS
    ]


@dataclass(frozen=True)
class ValidationConfig:
    n_splits: int = 6
    test_days: int = 252
    embargo_days: int = 10


@dataclass(frozen=True)
class TrainConfig:
    ticker: str = "AAPL"
    start: str = "2005-01-01"
    end: str | None = None
    label: LabelConfig = field(default_factory=LabelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    seed: int = 7

    def lgb_params(self) -> dict:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
        }
