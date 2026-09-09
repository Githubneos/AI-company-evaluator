"""Per-ticker assembly: features + labels, aligned and time-ordered.

This is the single path from raw inputs to a model-ready row, used by the panel
builder at training time and by the scoring endpoint at serving time. Keeping
one implementation is what prevents train/serve skew (spec 2.5) -- the moment
serving gets its own "faster" feature code, the two drift and the drift is
invisible until predictions are quietly wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from evaluator.config import LabelConfig, TargetSpec, default_targets
from evaluator.data.edgar import EdgarClient
from evaluator.data.fundamentals import join_as_of, load_fundamentals
from evaluator.data.sources import load_benchmark, load_macro, load_prices
from evaluator.data.universe import cik_for, sector_map
from evaluator.events import build_event_table
from evaluator.features.build import build_features
from evaluator.labels import make_labels

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.DataFrame
    prices: pd.DataFrame
    ticker: str

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def trainable(self, label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
        """Rows with a resolved label and at least one usable feature.

        Rows are *not* dropped for having some NaN features: LightGBM handles
        missing values natively, and dropping them would throw away the early
        history of every long-lookback feature.
        """
        mask = self.labels[label_column].notna() & self.features.notna().any(axis=1)
        return self.features.loc[mask], self.labels.loc[mask, label_column].astype(int)

    def latest_row(self) -> pd.DataFrame:
        """Most recent feature row -- the one with no resolved label yet."""
        return self.features.iloc[[-1]]


def build_dataset(
    ticker: str,
    start: str,
    end: str | None = None,
    label_cfg: LabelConfig | None = None,
    *,
    use_cache: bool = True,
    with_events: bool = True,
    with_fundamentals: bool = True,
    with_sector: bool = True,
    targets: list[TargetSpec] | None = None,
    events: pd.DataFrame | None = None,
) -> Dataset:
    """Assemble one ticker.

    Optional inputs degrade to absent rather than failing: a company with no
    EDGAR coverage still produces a usable row set, with its event features held
    at the "never happened" sentinel.
    """
    label_cfg = label_cfg or LabelConfig()

    prices = load_prices(ticker, start, end, use_cache=use_cache)
    benchmark = load_benchmark(start, end, use_cache=use_cache)
    macro = load_macro(start, end, use_cache=use_cache)

    sector_prices = None
    if with_sector:
        etf = sector_map().get(ticker)
        if etf:
            try:
                sector_prices = load_prices(etf, start, end, use_cache=use_cache)
            except Exception as exc:  # noqa: BLE001 - sector data is optional
                log.warning("sector ETF %s unavailable for %s: %s", etf, ticker, exc)

    if with_events and events is None:
        cik = cik_for(ticker)
        if cik is not None:
            try:
                filings = EdgarClient().filings(cik, ticker, use_cache=use_cache)
                events = build_event_table(filings)
            except Exception as exc:  # noqa: BLE001 - events are optional
                log.warning("events unavailable for %s: %s", ticker, exc)

    features = build_features(
        prices,
        benchmark=benchmark,
        macro=macro,
        events=events if with_events else None,
        sector=sector_prices,
    )

    if with_fundamentals:
        cik = cik_for(ticker)
        if cik is not None:
            try:
                fundamentals = load_fundamentals(ticker, cik, use_cache=use_cache)
                features = features.join(join_as_of(features.index, fundamentals))
            except Exception as exc:  # noqa: BLE001 - fundamentals are optional
                log.warning("fundamentals unavailable for %s: %s", ticker, exc)

    # One label frame per horizon; columns are namespaced by target name so a
    # single row carries every target the six models need.
    targets = targets or default_targets()
    label_frames = {}
    for horizon in sorted({t.horizon_days for t in targets}):
        cfg = LabelConfig(
            horizon_days=horizon,
            threshold_sigmas=label_cfg.threshold_sigmas,
            vol_lookback=label_cfg.vol_lookback,
        )
        frame = make_labels(prices, cfg)
        # "label" is a back-compat alias for label_direction; keeping it here
        # would store the same column twice per horizon.
        for column in frame.columns.drop("label"):
            label_frames[f"{column}_{horizon}d"] = frame[column]

    labels = pd.DataFrame(label_frames, index=prices.index)
    labels["label"] = labels.get(f"label_direction_{label_cfg.horizon_days}d")

    return Dataset(features=features, labels=labels, prices=prices, ticker=ticker)
