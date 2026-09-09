"""Feature construction.

Every feature here is a function of data at or before its own timestamp. That
invariant is what makes training honest, and it is the one thing to check
before adding anything to this module: pandas `rolling`, `ewm`, `shift(+n)`
and `diff` all look backwards, while `shift(-n)`, centered windows, and any
full-column normalisation (z-scoring against the whole sample, fitting a
scaler outside a fold) look forwards and will silently inflate every metric
downstream.

This is also the single definition of the feature set, used by both training
and serving, so the two cannot drift apart. See docs/SPEC.md 2.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluator.regimes import recession_flag

EPS = 1e-12


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Trailing daily return standard deviation (not annualised)."""
    return returns.rolling(window, min_periods=window // 2).std()


#: Event types given their own "days since" feature. Restricted to types that
#: are both reasonably frequent and plausibly informative -- one column per
#: taxonomy entry would add mostly-constant features that dilute the split
#: search without carrying signal.
TRACKED_EVENT_TYPES = (
    "EARNINGS_RESULT",
    "EXECUTIVE_CHANGE",
    "MA_ANNOUNCED",
    "REGULATORY_ACTION",
    "FRAUD_ACCOUNTING",
    "AUDITOR_CHANGE",
    "DISCLOSURE",
    "PERIODIC_REPORT",
)

EVENT_DENSITY_WINDOW = "90D"
NO_EVENT_SENTINEL = 9999.0


def _as_ns(values) -> pd.DatetimeIndex:
    """Datetimes at nanosecond resolution, tz-naive."""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.as_unit("ns")


def build_event_features(index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    """Event-derived features (spec 2.3), causal by construction.

    Every column answers a question of the form "as of the close on day t, what
    had already been filed?" -- `merge_asof(direction="backward")` enforces
    that, and the event dates themselves were already shifted past the close by
    `evaluator.events.effective_date`.

    Companies with no event of a given type carry `NO_EVENT_SENTINEL` rather
    than NaN: "never happened" and "no data" are different, and LightGBM would
    otherwise route them down the same branch.
    """
    out = pd.DataFrame(index=index)
    # merge_asof refuses to join keys of differing datetime resolution, and a
    # parquet round-trip can hand back second-precision timestamps where the
    # price index is nanosecond. Normalise both sides rather than depending on
    # how the caller happened to load its data.
    frame = pd.DataFrame({"date": _as_ns(index)}).sort_values("date")

    if events is None or events.empty:
        for event_type in TRACKED_EVENT_TYPES:
            out[f"days_since_{event_type.lower()}"] = NO_EVENT_SENTINEL
        out["event_density_90d"] = 0.0
        out["days_since_any_event"] = NO_EVENT_SENTINEL
        return out

    events = events.copy()
    events["event_date"] = _as_ns(events["event_date"]).normalize()
    events = events.sort_values("event_date")

    def days_since(subset: pd.DataFrame) -> pd.Series:
        if subset.empty:
            return pd.Series(NO_EVENT_SENTINEL, index=index)
        merged = pd.merge_asof(
            frame,
            subset[["event_date"]].rename(columns={"event_date": "last_event"}),
            left_on="date",
            right_on="last_event",
            direction="backward",
        )
        delta = (merged["date"] - merged["last_event"]).dt.days
        return pd.Series(delta.to_numpy(), index=index).fillna(NO_EVENT_SENTINEL)

    for event_type in TRACKED_EVENT_TYPES:
        out[f"days_since_{event_type.lower()}"] = days_since(
            events[events["event_type"] == event_type]
        )
    out["days_since_any_event"] = days_since(events)

    # Event density: a proxy for company instability (spec 2.3). Counted on a
    # calendar window so it does not distort across holidays.
    per_day = events.groupby("event_date").size()
    daily = per_day.reindex(index.union(per_day.index)).fillna(0.0).sort_index()
    density = daily.rolling(EVENT_DENSITY_WINDOW).sum()
    out["event_density_90d"] = density.reindex(index).fillna(0.0)

    return out


def build_features(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = prices["close"]
    volume = prices["volume"]
    returns = close.pct_change()

    out = pd.DataFrame(index=prices.index)

    for window in (5, 20, 60, 252):
        out[f"ret_{window}"] = close.pct_change(window)

    out["vol_20"] = realized_vol(returns, 20)
    out["vol_60"] = realized_vol(returns, 60)
    out["vol_ratio_20_60"] = out["vol_20"] / (out["vol_60"] + EPS)
    out["atr_14_pct"] = _atr(prices, 14) / (close + EPS)

    log_volume = np.log1p(volume.clip(lower=0.0))
    vol_mean = log_volume.rolling(60, min_periods=30).mean()
    vol_std = log_volume.rolling(60, min_periods=30).std()
    out["volume_z_60"] = (log_volume - vol_mean) / (vol_std + EPS)

    out["rsi_14"] = _rsi(close, 14)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    out["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (close + EPS)

    sma_20 = close.rolling(20, min_periods=10).mean()
    sd_20 = close.rolling(20, min_periods=10).std()
    out["bb_position"] = (close - sma_20) / (2.0 * sd_20 + EPS)

    high_252 = close.rolling(252, min_periods=60).max()
    low_252 = close.rolling(252, min_periods=60).min()
    out["drawdown_from_252h"] = close / (high_252 + EPS) - 1.0
    out["runup_from_252l"] = close / (low_252 + EPS) - 1.0

    if benchmark is not None and not benchmark.empty:
        bench_close = benchmark["close"].reindex(prices.index).ffill()
        bench_returns = bench_close.pct_change()
        cov = returns.rolling(60, min_periods=30).cov(bench_returns)
        var = bench_returns.rolling(60, min_periods=30).var()
        beta = cov / (var + EPS)
        out["beta_60"] = beta
        for window in (5, 20, 60):
            stock_ret = close.pct_change(window)
            bench_ret = bench_close.pct_change(window)
            out[f"excess_ret_{window}"] = stock_ret - beta * bench_ret

    # Sector-relative return (spec 2.1). The spec calls this critical: raw
    # returns are dominated by whatever the sector did, so a stock down 3% on a
    # day its sector fell 4% is strength, not weakness.
    if sector is not None and not sector.empty:
        sector_close = sector["close"].reindex(prices.index).ffill()
        for window in (5, 20, 60):
            out[f"sector_rel_ret_{window}"] = (
                close.pct_change(window) - sector_close.pct_change(window)
            )
        sector_returns = sector_close.pct_change()
        out["sector_vol_20"] = realized_vol(sector_returns, 20)
        out["vol_vs_sector"] = out["vol_20"] / (out["sector_vol_20"] + EPS)

    if events is not None:
        out = out.join(build_event_features(prices.index, events))

    out["recession"] = recession_flag(prices.index)

    if macro is not None and not macro.empty:
        aligned = macro.reindex(prices.index).ffill()
        if "vix" in aligned:
            out["vix"] = aligned["vix"]
            out["vix_chg_20"] = aligned["vix"] - aligned["vix"].shift(20)
        if "curve_slope" in aligned:
            out["curve_slope"] = aligned["curve_slope"]
            out["curve_slope_chg_60"] = aligned["curve_slope"] - aligned["curve_slope"].shift(60)

    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_DESCRIPTIONS = {
    "ret_5": "5-day trailing return",
    "ret_20": "20-day trailing return",
    "ret_60": "60-day trailing return",
    "ret_252": "1-year trailing return",
    "vol_20": "20-day realized daily volatility",
    "vol_60": "60-day realized daily volatility",
    "vol_ratio_20_60": "short-vs-long volatility ratio (vol regime shift)",
    "atr_14_pct": "14-day ATR as a fraction of price",
    "volume_z_60": "log-volume z-score vs 60-day average",
    "rsi_14": "14-day RSI",
    "macd_hist_pct": "MACD histogram, price-normalised",
    "bb_position": "position within 20-day Bollinger Bands",
    "drawdown_from_252h": "drawdown from 1-year closing high",
    "runup_from_252l": "run-up from 1-year closing low",
    "beta_60": "60-day beta to the market benchmark",
    "excess_ret_5": "5-day beta-adjusted excess return vs market",
    "excess_ret_20": "20-day beta-adjusted excess return vs market",
    "excess_ret_60": "60-day beta-adjusted excess return vs market",
    "vix": "VIX level",
    "vix_chg_20": "20-day change in VIX",
    "curve_slope": "10y minus 3m Treasury yield spread",
    "curve_slope_chg_60": "60-day change in the yield-curve slope",
    "sector_rel_ret_5": "5-day return relative to the stock's sector ETF",
    "sector_rel_ret_20": "20-day return relative to the stock's sector ETF",
    "sector_rel_ret_60": "60-day return relative to the stock's sector ETF",
    "sector_vol_20": "20-day realized volatility of the sector ETF",
    "vol_vs_sector": "stock volatility divided by sector volatility",
    "event_density_90d": "material SEC filings in the trailing 90 days",
    "days_since_any_event": "days since the last material filing",
    "recession": "NBER recession indicator",
    **{
        f"days_since_{t.lower()}": f"days since the last {t.replace('_', ' ').lower()} filing"
        for t in TRACKED_EVENT_TYPES
    },
}
