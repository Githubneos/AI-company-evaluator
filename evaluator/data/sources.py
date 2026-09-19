"""Daily OHLCV and macro series from free sources.

SURVIVORSHIP BIAS WARNING
-------------------------
yfinance serves only currently-listed tickers. Companies that went bankrupt,
were acquired, or went private are absent, and their delisting returns are
unrecoverable here. A model trained on this data systematically underestimates
downside risk. This is the single largest known defect in the free-data build
and is not fixable by modelling choices -- it needs CRSP (which carries
delisting codes and returns) to actually go away. See docs/SPEC.md 1.2.

`load_prices` is the only place that knows where bars come from, so a CRSP or
Polygon loader can replace it without touching features, labels, or models.
"""

from __future__ import annotations

import logging

import pandas as pd

from evaluator.config import (
    CACHE_DIR,
    MARKET_BENCHMARK,
    VIX_TICKER,
    YIELD_3M_TICKER,
    YIELD_10Y_TICKER,
)
from evaluator.io import atomic_write_parquet, read_parquet_or_none

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str, start: str, end: str | None) -> object:
    safe = ticker.replace("^", "idx_").replace("/", "_")
    return CACHE_DIR / f"{safe}__{start}__{end or 'latest'}.parquet"


def _expected_latest_bar() -> pd.Timestamp:
    """Latest regular-session date expected from a live daily feed."""
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    return today if today.weekday() < 5 else today - pd.offsets.BDay(1)


def load_prices(
    ticker: str,
    start: str,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Split/dividend-adjusted daily bars indexed by tz-naive date.

    Returns columns: open, high, low, close, volume.
    """
    path = _cache_path(ticker, start, end)
    cached = _read_cache(path) if use_cache else None
    # Historical requests are immutable.  A request with no end date is a live
    # series, however: returning its first cached copy forever would silently
    # freeze serving features and every scheduled panel refresh.  Refresh only
    # the missing tail rather than downloading decades of bars again.
    if cached is not None and end is not None:
        return cached

    if cached is not None and end is None:
        expected_last_bar = _expected_latest_bar()
        if cached.index.max() >= expected_last_bar:
            return cached
        fetch_start = str((cached.index.max() + pd.offsets.BDay(1)).date())
    else:
        fetch_start = start

    import yfinance as yf

    raw = yf.download(
        ticker,
        start=fetch_start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        if cached is not None:
            # An unavailable provider must not turn a usable historical cache
            # into a hard serving failure.  The cache age remains observable via
            # its final date and a later request will retry the refresh.
            log.warning("no refresh bars returned for %s; using cache through %s", ticker, cached.index.max())
            return cached
        raise ValueError(f"no price data returned for {ticker!r} ({start} -> {end})")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower)[OHLCV].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    if cached is not None:
        df = pd.concat([cached, df]).loc[lambda frame: ~frame.index.duplicated(keep="last")].sort_index()

    if use_cache:
        _write_cache(df, path)
    return df


def _read_cache(path) -> pd.DataFrame | None:
    """A cache that cannot be read is a miss, not a failure: the caller refetches."""
    return read_parquet_or_none(path)


def _write_cache(df: pd.DataFrame, path) -> None:
    """Atomic: concurrent refreshes of one live series once left a corrupt file (see evaluator.io)."""
    atomic_write_parquet(df, path)


def load_macro(start: str, end: str | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    """VIX level and a yield-curve slope proxy, forward-filled onto trading days.

    ^TNX and ^IRX are quoted in percentage points, so their difference is the
    10y-3m slope in percentage points. Yahoo's coverage of these indices is
    patchy before the mid-1990s; missing values are left as NaN rather than
    back-filled, and LightGBM handles them natively.
    """
    frames = {}
    for name, ticker in (
        ("vix", VIX_TICKER),
        ("y10", YIELD_10Y_TICKER),
        ("y3m", YIELD_3M_TICKER),
    ):
        try:
            frames[name] = load_prices(ticker, start, end, use_cache=use_cache)["close"]
        except Exception as exc:  # noqa: BLE001 - macro series are best-effort
            log.warning("macro series %s (%s) unavailable: %s", name, ticker, exc)

    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    macro = pd.DataFrame(frames).sort_index()
    if {"y10", "y3m"} <= set(macro.columns):
        macro["curve_slope"] = macro["y10"] - macro["y3m"]
    return macro.drop(columns=[c for c in ("y10", "y3m") if c in macro.columns])


def load_benchmark(start: str, end: str | None = None, *, use_cache: bool = True) -> pd.DataFrame:
    return load_prices(MARKET_BENCHMARK, start, end, use_cache=use_cache)
