"""Dividend events, derived from the payment series.

The spec's `DIVIDEND_CHANGE` category could not be populated from 8-K item
codes: a dividend decision usually arrives in an Item 8.01 narrative or a press
release, and item codes cannot express it. The payment series can, and it is
free.

Events are dated on the **ex-dividend date**, the day the payment is actually
attached to the shares. The announcement precedes it, often by weeks, so this
is conservative: a feature that says "a cut happened N days ago" refers to
information the market had strictly earlier. Dating events on the announcement
would need announcement data this project does not have.

A suspension has no event of its own -- nothing gets paid, which is the point.
It is detected when a payment that the company's own rhythm predicted fails to
appear, and it is dated at that detection moment (the missed date plus the
tolerance), never at the last payment, which would be a look-ahead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.config import CACHE_DIR
from evaluator.io import atomic_write_parquet, read_parquet_or_none

log = logging.getLogger(__name__)

DIVIDEND_DIR = Path(CACHE_DIR) / "dividends"

#: A change smaller than this is a rounding or FX artefact, not a decision.
CHANGE_TOLERANCE = 0.01
#: How late a payment may be before the schedule counts as broken.
MISSED_TOLERANCE = 1.5
#: A gap longer than this ends a paying period; the next payment re-initiates.
RESTART_TOLERANCE = 2.5
DEFAULT_INTERVAL_DAYS = 91.0  # quarterly, for a company with too little history

DIVIDEND_EVENT_TYPES = ("DIVIDEND_INITIATE", "DIVIDEND_RAISE", "DIVIDEND_CUT", "DIVIDEND_SUSPEND")
COLUMNS = ["ticker", "event_date", "event_type", "event_subtype", "source", "confidence"]


def _cache_path(ticker: str) -> Path:
    return DIVIDEND_DIR / f"{ticker.replace('/', '_')}.parquet"


def load_dividends(ticker: str, *, use_cache: bool = True) -> pd.Series:
    """Ex-date -> dividend per share. Empty for a company that has never paid."""
    path = _cache_path(ticker)
    if use_cache:
        cached = read_parquet_or_none(path)
        if cached is not None:
            series = cached.set_index("ex_date")["dividend"]
            series.index = pd.DatetimeIndex(series.index).normalize()
            return series

    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).dividends
    except Exception as exc:  # noqa: BLE001 - dividends are optional, like events
        log.warning("dividend fetch failed for %s: %s", ticker, exc)
        return pd.Series(dtype="float64")

    series = pd.Series(dtype="float64") if raw is None or len(raw) == 0 else raw.astype(float)
    if not series.empty:
        index = pd.DatetimeIndex(series.index)
        # Ex-dates come back stamped at market open; the event is the day.
        series.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
        series = series[series > 0].sort_index()
    if use_cache:
        atomic_write_parquet(
            pd.DataFrame({"ex_date": series.index, "dividend": series.to_numpy()}), path, index=False
        )
    return series


def dividend_events(ticker: str, dividends: pd.Series, *, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Initiations, raises, cuts and suspensions, each dated when it was observable.

    `as_of` is how far the dividend data is known to run (default: today). A
    company that simply stopped paying gets its suspension only if the expected
    payment date has already passed by then, so a stale cache cannot invent one.
    """
    if dividends is None or dividends.empty:
        return pd.DataFrame(columns=COLUMNS)

    series = dividends[dividends > 0].sort_index()
    gaps = series.index.to_series().diff().dt.days.dropna()
    # The company's own rhythm; quarterly if it has not paid often enough to tell.
    interval = float(np.median(gaps)) if len(gaps) >= 3 else DEFAULT_INTERVAL_DAYS

    records: list[tuple] = []
    previous_amount: float | None = None
    previous_date: pd.Timestamp | None = None

    for date, amount in series.items():
        restarted = previous_date is not None and (date - previous_date).days > interval * RESTART_TOLERANCE
        if previous_amount is None or restarted:
            why = "first payment" if previous_amount is None else "resumed"
            records.append((ticker, date, "DIVIDEND_INITIATE", why, "dividends", 0.9))
        elif amount > previous_amount * (1 + CHANGE_TOLERANCE):
            records.append((ticker, date, "DIVIDEND_RAISE", f"{amount / previous_amount - 1:.1%}", "dividends", 0.9))
        elif amount < previous_amount * (1 - CHANGE_TOLERANCE):
            records.append((ticker, date, "DIVIDEND_CUT", f"{amount / previous_amount - 1:.1%}", "dividends", 0.9))

        if previous_date is not None and restarted:
            # The suspension became observable when the expected payment did not
            # arrive -- dated there, not at the payment that eventually resumed.
            missed = previous_date + pd.Timedelta(days=round(interval * MISSED_TOLERANCE))
            records.append((ticker, missed, "DIVIDEND_SUSPEND", "missed expected payment", "dividends", 0.7))

        previous_amount, previous_date = float(amount), date

    # A company that stopped paying and never resumed: same rule, applied to the
    # last payment. Only once the expected date has actually passed.
    horizon = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
    if previous_date is not None:
        missed = previous_date + pd.Timedelta(days=round(interval * MISSED_TOLERANCE))
        if missed <= horizon:
            records.append((ticker, missed, "DIVIDEND_SUSPEND", "payments stopped", "dividends", 0.7))

    events = pd.DataFrame(records, columns=COLUMNS)
    return events.sort_values("event_date").reset_index(drop=True)


def load_dividend_events(ticker: str, *, use_cache: bool = True) -> pd.DataFrame:
    return dividend_events(ticker, load_dividends(ticker, use_cache=use_cache))


__all__ = [
    "DIVIDEND_EVENT_TYPES",
    "dividend_events",
    "load_dividend_events",
    "load_dividends",
]
