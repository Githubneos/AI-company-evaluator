"""Price data for names that left the index, and when not to trust it.

Recovering removed names is what turns "today's S&P 500, backfilled" into
something closer to the index as it actually was. The hazard is **ticker
recycling**: symbols are reassigned, so asking a price API for a symbol that
was delisted in 2009 can return a different company that trades under it
today, with a continuous history covering the period we care about. That data
looks perfect and is about the wrong company.

The removal reason tells us what to expect:

- The company stopped trading under that symbol (acquired, merged, taken
  private, bankrupt, delisted). Its series must **end** near the removal date.
  If it runs on well past it, the symbol now belongs to someone else, and the
  name is rejected.
- The index simply dropped it (market-cap reshuffles). The company is still
  listed, so a continuing series is expected and fine.

Either way only rows inside the membership interval enter the panel, so a
symbol reused years later cannot contaminate the period it was a member.
"""

from __future__ import annotations

import logging

import pandas as pd

from evaluator.data.universe import CHANGES_PATH, is_delisting, membership_intervals, removed_tickers

log = logging.getLogger(__name__)

#: How long a delisted name's price series may run past its removal date. Index
#: removal often precedes the final tape by a few sessions.
TAIL_TOLERANCE_DAYS = 25
#: A recovered name is only useful if it covers the window it was a member.
MIN_COVERAGE = 0.5

COVERAGE_PATH = CHANGES_PATH.parent / "removed_coverage.csv"

OK = "ok"
NO_DATA = "no data"
RECYCLED = "ticker reused by another company"
TOO_SHORT = "too little of the membership window"


def membership_window(ticker: str, start: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The part of `ticker`'s membership that overlaps the panel's span."""
    floor = pd.Timestamp(start)
    spells = membership_intervals().get(ticker) or []
    windows = []
    for spell_start, spell_end in spells:
        begin = max(spell_start, floor)
        end = spell_end if spell_end is not None else pd.Timestamp.now().normalize()
        if end > begin:
            windows.append((begin, end))
    if not windows:
        return None
    return min(w[0] for w in windows), max(w[1] for w in windows)


def classify(
    ticker: str,
    prices: pd.DataFrame | None,
    removal_date: pd.Timestamp,
    reason: str,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> dict:
    """Whether this recovered series can be trusted for this name."""
    result = {
        "ticker": ticker,
        "removal_date": removal_date,
        "delisting": is_delisting(reason),
        "reason": reason,
        "rows": 0 if prices is None else int(len(prices)),
        "last_date": None if prices is None or prices.empty else prices.index.max(),
        "window_start": None if window is None else window[0],
        "window_end": None if window is None else window[1],
    }
    if prices is None or prices.empty or window is None:
        return {**result, "status": NO_DATA, "coverage": 0.0}

    start, end = window
    inside = prices.loc[(prices.index >= start) & (prices.index <= end)]
    expected = max(1, len(pd.bdate_range(start, end)))
    coverage = len(inside) / expected
    result["coverage"] = float(coverage)

    if result["delisting"]:
        overrun = (prices.index.max() - removal_date).days
        if overrun > TAIL_TOLERANCE_DAYS:
            # Delisted here, still trading there: not the same company.
            return {**result, "status": RECYCLED, "overrun_days": int(overrun)}
    if coverage < MIN_COVERAGE:
        return {**result, "status": TOO_SHORT}
    return {**result, "status": OK}


def survey(
    start: str = "2005-01-01",
    *,
    limit: int | None = None,
    loader=None,
) -> pd.DataFrame:
    """Try to recover every removed name; report what can be trusted."""
    if loader is None:
        from evaluator.data.sources import load_prices

        def loader(ticker: str):  # noqa: ANN202 - local default
            return load_prices(ticker, start)

    removed = removed_tickers()
    rows = []
    for i, (ticker, (removal_date, reason)) in enumerate(sorted(removed.items())[: limit or None], start=1):
        window = membership_window(ticker, start)
        if window is None:
            continue  # membership ended before the panel's span
        try:
            prices = loader(ticker)
        except Exception as exc:  # noqa: BLE001 - a missing delisted name is the normal case
            log.debug("no prices for %s: %s", ticker, exc)
            prices = None
        rows.append(classify(ticker, prices, removal_date, reason, window))
        if i % 25 == 0:
            log.info("delisted survey %d/%d", i, len(removed))
    return pd.DataFrame(rows)


def usable_tickers(coverage: pd.DataFrame) -> list[str]:
    return sorted(coverage.loc[coverage["status"] == OK, "ticker"]) if not coverage.empty else []


__all__ = [
    "COVERAGE_PATH",
    "MIN_COVERAGE",
    "NO_DATA",
    "OK",
    "RECYCLED",
    "TAIL_TOLERANCE_DAYS",
    "TOO_SHORT",
    "classify",
    "membership_window",
    "survey",
    "usable_tickers",
]
