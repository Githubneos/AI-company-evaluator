"""The modelling universe: ~500 S&P names with sector and SEC CIK.

Checked in as a static CSV rather than scraped at runtime, so a training run is
reproducible: re-running last month's build must not silently pick up this
month's index changes.

SURVIVORSHIP BIAS, RESTATED AND WORSE
-------------------------------------
This is the S&P 500 as constituted *today*. Every company in it survived to
today and was successful enough to be in the index. Firms that went bankrupt,
were acquired, or were dropped for underperformance are absent -- and those are
exactly the names that produced the largest downside moves. A model trained here
will understate the frequency and severity of large drops, and no amount of
validation rigour detects it, because the bias is in the sample rather than the
method.

Two consequences to keep in view:
  - Reported DROP-class frequencies are a floor, not an estimate.
  - `date_added` is kept so the bias can at least be *measured*: restricting to
    names added before a backtest's start date removes look-ahead index
    membership, though it cannot resurrect the firms that failed.

Fixing this properly needs CRSP delisting codes and returns (spec 1.2).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from evaluator.config import REPO_ROOT

log = logging.getLogger(__name__)

UNIVERSE_PATH = REPO_ROOT / "data" / "universe" / "sp500.csv"
CHANGES_PATH = REPO_ROOT / "data" / "universe" / "sp500_changes.csv"
CHANGES_META_PATH = REPO_ROOT / "data" / "universe" / "sp500_changes_meta.json"

#: Removal reasons that mean the company itself stopped trading under that
#: ticker. Anything else (an index reshuffle) leaves it listed, so its price
#: series legitimately continues past the removal date.
DELISTING_WORDS = (
    "acquir", "merg", "bought", "purchas", "taken private", "buyout", "bankrupt",
    "delist", "spun off", "split into", "dissolv", "liquidat",
)

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
}


@lru_cache(maxsize=1)
def load_changes() -> pd.DataFrame:
    """Index additions and removals, oldest first. Empty if not fetched yet."""
    if not CHANGES_PATH.exists():
        return pd.DataFrame(columns=["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"])
    frame = pd.read_csv(CHANGES_PATH, parse_dates=["date"])
    for col in ("added_ticker", "removed_ticker"):
        frame[col] = frame[col].astype("string").str.strip().str.upper()
    return frame.sort_values("date").reset_index(drop=True)


def is_delisting(reason: object) -> bool:
    text = str(reason or "").lower()
    return any(word in text for word in DELISTING_WORDS)


@lru_cache(maxsize=1)
def membership_intervals() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]]:
    """When each ticker was an index member: {ticker: [(start, end or None), ...]}.

    Reconstructed by walking the change history backwards from today's index.
    Think of a cursor moving back through time, holding the set of members
    "just after" the cursor. At a change dated d, stepping across it means:
    a name added on d was *not* a member before d, and a name removed on d
    *was*. `end` is None for a name still in the index; a start of
    `Timestamp.min` means membership predates both the constituents file's
    `date_added` and the change history.
    """
    current = load_universe()
    date_added = dict(zip(current["ticker"], current["date_added"]))
    member = set(current["ticker"])
    spell_end: dict[str, pd.Timestamp | None] = {t: None for t in member}
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = {}
    conflicts = 0

    def record(ticker: str, start: pd.Timestamp, end: pd.Timestamp | None) -> None:
        intervals.setdefault(ticker, []).append((start, end))

    for row in load_changes().sort_values("date", ascending=False).itertuples(index=False):
        date = pd.Timestamp(row.date)
        addition, removal = row.added_ticker, row.removed_ticker

        if isinstance(addition, str) and addition in member:
            record(addition, date, spell_end.get(addition))  # this spell began here
            member.discard(addition)
            spell_end.pop(addition, None)

        if isinstance(removal, str):
            if removal in member:
                # Removed at d yet recorded as a member after d: the history is
                # missing the addition in between. Close what we have at d.
                record(removal, date, spell_end.get(removal))
                conflicts += 1
            member.add(removal)
            spell_end[removal] = date

    for ticker in member:
        end = spell_end.get(ticker)
        start = date_added.get(ticker)
        start = pd.Timestamp(start) if start is not None and pd.notna(start) else pd.Timestamp.min
        # `date_added` describes the name's *current* spell. For a name that
        # left and rejoined, this earliest spell predates it, so the constituents
        # file says nothing about when it began.
        if end is not None and start >= end:
            start = pd.Timestamp.min
        record(ticker, start, end)

    if conflicts:
        log.info("membership history: %d removals without a matching addition", conflicts)
    return {t: sorted(v, key=lambda iv: iv[0]) for t, v in intervals.items()}


def was_member(ticker: str, dates) -> pd.Series:
    """Boolean mask: was `ticker` in the index on each of `dates`?"""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    spells = membership_intervals().get(ticker)
    if not spells:
        return pd.Series(False, index=index)
    mask = pd.Series(False, index=index)
    for start, end in spells:
        inside = index >= start
        if end is not None:
            inside &= index < end
        mask |= inside
    return mask


def historical_tickers() -> list[str]:
    """Every ticker that was ever in the index, per the change history."""
    return sorted(membership_intervals())


def removed_tickers() -> dict[str, tuple[pd.Timestamp, str]]:
    """Tickers no longer in the index: {ticker: (removal date, reason)}."""
    current = set(load_universe()["ticker"])
    out: dict[str, tuple[pd.Timestamp, str]] = {}
    for row in load_changes().itertuples(index=False):
        removal = row.removed_ticker
        if isinstance(removal, str) and removal not in current:
            out[removal] = (pd.Timestamp(row.date), str(row.reason or ""))
    return out


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    """Columns: ticker, name, sector, date_added, cik, sector_etf."""
    df = pd.read_csv(UNIVERSE_PATH, dtype={"cik": "Int64"})
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["sector_etf"] = df["sector"].map(SECTOR_ETF)
    return df


def tickers(limit: int | None = None) -> list[str]:
    df = load_universe()
    return df["ticker"].head(limit).tolist() if limit else df["ticker"].tolist()


def cik_for(ticker: str) -> int | None:
    df = load_universe()
    row = df.loc[df["ticker"] == ticker, "cik"]
    return None if row.empty or pd.isna(row.iloc[0]) else int(row.iloc[0])


def sector_map() -> dict[str, str]:
    """ticker -> sector ETF, for the sector-relative features of spec 2.1."""
    df = load_universe()
    return dict(zip(df["ticker"], df["sector_etf"]))


def sector_etfs() -> list[str]:
    return sorted(set(SECTOR_ETF.values()))
