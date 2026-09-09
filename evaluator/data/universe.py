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

from functools import lru_cache

import pandas as pd

from evaluator.config import REPO_ROOT

UNIVERSE_PATH = REPO_ROOT / "data" / "universe" / "sp500.csv"

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
