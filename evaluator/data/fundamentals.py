"""Fundamentals from SEC XBRL company facts (spec 2.2, Compustat substitute).

POINT-IN-TIME CORRECTNESS
-------------------------
Every XBRL fact carries both `end` (the fiscal period it describes) and `filed`
(the date the filing became public). These are typically 20-60 days apart, and
using `end` as the feature date is the classic fundamentals backtest error: it
hands the model Q3 revenue weeks before anyone outside the company knew it.
Facts are therefore stamped with `filed` and joined as-of that date.

Restatements are handled by keeping the *earliest* filing for each period. What
matters for a point-in-time feature is what was known then, not the corrected
figure published later.

Flow concepts (revenue, income) are period sums, so a 10-K's annual figure and a
10-Q's quarterly figure are not comparable. Only ~quarterly durations are kept.
Balance-sheet concepts are instants and need no such filter.

Raw `companyfacts` payloads are ~4 MB each and are deliberately not cached --
only the extracted quarterly series is kept, which is a few KB per company.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.config import CACHE_DIR
from evaluator.data.edgar import EdgarClient
from evaluator.features.build import _as_ns
from evaluator.io import atomic_write_parquet, read_parquet_or_none

log = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

#: Some filers tag revenue under the newer ASC 606 concept, others under the
#: legacy one. Tried in order; first hit wins.
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]

FLOW_CONCEPTS = {"NetIncomeLoss", "GrossProfit", *REVENUE_CONCEPTS}
INSTANT_CONCEPTS = [
    "Assets",
    "Liabilities",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "StockholdersEquity",
]

QUARTER_MIN_DAYS, QUARTER_MAX_DAYS = 80, 100
FUNDAMENTAL_FEATURES = [
    "revenue_growth_yoy",
    "gross_margin",
    "net_margin",
    "debt_to_equity",
    "current_ratio",
    "earnings_surprise_streak",
]


def _concept_series(facts: dict, concept: str) -> pd.DataFrame:
    """Point-in-time series for one XBRL concept: columns end, val, filed."""
    node = facts.get("us-gaap", {}).get(concept)
    if not node:
        return pd.DataFrame(columns=["end", "val", "filed"])

    rows = []
    for unit, entries in node.get("units", {}).items():
        if unit != "USD":
            continue
        for fact in entries:
            if fact.get("form") not in ("10-K", "10-Q"):
                continue
            if concept in FLOW_CONCEPTS:
                start, end = fact.get("start"), fact.get("end")
                if not start:
                    continue
                days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                if not QUARTER_MIN_DAYS <= days <= QUARTER_MAX_DAYS:
                    continue
            rows.append((fact["end"], fact["val"], fact.get("filed")))

    if not rows:
        return pd.DataFrame(columns=["end", "val", "filed"])

    frame = pd.DataFrame(rows, columns=["end", "val", "filed"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame = frame.dropna(subset=["filed"])

    # Earliest filing wins: that is what was public at the time.
    return (
        frame.sort_values("filed")
        .drop_duplicates(subset=["end"], keep="first")
        .sort_values("end")
        .reset_index(drop=True)
    )


def extract_fundamentals(facts: dict) -> pd.DataFrame:
    """Derived quarterly fundamentals stamped with their filing dates."""
    revenue = pd.DataFrame(columns=["end", "val", "filed"])
    for concept in REVENUE_CONCEPTS:
        revenue = _concept_series(facts, concept)
        if not revenue.empty:
            break

    series = {c: _concept_series(facts, c) for c in ["NetIncomeLoss", "GrossProfit"]}
    series.update({c: _concept_series(facts, c) for c in INSTANT_CONCEPTS})
    series["Revenue"] = revenue

    values: dict[str, pd.Series] = {}
    filed_dates: dict[str, pd.Series] = {}
    for name, frame in series.items():
        if frame.empty:
            continue
        indexed = frame.set_index("end")
        values[name] = indexed["val"]
        filed_dates[name] = indexed["filed"]

    if not values:
        return pd.DataFrame(columns=["filed", *FUNDAMENTAL_FEATURES])

    merged = pd.DataFrame(values).sort_index()

    # A derived ratio is only knowable once every input behind it has been
    # filed, so the row's as-of date is the latest of them.
    filed = pd.DataFrame(filed_dates).max(axis=1).reindex(merged.index)

    out = pd.DataFrame(index=merged.index)
    out["filed"] = filed

    if "Revenue" in merged:
        out["revenue_growth_yoy"] = merged["Revenue"].pct_change(4)
        if "GrossProfit" in merged:
            out["gross_margin"] = merged["GrossProfit"] / merged["Revenue"].replace(0, np.nan)
        if "NetIncomeLoss" in merged:
            out["net_margin"] = merged["NetIncomeLoss"] / merged["Revenue"].replace(0, np.nan)

    if {"Liabilities", "StockholdersEquity"} <= set(merged.columns):
        out["debt_to_equity"] = merged["Liabilities"] / merged["StockholdersEquity"].replace(0, np.nan)
    if {"AssetsCurrent", "LiabilitiesCurrent"} <= set(merged.columns):
        out["current_ratio"] = merged["AssetsCurrent"] / merged["LiabilitiesCurrent"].replace(0, np.nan)

    # Spec 2.2 asks for a "serial disappointer" signal. Without analyst
    # estimates the closest free proxy is a run of falling net income.
    if "NetIncomeLoss" in merged:
        declining = (merged["NetIncomeLoss"].diff() < 0).astype(int)
        out["earnings_surprise_streak"] = (
            declining.groupby((declining != declining.shift()).cumsum()).cumsum()
        )

    out = out.dropna(subset=["filed"]).replace([np.inf, -np.inf], np.nan)
    for column in FUNDAMENTAL_FEATURES:
        if column not in out:
            out[column] = np.nan
    return out.reset_index(drop=True).sort_values("filed").reset_index(drop=True)


def load_fundamentals(
    ticker: str,
    cik: int,
    *,
    client: EdgarClient | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_path = Path(CACHE_DIR) / "fundamentals" / f"{ticker}.parquet"
    if use_cache:
        cached = read_parquet_or_none(cache_path)
        if cached is not None:
            return cached

    client = client or EdgarClient()
    payload = client._get_json(COMPANYFACTS_URL.format(cik=cik))
    if not payload or "facts" not in payload:
        return pd.DataFrame(columns=["filed", *FUNDAMENTAL_FEATURES])

    frame = extract_fundamentals(payload["facts"])
    if use_cache:
        atomic_write_parquet(frame, cache_path, index=False)
    return frame


def join_as_of(index: pd.DatetimeIndex, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """As-of join onto a daily index, keyed on filing date.

    Backward direction only: a row dated t sees a quarterly report only once it
    has actually been filed.
    """
    columns = FUNDAMENTAL_FEATURES
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)

    # Both keys must share a datetime resolution or merge_asof refuses to join;
    # a parquet round-trip can change it under us.
    left = pd.DataFrame({"date": _as_ns(index)}).sort_values("date")
    right = fundamentals.sort_values("filed")[["filed", *columns]].copy()
    right["filed"] = _as_ns(right["filed"])

    # Carry each metric forward independently. A quarter where one ratio could
    # not be computed should not blank out the others, and the last reported
    # value remains the best point-in-time estimate until the next filing.
    right[columns] = right[columns].ffill()

    merged = pd.merge_asof(left, right, left_on="date", right_on="filed", direction="backward")
    return pd.DataFrame(merged[columns].to_numpy(), index=index, columns=columns)
