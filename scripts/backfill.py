"""CLI: backfill prices and EDGAR events for the universe (spec 1.1-1.3).

    python -m scripts.backfill --what all

Both stages cache per ticker, so an interrupted run resumes rather than
restarting. EDGAR is rate-limited to well under SEC's 10 req/sec ceiling, which
makes the events stage the slow one -- expect tens of minutes for ~500 names.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from evaluator.config import CACHE_DIR
from evaluator.data.edgar import EdgarClient, load_filings
from evaluator.data.universe import load_universe, sector_etfs
from evaluator.events import UNPOPULATED_TYPES, build_event_table
from evaluator.io import atomic_write_parquet

log = logging.getLogger("backfill")

EVENTS_PATH = CACHE_DIR / "events.parquet"
PRICES_PATH = CACHE_DIR / "prices.parquet"


def backfill_prices(start: str, end: str | None, limit: int | None) -> pd.DataFrame:
    from evaluator.data.sources import load_prices

    universe = load_universe()
    tickers = universe["ticker"].tolist()[: limit or None]
    # Benchmark and sector ETFs are features, not predictions, but they come
    # through the same loader and must cover the same span.
    tickers += ["SPY", "^VIX", "^TNX", "^IRX", *sector_etfs()]

    frames = []
    for i, ticker in enumerate(tickers, start=1):
        try:
            frame = load_prices(ticker, start, end)
            frame = frame.assign(ticker=ticker)
            frames.append(frame.reset_index())
        except Exception as exc:  # noqa: BLE001 - a delisted or renamed ticker must not stop the run
            log.warning("prices failed for %s: %s", ticker, exc)
        if i % 50 == 0:
            log.info("prices %d/%d", i, len(tickers))

    panel = pd.concat(frames, ignore_index=True)
    atomic_write_parquet(panel, PRICES_PATH, index=False)
    log.info("wrote %s rows to %s", f"{len(panel):,}", PRICES_PATH)
    return panel


def backfill_events(limit: int | None) -> pd.DataFrame:
    universe = load_universe()
    universe = universe.head(limit) if limit else universe
    ciks = {
        row.ticker: int(row.cik)
        for row in universe.itertuples(index=False)
        if not pd.isna(row.cik)
    }

    filings = load_filings(list(ciks), ciks, client=EdgarClient())
    events = build_event_table(filings)

    atomic_write_parquet(events, EVENTS_PATH, index=False)
    log.info("wrote %s events to %s", f"{len(events):,}", EVENTS_PATH)
    return events


def backfill_dividends(limit: int | None) -> pd.DataFrame:
    """Per-ticker dividend history, cached; the source of DIVIDEND_* events."""
    from evaluator.data.dividends import dividend_events, load_dividends

    tickers = load_universe()["ticker"].tolist()[: limit or None]
    frames, payers = [], 0
    for i, ticker in enumerate(tickers, start=1):
        try:
            series = load_dividends(ticker)
        except Exception as exc:  # noqa: BLE001 - one bad name must not stop the run
            log.warning("dividends failed for %s: %s", ticker, exc)
            continue
        if series.empty:
            continue
        payers += 1
        frames.append(dividend_events(ticker, series))
        if i % 50 == 0:
            log.info("dividends %d/%d (%d payers)", i, len(tickers), payers)

    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("dividends: %d payers of %d tickers, %s events", payers, len(tickers), f"{len(events):,}")
    return events


def backfill_delisted(start: str, limit: int | None) -> pd.DataFrame:
    """Try to recover names that left the index, and record what can be trusted."""
    from evaluator.data.delisted import COVERAGE_PATH, survey

    coverage = survey(start, limit=limit)
    if coverage.empty:
        log.warning("no removed names in range; fetch the change history first")
        return coverage
    COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE_PATH, index=False)
    log.info("delisted coverage written to %s", COVERAGE_PATH)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what", choices=["prices", "events", "dividends", "delisted", "all"], default="all"
    )
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N tickers only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.what in ("prices", "all"):
        panel = backfill_prices(args.start, args.end, args.limit)
        print(f"prices: {len(panel):,} rows, {panel.ticker.nunique()} tickers")

    if args.what in ("delisted", "all"):
        coverage = backfill_delisted(args.start, args.limit)
        if not coverage.empty:
            usable = (coverage["status"] == "ok").sum()
            print(f"\ndelisted names surveyed: {len(coverage):,}, usable: {usable:,}")
            print(coverage["status"].value_counts().to_string())

    if args.what in ("dividends", "all"):
        payouts = backfill_dividends(args.limit)
        if not payouts.empty:
            print(f"\ndividend events: {len(payouts):,} rows, {payouts.ticker.nunique()} payers")
            print(payouts.event_type.value_counts().to_string())

    if args.what in ("events", "all"):
        events = backfill_events(args.limit)
        print(f"\nevents: {len(events):,} rows, {events.ticker.nunique()} tickers")
        print(f"date range: {events.event_date.min().date()} -> {events.event_date.max().date()}")
        print("\nby type:")
        print(events.event_type.value_counts().to_string())
        print(f"\nnot derivable from item codes (see evaluator/events.py): {', '.join(UNPOPULATED_TYPES)}")


if __name__ == "__main__":
    main()
