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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PRICES_PATH, index=False)
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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_PATH, index=False)
    log.info("wrote %s events to %s", f"{len(events):,}", EVENTS_PATH)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", choices=["prices", "events", "all"], default="all")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N tickers only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.what in ("prices", "all"):
        panel = backfill_prices(args.start, args.end, args.limit)
        print(f"prices: {len(panel):,} rows, {panel.ticker.nunique()} tickers")

    if args.what in ("events", "all"):
        events = backfill_events(args.limit)
        print(f"\nevents: {len(events):,} rows, {events.ticker.nunique()} tickers")
        print(f"date range: {events.event_date.min().date()} -> {events.event_date.max().date()}")
        print("\nby type:")
        print(events.event_type.value_counts().to_string())
        print(f"\nnot derivable from item codes (see evaluator/events.py): {', '.join(UNPOPULATED_TYPES)}")


if __name__ == "__main__":
    main()
