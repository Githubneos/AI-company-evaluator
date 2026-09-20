"""CLI: score the whole universe from the production models (spec 7).

    python -m scripts.score_universe              # every name
    python -m scripts.score_universe --limit 25   # a quick check

Writes artifacts/scores/<date>.parquet and latest.json, which the dashboard's
leaderboard reads directly. Meant to run after the close; the scheduler job
`score_universe` does exactly this.
"""

from __future__ import annotations

import argparse
import logging

from evaluator.scoring import score_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lookback-start", default="2015-01-01")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    summary = score_universe(args.tickers, lookback_start=args.lookback_start, limit=args.limit)

    print(f"\nas of {summary['as_of']}: {summary['tickers']} tickers, {summary['rows']:,} rows, "
          f"{summary['seconds']:.0f}s")
    print(f"targets: {', '.join(summary['targets'])}")
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} failures:")
        for ticker, reason in list(summary["failures"].items())[:10]:
            print(f"  {ticker:<6} {reason[:90]}")


if __name__ == "__main__":
    main()
