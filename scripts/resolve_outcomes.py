"""CLI: fill in outcomes for predictions whose forward window has closed.

    python -m scripts.resolve_outcomes

Run on a schedule. Until this runs, every prediction sits unresolved and the
feedback loop has nothing to learn from (spec 6.1).
"""

from __future__ import annotations

import argparse
import logging

from evaluator.feedback.store import feedback_context, resolve_pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-for", default=None, help="print track record for a ticker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    n = resolve_pending()
    print(f"resolved {n} prediction(s)")

    if args.summary_for:
        ticker = args.summary_for.upper()
        context = feedback_context(ticker)
        print(f"\n{ticker} track record:")
        for key, value in context.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
