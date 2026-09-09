"""CLI: tag why wrong predictions were wrong (spec 6.2).

    python -m scripts.postmortem
    python -m scripts.postmortem --no-llm     # rules only, zero API cost

Rules run first; the LLM is consulted only for errors the rules cannot explain.
UNEXPLAINED is a legitimate outcome -- forcing every miss into a story would
make the tag distribution useless for deciding what to fix.
"""

from __future__ import annotations

import argparse
import logging

from evaluator.feedback.postmortem import tag_predictions, tag_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--ticker", default=None, help="summarise one ticker only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = tag_predictions(use_llm=not args.no_llm, limit=args.limit)
    print(f"candidates: {result['candidates']}   tagged: {result['tagged']}")
    for tag, count in sorted(result["by_tag"].items(), key=lambda kv: -kv[1]):
        print(f"  {tag:<24} {count}")

    summary = tag_summary(args.ticker)
    if summary:
        scope = args.ticker or "all tickers"
        print(f"\ncumulative tags ({scope}):")
        for tag, count in sorted(summary.items(), key=lambda kv: -kv[1]):
            print(f"  {tag:<24} {count}")


if __name__ == "__main__":
    main()
