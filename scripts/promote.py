"""CLI: model promotion gate (spec 8.3).

    python -m scripts.promote --target magnitude_5d
    python -m scripts.promote --all --apply

A candidate is only promoted if it improves on the incumbent -- on the rows both
models scored out of fold, when both recorded them -- *and* does not
regress materially in any single regime. Spec 8.3 is explicit about why: a model
can improve on average while getting worse in high-volatility regimes, which is
exactly when accuracy matters most. Aggregate metrics hide that trade, so the
gate checks regimes individually and refuses on any one of them.

The other refusal is absolute: a candidate with no measured skill over its
baseline is never promoted, however much better it looks than the incumbent.
Beating a bad model is not evidence of being a useful one.
"""

from __future__ import annotations

import argparse
import logging

from evaluator.model.promotion import promote
from evaluator.model.registry import CANDIDATE, available_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true", help="actually copy to production")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    targets = [args.target] if args.target else available_targets(CANDIDATE)
    if not targets:
        raise SystemExit("no trained candidate models found. Run: python -m scripts.train")

    for verdict in [promote(t, apply=args.apply) for t in targets]:
        mark = "PROMOTE" if verdict["promote"] else "HOLD   "
        print(f"{mark}  {verdict['target']:<16} {verdict['reason']}")
        common = verdict.get("common_rows")
        if common and common.get("available"):
            lo, hi = common["ci90"]
            print(f"         {'':<16} shared rows {common['n']:,} over {common['n_dates']:,} dates, "
                  f"edge {common['edge']:+.4f} (90% CI {lo:+.4f}..{hi:+.4f})")
        elif common:
            print(f"         {'':<16} common-row check unavailable: {common['reason']}")

    if not args.apply:
        print("\n(dry run -- pass --apply to copy approved models to production)")


if __name__ == "__main__":
    main()
