"""CLI: build the cross-sectional feature panel (spec 2.5).

    python -m scripts.build_panel --limit 50     # quick smoke build
    python -m scripts.build_panel                # full universe

Requires `python -m scripts.backfill` to have populated the price and EDGAR
caches first.
"""

from __future__ import annotations

import argparse
import logging

from evaluator.features.store import build_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    panel = build_panel(start=args.start, end=args.end, limit=args.limit)
    frame = panel.frame

    print(f"\nrows:     {len(frame):,}")
    print(f"tickers:  {frame['ticker'].nunique()}")
    print(f"features: {len(panel.feature_names)}")
    print(f"schema:   {panel.schema}")
    print(f"span:     {frame['date'].min().date()} -> {frame['date'].max().date()}")
    print(f"memory:   {frame.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    print("\nlabel balance:")
    for column in sorted(c for c in frame.columns if c.startswith("label_")):
        counts = frame[column].value_counts(normalize=True).sort_index()
        share = "  ".join(f"{int(k)}:{v:.3f}" for k, v in counts.items())
        print(f"  {column:<26} n={frame[column].notna().sum():>9,}  {share}")

    print("\nfeature coverage (non-null share):")
    coverage = frame[panel.feature_names].notna().mean().sort_values()
    for name, value in coverage.head(12).items():
        print(f"  {name:<32} {value:.3f}")


if __name__ == "__main__":
    main()
