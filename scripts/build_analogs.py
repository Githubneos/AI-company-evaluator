"""CLI: build the historical analog retrieval index (spec 3.5).

    python -m scripts.build_analogs
"""

from __future__ import annotations

import argparse
import logging

from evaluator.features.store import load_panel
from evaluator.model.analogs import build_index, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=400_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    panel = load_panel()
    index = build_index(panel.frame, max_rows=args.max_rows)
    save_index(index)

    print(f"analog index: {len(index.reference):,} reference rows")
    print(f"features:     {', '.join(index.features)}")


if __name__ == "__main__":
    main()
