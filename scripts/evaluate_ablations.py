"""CLI: which feature groups earn their place over the volatility features?

    python -m scripts.evaluate_ablations --targets magnitude_1d
    for t in magnitude_1d magnitude_5d magnitude_20d direction_1d direction_5d direction_20d; do
        python -m scripts.evaluate_ablations --targets $t
    done

Fits small models on volatility plus one feature group at a time (and on all
features) over the trained model's own folds, checks they reproduce its
out-of-fold rows, and reports each set's Brier edge over volatility alone. One
target per process on 8 GB.
"""

from __future__ import annotations

import argparse
import json
import logging

from evaluator.config import ValidationConfig, default_targets
from evaluator.features.store import load_panel
from evaluator.model.ablations import evaluate_ablations
from evaluator.model.train import MODEL_DIR, PanelTrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=None,
                        help="default: the fold count stored with each trained model")
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--embargo-days", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    targets = default_targets()
    if args.targets:
        wanted = set(args.targets)
        targets = [t for t in targets if t.name in wanted]
        if not targets:
            raise SystemExit(f"unknown target(s). Available: {', '.join(t.name for t in default_targets())}")

    for spec in targets:
        meta_path = MODEL_DIR / spec.name / "metadata.json"
        if not meta_path.exists():
            raise SystemExit(f"{spec.name} is not trained. Run: python -m scripts.train --targets {spec.name}")
        n_splits = args.n_splits or len(json.loads(meta_path.read_text())["validation"]["folds"])
        cfg = PanelTrainConfig(
            validation=ValidationConfig(n_splits=n_splits, test_days=args.test_days, embargo_days=args.embargo_days)
        )
        report = evaluate_ablations(load_panel(targets=[spec]), spec, cfg)

        level = f"{report['ci_level']:.1%} CI"
        print(f"\n{spec.name}: Brier edge over volatility alone (same rows, small trees)")
        print(f"intervals are Bonferroni-adjusted ({level}); 'earns its place' also needs "
              f"edge >= {report['min_edge']:+.3f} and 70% of folds\n")
        print(f"{'set':<22} {'n':>3} {'skill':>8} {'edge vs vol':>12} {level:>19} {'folds':>7}  verdict")
        print("-" * 96)
        rows = [(name, r) for name, r in report["sets"].items()] + [("production model", report["model"])]
        for name, r in rows:
            e = r["edge_vs_vol"]
            lo, hi = e["ci"]
            n = r.get("n_features", "")
            print(f"{name:<22} {n!s:>3} {r['pooled']['brier_skill']:>+8.4f} {e['pooled']:>+12.4f} "
                  f"{f'{lo:+.4f}..{hi:+.4f}':>19} {str(e['fold_wins']) + '/' + str(e['n_folds']):>7}  {r['verdict']}")
        print(f"\ngroups that earn their place: {', '.join(report['groups_that_earn_their_place']) or 'none'}")
        recommended = report["recommended_features"]
        print(f"recommended feature set ({len(recommended)}): {', '.join(recommended)}")


if __name__ == "__main__":
    main()
