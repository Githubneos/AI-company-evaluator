"""CLI: how much skill does each model add over simple signals?

    python -m scripts.evaluate_baselines --targets magnitude_1d
    for t in magnitude_1d magnitude_5d magnitude_20d direction_1d direction_5d direction_20d; do
        python -m scripts.evaluate_baselines --targets $t
    done

Fits small models on volatility, earnings-cycle and VIX features over the
*same* purged walk-forward folds the trained model was validated on, checks
they reproduce its out-of-fold rows exactly, and reports the model's skill over
them. One target per process is the safe way to run all six on 8 GB, for the
same reason training is run that way.

Fold settings default to what the saved model used (fold count from its
metadata; 252 test days and a 10-day embargo, as `scripts.train` defaults). A
mismatch is refused rather than compared.
"""

from __future__ import annotations

import argparse
import json
import logging

from evaluator.config import ValidationConfig, default_targets
from evaluator.features.store import load_panel
from evaluator.model.baselines import evaluate_baselines
from evaluator.model.train import MODEL_DIR, PanelTrainConfig


def _fmt(value, spec: str = "+.4f") -> str:
    return "n/a" if value is None else format(value, spec)


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

    reports = []
    for spec in targets:
        meta_path = MODEL_DIR / spec.name / "metadata.json"
        if not meta_path.exists():
            raise SystemExit(f"{spec.name} is not trained. Run: python -m scripts.train --targets {spec.name}")
        n_splits = args.n_splits or len(json.loads(meta_path.read_text())["validation"]["folds"])
        cfg = PanelTrainConfig(
            validation=ValidationConfig(n_splits=n_splits, test_days=args.test_days, embargo_days=args.embargo_days)
        )
        panel = load_panel(targets=[spec])
        reports.append(evaluate_baselines(panel, spec, cfg))
        del panel

    print("\nBrier skill vs class priors on identical out-of-sample rows. 'edge' is the model's "
          "skill over the strongest baseline for that target.\n")
    print(f"{'target':<15} {'model':>8} {'vol':>8} {'earnings':>9} {'simple':>8} "
          f"{'best':>9} {'edge':>8} {'90% CI':>19} {'folds won':>10}")
    print("-" * 100)
    for r in reports:
        p, ref = r["pooled"], r["reference"]
        inc = r["incremental"][f"vs_{ref}"]
        lo, hi = inc["ci90"]
        print(f"{r['target']:<15} {_fmt(p['model']['brier_skill']):>8} {_fmt(p['vol']['brier_skill']):>8} "
              f"{_fmt(p['earnings']['brier_skill']):>9} {_fmt(p['simple']['brier_skill']):>8} "
              f"{ref:>9} {_fmt(inc['pooled']):>8} {f'{lo:+.4f}..{hi:+.4f}':>19} "
              f"{str(inc['fold_wins']) + '/' + str(inc['n_folds']):>10}")

    print("\nedge over the best baseline, by regime (negative = the baseline did better):")
    for r in reports:
        cells = "  ".join(f"{k}={_fmt(v['incremental'])}" for k, v in sorted(r["by_regime"].items()))
        print(f"  {r['target']:<15} {cells}")

    print()
    for r in reports:
        print(f"{r['target']}: {r['verdict']}")


if __name__ == "__main__":
    main()
