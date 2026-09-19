"""CLI: how does each model compare with HAR-RV and GARCH(1,1)?

    python -m scripts.evaluate_vol_benchmarks --targets magnitude_1d
    for t in magnitude_1d magnitude_5d magnitude_20d direction_1d direction_5d direction_20d; do
        python -m scripts.evaluate_vol_benchmarks --targets $t
    done

Both benchmarks forecast h-day variance relative to the trailing variance the
label is scaled by, are calibrated to probabilities on each fold's training
rows, and are scored on exactly the model's out-of-fold rows (a mismatch is
refused). Prices come from the local cache only -- no network.
"""

from __future__ import annotations

import argparse
import json
import logging

from evaluator.config import ValidationConfig, default_targets
from evaluator.features.store import load_panel
from evaluator.model.train import MODEL_DIR, PanelTrainConfig
from evaluator.model.vol_benchmarks import evaluate_vol_benchmarks


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
        r = evaluate_vol_benchmarks(load_panel(targets=[spec]), spec, cfg)

        print(f"\n{spec.name}: {r['n_rows']:,} out-of-sample rows, price coverage {r['coverage']:.1%}")
        print(f"  {'model':<12} Brier skill {r['model']['pooled']['brier_skill']:+.4f}")
        for b in r["benchmarks"].values():
            e = b["model_edge"]
            lo, hi = e["ci90"]
            wins = f"{e['fold_wins']}/{e['n_folds']}"
            print(f"  {b['label']:<12} Brier skill {b['pooled']['brier_skill']:+.4f}   "
                  f"model edge {e['pooled']:+.4f} (90% CI {lo:+.4f}..{hi:+.4f}, better in {wins} folds)")


if __name__ == "__main__":
    main()
