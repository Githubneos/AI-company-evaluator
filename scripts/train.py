"""CLI: train all six models on the feature panel (spec 3).

    python -m scripts.train
    python -m scripts.train --targets direction_5d magnitude_5d

Reports pooled and per-regime skill against the base-rate baseline. Skill, not
accuracy, is the number that matters: accuracy near 70% is what you get by
always predicting "no large move".
"""

from __future__ import annotations

import argparse
import logging

from evaluator.config import ValidationConfig, default_targets
from evaluator.features.store import load_panel
from evaluator.model.train import PanelTrainConfig, train_all


def _fmt(value, spec: str = "+.4f") -> str:
    return "n/a" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--max-train-rows", type=int, default=800_000,
                        help="cap rows by keeping every Nth date; lower it if memory is tight")
    parser.add_argument("--date-stride", type=int, default=None,
                        help="force the date stride (use the incumbent's, so the promotion gate "
                             "can compare the two models on shared rows)")
    parser.add_argument("--tune", type=int, default=0, metavar="N",
                        help="Optuna trials before training (spec 3.2); 0 disables")
    parser.add_argument("--challenger", action="store_true",
                        help="also train XGBoost and compare feature importances (spec 3.1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    targets = default_targets()
    if args.targets:
        wanted = set(args.targets)
        targets = [t for t in targets if t.name in wanted]
        if not targets:
            names = [t.name for t in default_targets()]
            raise SystemExit(f"unknown target(s). Available: {', '.join(names)}")

    cfg = PanelTrainConfig(
        validation=ValidationConfig(
            n_splits=args.n_splits,
            test_days=args.test_days,
            embargo_days=args.embargo_days,
        ),
        max_train_rows=args.max_train_rows,
        date_stride=args.date_stride,
    )

    # Load only the label columns these targets need: on 8 GB the unused label
    # columns are the difference between finishing and being OOM-killed.
    panel = load_panel(targets=targets)
    print(f"panel: {len(panel.frame):,} rows, {panel.frame['ticker'].nunique()} tickers, "
          f"{len(panel.feature_names)} features, schema {panel.schema}, "
          f"{panel.frame.memory_usage(deep=True).sum() / 1e6:.0f} MB\n")

    if args.tune:
        from evaluator.model.tuning import tune

        # Tuned on the first target and applied to all: the search is expensive
        # and the targets differ by horizon, not by feature geometry.
        spec = targets[0]
        X, y, dates, _ = panel.target_frame(spec)
        print(f"tuning on {spec.name} for {args.tune} trials...")
        result = tune(X, y, dates, spec, n_trials=args.tune)
        print(f"  best Brier skill {result.best_skill:+.4f}  params {result.best_params}\n")
        for key, value in result.best_params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    results = train_all(targets, cfg, panel)

    print(f"\n{'target':<16} {'brier_skill':>12} {'auc':>7} {'avg_prec':>9} {'p@5%':>7} {'base':>7}")
    print("-" * 62)
    for name, meta in results.items():
        o = meta["validation"]["pooled_out_of_sample"]
        print(f"{name:<16} {_fmt(o['brier_skill']):>12} {_fmt(o['macro_auc'], '.4f'):>7} "
              f"{_fmt(o['average_precision'], '.4f'):>9} {_fmt(o['precision_at_5pct'], '.4f'):>7} "
              f"{_fmt(o['base_rate_positive'], '.4f'):>7}")

    print("\nper-regime Brier skill (spec 3.3 -- a model that only works in calm markets is not safe):")
    for name, meta in results.items():
        by_regime = meta["validation"].get("by_regime", {})
        if not by_regime:
            continue
        cells = "  ".join(f"{r}={_fmt(m['brier_skill'])}" for r, m in sorted(by_regime.items()))
        print(f"  {name:<16} {cells}")

    if args.challenger:
        from evaluator.model.tuning import compare_importances, xgboost_challenger

        print("\nXGBoost challenger (spec 3.1 -- a sanity check, not an ensemble member):")
        for spec in targets:
            X, y, dates, _ = panel.target_frame(spec)
            challenger = xgboost_challenger(X, y, dates, spec)
            lgb_top = [
                f["feature"]
                for f in results[spec.name]["validation"]["pooled_out_of_sample"].get("top_features", [])
            ] or panel.feature_names[:10]
            xgb_top = [f["feature"] for f in challenger["top_features"]]
            agreement = compare_importances(lgb_top, xgb_top)
            print(f"  {spec.name:<16} xgb_skill={_fmt(challenger['metrics']['brier_skill'])}  "
                  f"importance_agreement={agreement['agreement']:.2f}")
            print(f"    {agreement['interpretation']}")

    best = max(
        (m["validation"]["pooled_out_of_sample"]["brier_skill"] or -9, n)
        for n, m in results.items()
    )
    print(f"\nbest target: {best[1]} at {best[0]:+.4f} Brier skill vs baseline")
    if best[0] <= 0.01:
        print("NOTE: no target clears the noise floor. Treat all outputs as no signal.")


if __name__ == "__main__":
    main()
