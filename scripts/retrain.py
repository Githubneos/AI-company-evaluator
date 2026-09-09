"""CLI: cadence- and drift-driven retraining (spec 6.3).

    python -m scripts.retrain --check     # decide only, do not train
    python -m scripts.retrain             # train if the decision says so
    python -m scripts.retrain --force

Three triggers: quarterly full retrain, monthly incremental, and out-of-cycle
when PSI crosses 0.2 on any feature. The drift trigger matters because the
calendar does not know when the market changed.
"""

from __future__ import annotations

import argparse
import json
import logging

from evaluator.features.store import load_panel
from evaluator.retrain import RetrainDecision, decide, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report the decision only")
    parser.add_argument("--force", action="store_true", help="retrain regardless of cadence")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    panel = load_panel()
    decision = decide(panel.frame, panel.feature_names)

    print(f"decision: {decision.mode}  ({'run' if decision.should_run else 'hold'})")
    print(f"reason:   {decision.reason}")
    if decision.drift:
        print("\nworst drift (PSI):")
        for row in decision.drift[:8]:
            print(f"  {row['feature']:<32} {row['psi']:>7.4f}  {row['verdict']}")

    if args.check:
        return

    if args.force:
        decision = RetrainDecision(True, "full", "Forced by --force.", decision.drift)

    result = run(decision)
    print("\n" + json.dumps({k: v for k, v in result.items() if k != "drift"}, indent=2, default=str))


if __name__ == "__main__":
    main()
