"""CLI: train the optional fusion meta-model (spec 5).

    python -m scripts.train_fusion --target magnitude_5d

Trains logistic regression over the GBM's own out-of-fold probabilities under
the same purged walk-forward discipline as Phase 3, then reports whether it
actually beats the GBM alone. It usually will not, and that is the useful
result: combining weak signals mostly produces a more confident weak signal, and
the fusion layer only adopts the meta-model when it clears a real margin.

Sentiment is absent from the training matrix because free news feeds return only
recent headlines -- there is no historical sentiment column to learn from. Until
a news archive exists, this is a placeholder for the version the spec describes.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from evaluator.config import default_targets
from evaluator.fusion_model import save, should_prefer, train_fusion
from evaluator.model.train import MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="magnitude_5d")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    spec = next((t for t in default_targets() if t.name == args.target), None)
    if spec is None:
        raise SystemExit(f"unknown target {args.target!r}")

    # These are predictions made by the appropriate prior walk-forward GBM
    # fold.  Using the final model on its own training rows would let the meta
    # model learn its base model's in-sample overfit and produce a fictitious
    # fusion improvement.
    path = MODEL_DIR / spec.name / "oos_predictions.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Retrain {spec.name} first so its leakage-free "
            "out-of-fold predictions are available."
        )
    with np.load(path) as saved:
        base_proba = np.asarray(saved["probabilities"], dtype=float)
        y = np.asarray(saved["y"], dtype=int)
        dates = pd.Series(pd.to_datetime(saved["dates"]))

    fusion = train_fusion(base_proba, y, dates, spec)
    save(fusion, spec.name)

    metrics = fusion.metrics
    print(json.dumps(
        {
            "target": spec.name,
            "meta_brier_skill": metrics["meta"]["brier_skill"],
            "gbm_alone_brier_skill": metrics["gbm_alone"]["brier_skill"],
            "improvement": metrics["improvement"],
            "has_sentiment": metrics["has_sentiment"],
            "adopt": should_prefer(fusion),
        },
        indent=2,
    ))
    if not should_prefer(fusion):
        print("\nMeta-model does not clear the adoption margin; fusion stays rules-based.")


if __name__ == "__main__":
    main()
