"""CLI: add out-of-fold tickers to models trained before they were recorded.

    python -m scripts.backfill_oos_tickers            # dry run
    python -m scripts.backfill_oos_tickers --apply

The promotion gate compares a candidate with the incumbent on the rows both
scored out of fold, which needs (ticker, date) keys on each side. Models
trained before `train_target` recorded tickers have only dates, so their
`oos_predictions.npz` is rewritten here: the walk-forward split is replayed on
the current panel, and the recovered labels must match the saved ones exactly
before anything is written. A mismatch means the panel changed since training,
and the file is left alone.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from evaluator.config import ValidationConfig, default_targets
from evaluator.features.store import load_panel
from evaluator.io import atomic_write_bytes
from evaluator.model.registry import CANDIDATE, model_dir
from evaluator.model.train import PanelTrainConfig
from evaluator.model.vol_benchmarks import replay_rows

log = logging.getLogger(__name__)


def backfill(spec, *, apply: bool) -> dict:
    path = model_dir(spec.name, CANDIDATE) / "oos_predictions.npz"
    meta_path = model_dir(spec.name, CANDIDATE) / "metadata.json"
    if not path.exists() or not meta_path.exists():
        return {"target": spec.name, "status": "no model"}

    saved = np.load(path, allow_pickle=False)
    if "tickers" in saved:
        return {"target": spec.name, "status": "already has tickers", "n": int(len(saved["y"]))}

    meta = json.loads(meta_path.read_text())
    folds = meta["validation"]["folds"]
    cfg = PanelTrainConfig(
        validation=ValidationConfig(n_splits=len(folds), test_days=252, embargo_days=10)
    )
    rows = replay_rows(load_panel(targets=[spec]), spec, cfg, int(meta.get("date_stride", 1)))
    order = np.concatenate([test for _, test in rows.folds])
    y, dates = rows.y[order], rows.dates.iloc[order]

    if len(y) != len(saved["y"]) or not np.array_equal(y, saved["y"]):
        return {"target": spec.name, "status": "labels differ; panel changed since training"}
    if not np.array_equal(
        dates.to_numpy(dtype="datetime64[ns]"), np.asarray(saved["dates"], dtype="datetime64[ns]")
    ):
        return {"target": spec.name, "status": "dates differ; panel changed since training"}

    if apply:
        atomic_write_bytes(
            lambda tmp: np.savez_compressed(
                tmp,
                y=saved["y"],
                probabilities=saved["probabilities"],
                dates=saved["dates"],
                tickers=rows.tickers[order].astype("U12"),
            ),
            path,
        )
    return {"target": spec.name, "status": "written" if apply else "would write", "n": int(len(y))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    targets = default_targets()
    if args.targets:
        wanted = set(args.targets)
        targets = [t for t in targets if t.name in wanted]

    for spec in targets:
        result = backfill(spec, apply=args.apply)
        print(f"{result['target']:<16} {result['status']}" + (f" ({result['n']:,} rows)" if "n" in result else ""))

    if not args.apply:
        print("\n(dry run -- pass --apply to rewrite the files)")


if __name__ == "__main__":
    main()
