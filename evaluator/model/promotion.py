"""The promotion gate (spec 8.3): what may replace a model that is serving traffic.

Three refusals, in order:

1. **No measured skill.** A candidate that cannot beat its own base rate is
   never promoted, however bad the incumbent is.
2. **Not better on shared ground.** Pooled skill from two different training
   runs is not comparable: a candidate trained on a wider universe or a longer
   history faces different rows. Where both sides have out-of-fold predictions
   keyed by (ticker, date), the candidate must win on the rows they *share*,
   with a bootstrap interval that clears zero. Dates, not rows, are resampled:
   rows within a day share one market move.
3. **Regime regression.** A candidate that improves on average while getting
   worse in one regime is refused, because the regimes it loses in are the
   volatile ones where the score matters most.

Promotion copies the model's whole directory -- booster, metadata, out-of-fold
predictions and every analysis report -- so the evidence travels with it, and
swaps it into place rather than writing over a directory serving is reading.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from evaluator.metrics import brier_score
from evaluator.model.registry import CANDIDATE, PRODUCTION, load_metadata, model_dir

log = logging.getLogger(__name__)

REGRESSION_TOLERANCE = 0.005
MIN_SKILL = 0.0
BOOTSTRAP_REPS = 2000
MIN_COMMON_ROWS = 1000


def _oos(target: str, stage: str) -> dict | None:
    """Out-of-fold predictions keyed by (ticker, date), when the model has them."""
    path = model_dir(target, stage) / "oos_predictions.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    if "tickers" not in data:
        return None  # trained before tickers were recorded
    return {
        "key": pd.MultiIndex.from_arrays(
            [data["tickers"].astype(str), pd.DatetimeIndex(data["dates"])]
        ),
        "y": data["y"],
        "proba": data["probabilities"],
    }


def common_row_comparison(target: str, *, seed: int = 0) -> dict:
    """Paired candidate-vs-incumbent Brier on the rows both scored out of fold."""
    cand, inc = _oos(target, CANDIDATE), _oos(target, PRODUCTION)
    if cand is None or inc is None:
        return {"available": False, "reason": "one side has no out-of-fold tickers recorded"}

    shared = cand["key"].intersection(inc["key"])
    if len(shared) < MIN_COMMON_ROWS:
        return {"available": False, "reason": f"only {len(shared)} rows in common", "n": int(len(shared))}

    ci = pd.Series(np.arange(len(cand["key"])), index=cand["key"]).groupby(level=[0, 1]).first().reindex(shared)
    ii = pd.Series(np.arange(len(inc["key"])), index=inc["key"]).groupby(level=[0, 1]).first().reindex(shared)
    ci, ii = ci.to_numpy(), ii.to_numpy()
    if not np.array_equal(cand["y"][ci], inc["y"][ii]):
        return {"available": False, "reason": "labels disagree on shared rows; the panel's labels changed"}

    y = cand["y"][ci]
    b_cand, b_inc = brier_score(y, cand["proba"][ci]), brier_score(y, inc["proba"][ii])
    edge = float(1.0 - b_cand / b_inc) if b_inc > 0 else None

    # Resample dates, not rows: one day's rows share a market move.
    dates = shared.get_level_values(1)
    unique, codes = np.unique(dates.to_numpy(), return_inverse=True)
    by_date = [np.flatnonzero(codes == d) for d in range(len(unique))]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOTSTRAP_REPS):
        pick = np.concatenate([by_date[i] for i in rng.integers(0, len(by_date), len(by_date))])
        bi = brier_score(y[pick], inc["proba"][ii][pick])
        boot.append(1.0 - brier_score(y[pick], cand["proba"][ci][pick]) / bi if bi > 0 else 0.0)
    lo, hi = (float(np.quantile(boot, q)) for q in (0.05, 0.95))
    return {
        "available": True,
        "n": int(len(shared)),
        "n_dates": int(len(unique)),
        "candidate_brier": float(b_cand),
        "incumbent_brier": float(b_inc),
        "edge": edge,
        "ci90": [lo, hi],
    }


def compare(candidate: dict, incumbent: dict | None, common: dict | None = None) -> dict:
    """Decide whether `candidate` may replace `incumbent`."""
    cand_pooled = candidate["validation"]["pooled_out_of_sample"]
    cand_skill = cand_pooled.get("brier_skill")

    if cand_skill is None or cand_skill <= MIN_SKILL:
        return {
            "promote": False,
            "reason": (
                f"Candidate has no measured skill over baseline (Brier skill {cand_skill}). "
                "A model that cannot beat the base rate is not promotable regardless of the incumbent."
            ),
            "candidate_skill": cand_skill,
        }

    if incumbent is None:
        return {
            "promote": True,
            "reason": f"No incumbent; candidate has positive skill ({cand_skill:+.4f}).",
            "candidate_skill": cand_skill,
        }

    inc_pooled = incumbent["validation"]["pooled_out_of_sample"]
    inc_skill = inc_pooled.get("brier_skill") or -9.0
    verdict = {"candidate_skill": cand_skill, "incumbent_skill": inc_skill, "common_rows": common}

    # Shared rows decide when they exist; pooled skill across different training
    # runs compares different populations and can favour the easier one.
    if common and common.get("available"):
        edge, (lo, _hi) = common["edge"], common["ci90"]
        if edge is None or edge <= 0 or lo <= -REGRESSION_TOLERANCE:
            return {
                **verdict,
                "promote": False,
                "reason": (
                    f"On the {common['n']:,} rows both models scored out of fold, the candidate's "
                    f"edge is {edge:+.4f} (90% CI {lo:+.4f}..{_hi:+.4f}). Not an improvement where "
                    "the comparison is like for like."
                ),
            }
    elif cand_skill <= inc_skill:
        return {
            **verdict,
            "promote": False,
            "reason": f"Candidate skill {cand_skill:+.4f} does not beat incumbent {inc_skill:+.4f}.",
        }

    regressions = []
    for regime, inc_metrics in incumbent["validation"].get("by_regime", {}).items():
        cand_metrics = candidate["validation"].get("by_regime", {}).get(regime)
        if not cand_metrics:
            continue
        before, after = inc_metrics.get("brier_skill"), cand_metrics.get("brier_skill")
        if before is None or after is None:
            continue
        if after < before - REGRESSION_TOLERANCE:
            regressions.append({"regime": regime, "incumbent": round(before, 4), "candidate": round(after, 4)})

    if regressions:
        worst = min(regressions, key=lambda r: r["candidate"] - r["incumbent"])
        return {
            **verdict,
            "promote": False,
            "reason": (
                f"Candidate improves overall ({inc_skill:+.4f} -> {cand_skill:+.4f}) but regresses in "
                f"{len(regressions)} regime(s), worst {worst['regime']}: {worst['incumbent']:+.4f} -> "
                f"{worst['candidate']:+.4f}. Spec 8.3 refuses aggregate-only promotion."
            ),
            "regressions": regressions,
        }

    where = (
        f"on {common['n']:,} shared rows ({common['edge']:+.4f})"
        if common and common.get("available")
        else f"overall ({inc_skill:+.4f} -> {cand_skill:+.4f})"
    )
    return {**verdict, "promote": True, "reason": f"Improves {where} with no regime regression."}


def swap_into_production(target: str) -> None:
    """Copy the candidate directory into production, atomically for readers."""
    source, destination = model_dir(target, CANDIDATE), model_dir(target, PRODUCTION)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.mkdtemp(dir=destination.parent, prefix=f".{target}.")
    incoming = os.path.join(staging, target)
    try:
        shutil.copytree(source, incoming)
        previous = None
        if destination.exists():
            previous = os.path.join(staging, f"{target}.previous")
            os.rename(destination, previous)
        try:
            os.rename(incoming, destination)
        except BaseException:
            if previous is not None:  # put the old one back before giving up
                os.rename(previous, destination)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Serving caches boosters by target; a promotion inside a running process
    # (the scheduler's retrain job) would otherwise keep serving the old one.
    from evaluator.model.predict import load_model

    load_model.cache_clear()


def promote(target: str, *, apply: bool = False) -> dict:
    candidate = load_metadata(target, CANDIDATE)
    if candidate is None:
        return {
            "target": target,
            "promote": False,
            "reason": f"No candidate model at {model_dir(target, CANDIDATE)}.",
        }

    incumbent = load_metadata(target, PRODUCTION)
    common = common_row_comparison(target) if incumbent is not None else None
    verdict = compare(candidate, incumbent, common)
    verdict["target"] = target

    if verdict["promote"] and apply:
        swap_into_production(target)
        verdict["applied"] = True
    return verdict


__all__ = ["common_row_comparison", "compare", "promote", "swap_into_production"]
