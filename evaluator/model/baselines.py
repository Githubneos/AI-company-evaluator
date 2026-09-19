"""How much does the GBM add over simple signals?

Every headline skill number in this project is measured against class priors,
the weakest baseline there is. This module asks the harder question: on the
exact same out-of-sample rows, how much better is the 44-feature model than a
small model that only knows

- ``vol``      -- recent volatility and its trend (volatility clusters), and
- ``earnings`` -- where the company is in its reporting cycle (large moves
  bunch around earnings, which arrive roughly every 63 trading days),
- ``simple``   -- both, plus the VIX level and its change.

The headline comparison is against the *strongest* of the three on each
target. "Simple" is not reliably the strongest -- on some targets adding VIX to
the volatility features makes a small model worse -- and comparing against a
fixed, weaker baseline would flatter the full model. Picking the best of three
fixed baselines on out-of-sample data slightly favours the baseline; that is
the conservative direction to err in.

The baselines are shallow LightGBM models rather than logistic regressions
because the earnings clock is non-monotonic: risk peaks some 60-70 days after
the last report. A linear baseline would understate what the simple signals
can do and flatter the full model.

They are fitted through `walk_forward`, the same purged, embargoed routine the
real model was validated with, and then *checked* against the real model's
saved out-of-fold predictions: identical labels, identical dates, identical
fold windows, or the comparison is refused. An ablation scored on different
rows than the model it is compared with would be worse than none.
"""

from __future__ import annotations

import gc
import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from evaluator.config import TargetSpec
from evaluator.features.build import FEATURE_DESCRIPTIONS
from evaluator.features.store import FeaturePanel
from evaluator.io import atomic_write_json
from evaluator.metrics import brier_score, evaluate
from evaluator.model.train import MIN_REGIME_ROWS, MODEL_DIR, PanelTrainConfig, walk_forward
from evaluator.regimes import regime_for

log = logging.getLogger(__name__)

VOL_FEATURES = ("vol_20", "vol_60", "vol_ratio_20_60", "atr_14_pct")
EARNINGS_FEATURES = ("days_since_earnings_result", "days_since_periodic_report")
MARKET_FEATURES = ("vix", "vix_chg_20")

BASELINE_FEATURES: dict[str, tuple[str, ...]] = {
    "vol": VOL_FEATURES,
    "earnings": EARNINGS_FEATURES,
    "simple": VOL_FEATURES + EARNINGS_FEATURES + MARKET_FEATURES,
}

#: A small tree on a handful of features: enough capacity for the earnings
#: clock's non-monotonic shape, not enough to become a second full model.
BASELINE_PARAMS = {"num_leaves": 15, "max_depth": 4}

#: How each baseline is named in prose.
BASELINE_LABELS = {
    "vol": "volatility-only",
    "earnings": "earnings-cycle",
    "simple": "volatility + earnings + VIX",
}

BOOTSTRAP_REPS = 2000
SUMMARY_METRICS = ("brier_skill", "log_loss_skill", "macro_auc", "average_precision", "precision_at_5pct")

_unknown = sorted({f for cols in BASELINE_FEATURES.values() for f in cols} - set(FEATURE_DESCRIPTIONS))
if _unknown:  # pragma: no cover - guards edits to the lists above
    raise RuntimeError(f"baseline features not defined in features.build: {_unknown}")


class BaselineMismatch(RuntimeError):
    """The baseline folds do not reproduce the saved model's out-of-fold rows."""


def _thin_to_stride(X, y, dates, stride: int):
    """Keep every `stride`-th date, exactly as `_thin_by_date` did for the model."""
    if stride <= 1:
        return X, y, dates
    keep = set(dates.unique()[::stride])
    mask = dates.isin(keep).to_numpy()
    return X.loc[mask], y.loc[mask], dates.loc[mask]


def _load_model_artifacts(spec: TargetSpec) -> tuple[dict, np.lib.npyio.NpzFile]:
    model_dir = MODEL_DIR / spec.name
    meta_path, oos_path = model_dir / "metadata.json", model_dir / "oos_predictions.npz"
    if not meta_path.exists() or not oos_path.exists():
        raise FileNotFoundError(
            f"no trained {spec.name} model with out-of-fold predictions in {model_dir}. "
            "Run: python -m scripts.train"
        )
    return json.loads(meta_path.read_text()), np.load(oos_path)


def _check_alignment(spec: TargetSpec, wf, saved, meta: dict) -> None:
    hint = (
        "Rerun with the --n-splits / --test-days / --embargo-days the model was trained "
        "with, or retrain the model on the current panel."
    )
    saved_folds = meta["validation"]["folds"]
    if len(wf.folds) != len(saved_folds):
        raise BaselineMismatch(
            f"{spec.name}: baseline produced {len(wf.folds)} folds, model has {len(saved_folds)}. {hint}"
        )
    for ours, theirs in zip(wf.folds, saved_folds):
        if (ours["test_start"], ours["test_end"]) != (theirs["test_start"], theirs["test_end"]):
            raise BaselineMismatch(
                f"{spec.name}: fold {ours['fold']} tests {ours['test_start']}..{ours['test_end']}, "
                f"model tested {theirs['test_start']}..{theirs['test_end']}. {hint}"
            )
    saved_dates = pd.to_datetime(saved["dates"])
    if len(wf.y) != len(saved["y"]) or not np.array_equal(wf.y, saved["y"]) or not np.array_equal(
        wf.dates.to_numpy(dtype="datetime64[ns]"), saved_dates.to_numpy(dtype="datetime64[ns]")
    ):
        raise BaselineMismatch(
            f"{spec.name}: out-of-fold labels or dates differ from the saved model's, so the "
            f"panel has changed since training. {hint}"
        )


def _skill(y: np.ndarray, proba: np.ndarray, priors: np.ndarray) -> dict:
    full = evaluate(y, proba, priors)
    return {k: full[k] for k in SUMMARY_METRICS}


def _incremental(brier_model: float, brier_reference: float) -> float | None:
    """1 - Brier(model)/Brier(reference): the model's skill *over* the reference."""
    return float(1.0 - brier_model / brier_reference) if brier_reference > 0 else None


def paired_fold_stats(
    fold_n: np.ndarray, brier_model: np.ndarray, brier_reference: np.ndarray, *, seed: int = 0
) -> dict:
    """Fold-level comparison of the model against a reference.

    Rows inside a fold share overlapping label windows and one market
    environment, so they are nowhere near independent. The fold, not the row,
    is the resampling unit.
    """
    fold_n = np.asarray(fold_n, dtype=float)
    bm, br = np.asarray(brier_model, dtype=float), np.asarray(brier_reference, dtype=float)
    per_fold = 1.0 - bm / br
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(fold_n), size=(BOOTSTRAP_REPS, len(fold_n)))
    # Pool each resample by row count, as the headline number is pooled.
    boot = 1.0 - (fold_n[draws] * bm[draws]).sum(axis=1) / (fold_n[draws] * br[draws]).sum(axis=1)
    return {
        "pooled": float(1.0 - (fold_n * bm).sum() / (fold_n * br).sum()),
        "fold_wins": int((per_fold > 0).sum()),
        "n_folds": int(len(fold_n)),
        "fold_mean": float(per_fold.mean()),
        "ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
    }


def verdict(
    model_skill: float | None, reference_skill: float | None, stats: dict, reference: str = "simple"
) -> str:
    """Plain language, in the same spirit as `fusion._quality_verdict`: never soften."""
    lo, hi = stats["ci90"]
    wins, n, edge = stats["fold_wins"], stats["n_folds"], stats["pooled"]
    label = BASELINE_LABELS.get(reference, reference)
    against = f"the best simple baseline ({label})"
    if model_skill and model_skill > 0 and reference_skill is not None and reference_skill > model_skill:
        share = f" The {label} baseline alone scores higher ({reference_skill:+.4f} vs {model_skill:+.4f})."
    elif model_skill and model_skill > 0 and reference_skill is not None and reference_skill > 0:
        share = f" The {label} baseline alone reaches {reference_skill / model_skill:.0%} of its skill."
    else:
        share = ""
    if lo > 0 and wins >= 0.7 * n:
        head = (
            f"Adds measurable skill over {against}: {edge:+.4f} Brier, "
            f"90% CI {lo:+.4f}..{hi:+.4f}, better in {wins}/{n} folds."
        )
    elif lo > 0:
        head = (
            f"Small but real edge over {against} ({edge:+.4f} Brier, 90% CI "
            f"{lo:+.4f}..{hi:+.4f}), and not a consistent one: better in only {wins}/{n} folds."
        )
    elif edge > 0:
        head = (
            f"Its edge over {against} ({edge:+.4f} Brier, 90% CI {lo:+.4f}..{hi:+.4f}, "
            f"better in {wins}/{n} folds) is not distinguishable from zero."
        )
    else:
        head = (
            f"Worse than {against}: {edge:+.4f} Brier, better in only {wins}/{n} folds. "
            "The extra features are not earning their place."
        )
    return head + share


def evaluate_baselines(panel: FeaturePanel, spec: TargetSpec, cfg: PanelTrainConfig) -> dict:
    """Fit every baseline on the model's own folds and compare. Writes baselines.json."""
    meta, saved = _load_model_artifacts(spec)
    stride = int(meta.get("date_stride", 1))

    wanted = sorted({f for cols in BASELINE_FEATURES.values() for f in cols})
    missing = [f for f in wanted if f not in panel.feature_names]
    if missing:
        raise ValueError(f"panel is missing baseline features {missing}; rebuild it with scripts.build_panel")

    # A panel view holding only the baseline columns keeps the working copy small.
    narrow = FeaturePanel(panel.frame, wanted, panel.label_names, panel.schema)
    X, y, dates, _ = narrow.target_frame(spec)
    X, y, dates = _thin_to_stride(X, y, dates, stride)

    model_proba = np.asarray(saved["probabilities"], dtype=float)
    results: dict[str, object] = {}
    for name, cols in BASELINE_FEATURES.items():
        log.info("%s: baseline %s on %d features", spec.name, name, len(cols))
        wf = walk_forward(X[list(cols)], y, dates, spec, cfg, stride, params_override=BASELINE_PARAMS)
        _check_alignment(spec, wf, saved, meta)
        results[name] = wf
        gc.collect()

    # Alignment is verified, so every baseline shares these rows, folds and priors.
    first = next(iter(results.values()))
    y_oos, priors, oos_dates = first.y, first.priors, first.dates
    fold_n = np.array([f["n"] for f in first.folds])
    bounds = np.concatenate([[0], np.cumsum(fold_n)])
    spans = list(zip(bounds[:-1], bounds[1:]))

    def fold_briers(proba: np.ndarray) -> np.ndarray:
        return np.array([brier_score(y_oos[a:b], proba[a:b]) for a, b in spans])

    pooled = {"model": _skill(y_oos, model_proba, priors)}
    pooled.update({name: _skill(y_oos, wf.proba, priors) for name, wf in results.items()})

    model_fb = fold_briers(model_proba)
    incremental = {
        f"vs_{name}": paired_fold_stats(fold_n, model_fb, fold_briers(wf.proba))
        for name, wf in results.items()
    }
    reference = min(results, key=lambda name: brier_score(y_oos, results[name].proba))
    ref_proba = results[reference].proba
    ref_fb = fold_briers(ref_proba)

    folds = []
    for f, (a, b), bm, br in zip(first.folds, spans, model_fb, ref_fb):
        base = brier_score(y_oos[a:b], priors[a:b])
        folds.append(
            {
                "fold": f["fold"],
                "test_start": f["test_start"],
                "test_end": f["test_end"],
                "n": int(f["n"]),
                "model_brier_skill": _incremental(bm, base),
                "reference_brier_skill": _incremental(br, base),
                "incremental": _incremental(bm, br),
            }
        )

    by_regime = {}
    regimes = regime_for(oos_dates).to_numpy()
    for regime in pd.unique(regimes):
        mask = regimes == regime
        if mask.sum() < MIN_REGIME_ROWS or len(np.unique(y_oos[mask])) < 2:
            continue
        base = brier_score(y_oos[mask], priors[mask])
        bm = brier_score(y_oos[mask], model_proba[mask])
        br = brier_score(y_oos[mask], ref_proba[mask])
        by_regime[str(regime)] = {
            "n": int(mask.sum()),
            "model_brier_skill": _incremental(bm, base),
            "reference_brier_skill": _incremental(br, base),
            "incremental": _incremental(bm, br),
        }

    report = {
        "target": spec.name,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_schema": meta.get("schema"),
        "date_stride": stride,
        "n_rows": int(len(y_oos)),
        #: The strongest baseline on this target; the headline comparison.
        "reference": reference,
        "features": {name: list(cols) for name, cols in BASELINE_FEATURES.items()},
        "baseline_params": BASELINE_PARAMS,
        "pooled": pooled,
        "incremental": incremental,
        "folds": folds,
        "by_regime": by_regime,
        "verdict": verdict(
            pooled["model"]["brier_skill"], pooled[reference]["brier_skill"],
            incremental[f"vs_{reference}"], reference,
        ),
    }
    atomic_write_json(report, MODEL_DIR / spec.name / "baselines.json")
    return report


def load_baselines(target: str) -> dict | None:
    path = MODEL_DIR / target / "baselines.json"
    return json.loads(path.read_text()) if path.exists() else None


__all__ = [
    "BASELINE_FEATURES",
    "BaselineMismatch",
    "evaluate_baselines",
    "load_baselines",
    "paired_fold_stats",
    "verdict",
]
