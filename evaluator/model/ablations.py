"""Which feature groups earn their place?

The baseline study found that a four-feature volatility model reaches most of
every model's skill. This module asks the follow-up one group at a time: fitted
on the model's own folds and scored on identical rows (via
`baselines.fit_feature_sets`), does adding a group to the volatility features
improve out-of-sample Brier, and is the improvement consistent across folds?

Sets, all with the same small-tree parameters as the baselines:

- ``vol``                      the reference
- ``vol+<group>``              one per non-volatility group
- ``vol+macro-vix_level``      macro without the VIX *level*, which the baseline
                               study suspected of hurting
- ``all``                      every feature, small trees -- if this beats the
                               deep production model, the problem is capacity
                               (overfitting), not the features

Groups partition the feature set exactly; an unassigned feature is an error,
so a feature added later cannot silently escape the ablation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from evaluator.config import TargetSpec
from evaluator.data.fundamentals import FUNDAMENTAL_FEATURES
from evaluator.features.store import FeaturePanel
from evaluator.io import atomic_write_json
from evaluator.metrics import brier_score
from evaluator.model.baselines import (
    VOL_FEATURES,
    _incremental,
    _load_model_artifacts,
    _skill,
    fit_feature_sets,
    paired_fold_stats,
)
from evaluator.model.train import MIN_REGIME_ROWS, MODEL_DIR, PanelTrainConfig
from evaluator.regimes import regime_for

log = logging.getLogger(__name__)

REFERENCE = "vol"
GROUP_ORDER = ("technical", "sector", "events", "macro", "fundamentals")
GROUP_LABELS = {
    "volatility": "Volatility",
    "technical": "Technical & returns",
    "sector": "Sector-relative",
    "events": "SEC events",
    "macro": "Macro & regime",
    "fundamentals": "Fundamentals",
}
MACRO_EXACT = {"vix", "vix_chg_20", "curve_slope", "curve_slope_chg_60", "recession"}
TECHNICAL_EXACT = {
    "ret_5", "ret_20", "ret_60", "ret_252", "volume_z_60", "rsi_14", "macd_hist_pct",
    "bb_position", "drawdown_from_252h", "runup_from_252l", "beta_60",
    "excess_ret_5", "excess_ret_20", "excess_ret_60",
}


def group_of(feature: str) -> str:
    """The one group a feature belongs to. Raises for anything unrecognised."""
    if feature in VOL_FEATURES:
        return "volatility"
    if feature in FUNDAMENTAL_FEATURES:
        return "fundamentals"
    if feature.startswith(("days_since_", "dividend_")) or feature == "event_density_90d":
        return "events"
    if feature.startswith("sector_") or feature == "vol_vs_sector":
        return "sector"
    if feature in MACRO_EXACT:
        return "macro"
    if feature in TECHNICAL_EXACT:
        return "technical"
    raise ValueError(
        f"feature {feature!r} has no ablation group; add it to evaluator.model.ablations.group_of"
    )


def feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    duplicates = sorted({f for f in feature_names if feature_names.count(f) > 1})
    if duplicates:
        raise ValueError(f"duplicate feature names: {duplicates}")
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        groups.setdefault(group_of(name), []).append(name)  # group_of raises if unassigned
    return groups


def ablation_sets(feature_names: list[str]) -> dict[str, list[str]]:
    groups = feature_groups(feature_names)
    vol = groups.get("volatility", [])
    if sorted(vol) != sorted(VOL_FEATURES):
        raise ValueError(f"panel lacks the volatility reference features: {sorted(set(VOL_FEATURES) - set(vol))}")
    sets: dict[str, list[str]] = {REFERENCE: list(vol)}
    for group in GROUP_ORDER:
        if groups.get(group):
            sets[f"vol+{group}"] = vol + groups[group]
    if "vix" in groups.get("macro", []):
        sets["vol+macro-vix_level"] = vol + [f for f in groups["macro"] if f != "vix"]
    sets["all"] = list(feature_names)
    return sets


#: Smallest Brier edge worth a feature group's complexity. The production
#: models' total skill is +0.016..+0.044, so +0.002 is 5-10% of it; a
#: statistically "significant" +0.0005 is not a reason to keep ten features.
MIN_EDGE = 0.002
#: Family-wise confidence across the groups compared in one run.
FAMILY_LEVEL = 0.90


def classify(stats: dict) -> str:
    """Plain verdict for one set's edge over the volatility reference.

    Uses the multiple-comparison-adjusted interval (``ci``) and a minimum
    practical effect: with five groups tested at 90% each, one noise group
    clearing zero by chance is the expected outcome, not a finding.
    """
    lo, hi = stats["ci"]
    if lo > 0 and stats["pooled"] >= MIN_EDGE and stats["fold_wins"] >= 0.7 * stats["n_folds"]:
        return "earns its place"
    if lo > 0:
        return "small, inconsistent gain"
    if hi < 0:
        return "hurts"
    return "no measurable effect"


#: Typical calendar days between earnings reports. Used only to say whether the
#: *next* report is expected inside the horizon -- which is knowable at t from
#: the last report's date, unlike the actual date, which is not.
EARNINGS_CYCLE_DAYS = 91
#: Trading days are about 0.7 of calendar days.
CALENDAR_PER_TRADING_DAY = 1.45


def earnings_window_diagnostic(panel: FeaturePanel, spec: TargetSpec, cfg: PanelTrainConfig) -> dict:
    """Model skill where the next earnings report is expected inside the horizon.

    Asking whether an actual report *fell* in the window would be a look-ahead.
    What is knowable at t is the company's own clock: days since the last
    report. Rows where the next one is due before the horizon closes are the
    interesting ones, and this splits the model's own out-of-fold predictions
    between them and everything else.
    """
    from evaluator.model.vol_benchmarks import replay_rows

    feature = "days_since_earnings_result"
    if feature not in panel.feature_names:
        return {"available": False, "reason": f"{feature} is not in the panel"}

    meta, saved = _load_model_artifacts(spec)
    rows = replay_rows(panel, spec, cfg, int(meta.get("date_stride", 1)))
    order = np.concatenate([test for _, test in rows.folds])
    y, proba = rows.y[order].astype(int), np.asarray(saved["probabilities"], dtype=float)
    if len(y) != len(proba) or not np.array_equal(y, saved["y"]):
        return {"available": False, "reason": "out-of-fold rows do not match the saved model"}

    values = (
        panel.frame.set_index(["ticker", "date"])[feature]
        .reindex(pd.MultiIndex.from_arrays([rows.tickers[order], pd.DatetimeIndex(rows.dates.iloc[order])]))
        .to_numpy(dtype=float)
    )
    horizon_days = spec.horizon_days * CALENDAR_PER_TRADING_DAY
    # Next report due before the horizon closes, and the last one actually happened.
    due = (values < 5000) & (values + horizon_days >= EARNINGS_CYCLE_DAYS)

    priors = np.vstack(
        [np.tile(np.bincount(rows.y[tr].astype(int), minlength=spec.n_classes) / len(tr), (len(te), 1))
         for tr, te in rows.folds]
    )
    out = {"available": True, "cycle_days": EARNINGS_CYCLE_DAYS, "n_rows": int(len(y))}
    for name, mask in (("earnings_expected", due), ("quiet_period", ~due)):
        if mask.sum() < MIN_REGIME_ROWS or len(np.unique(y[mask])) < 2:
            out[name] = {"n": int(mask.sum()), "skill": None}
            continue
        out[name] = {"n": int(mask.sum()), "share": float(mask.mean()), **_skill(y[mask], proba[mask], priors[mask])}
    return out


def evaluate_ablations(panel: FeaturePanel, spec: TargetSpec, cfg: PanelTrainConfig) -> dict:
    """Fit every ablation set on the model's folds, compare with `vol`. Writes ablations.json."""
    sets = ablation_sets(panel.feature_names)
    fitted = fit_feature_sets(panel, spec, cfg, sets)
    y, priors = fitted.y, fitted.priors
    ref_proba = fitted.results[REFERENCE].proba
    ref_fb = fitted.fold_briers(ref_proba)
    regimes = regime_for(fitted.dates).to_numpy()

    def regime_edges(proba: np.ndarray) -> dict:
        out = {}
        for regime in sorted(set(regimes)):
            mask = regimes == regime
            if mask.sum() < MIN_REGIME_ROWS or len(np.unique(y[mask])) < 2:
                continue
            out[str(regime)] = _incremental(brier_score(y[mask], proba[mask]), brier_score(y[mask], ref_proba[mask]))
        return out

    # Bonferroni across every set compared against the reference in this run.
    n_compared = max(1, len(sets) - 1)
    level = 1 - (1 - FAMILY_LEVEL) / n_compared

    def describe(proba: np.ndarray, features: list[str] | None) -> dict:
        stats = paired_fold_stats(fitted.fold_n, fitted.fold_briers(proba), ref_fb, level=level)
        entry = {
            "pooled": _skill(y, proba, priors),
            "edge_vs_vol": stats,
            "verdict": classify(stats),
            "by_regime": regime_edges(proba),
        }
        if features is not None:
            entry["features"] = features
            entry["n_features"] = len(features)
        return entry

    results = {name: describe(wf.proba, sets[name]) for name, wf in fitted.results.items()}
    results[REFERENCE]["verdict"] = "reference"
    model = describe(fitted.model_proba, None)

    groups = feature_groups(panel.feature_names)
    earns = [g for g in GROUP_ORDER if results.get(f"vol+{g}", {}).get("verdict") == "earns its place"]
    recommended = list(VOL_FEATURES) + [f for g in earns for f in groups[g]]

    report = {
        "target": spec.name,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_schema": fitted.meta.get("schema"),
        "reference": REFERENCE,
        "min_edge": MIN_EDGE,
        "ci_level": level,
        "n_rows": int(len(y)),
        "groups": groups,
        "group_labels": GROUP_LABELS,
        "sets": results,
        "model": model,
        "earnings_window": earnings_window_diagnostic(panel, spec, cfg),
        "groups_that_earn_their_place": earns,
        "recommended_features": recommended,
    }
    atomic_write_json(report, MODEL_DIR / spec.name / "ablations.json")
    return report


def load_ablations(target: str) -> dict | None:
    path = MODEL_DIR / target / "ablations.json"
    return json.loads(path.read_text()) if path.exists() else None


__all__ = ["ablation_sets", "classify", "evaluate_ablations", "feature_groups", "group_of", "load_ablations"]
