"""Scoring across the six models, with per-prediction attribution.

Attribution uses LightGBM's built-in `pred_contrib=True`, which is TreeSHAP
computed inside the booster. Same values as the `shap` package for tree models,
without constructing an explainer on every request.

Serving builds features through `build_dataset` -- the identical path training
used -- and then checks the resulting columns against the schema hash the model
was trained on. A silent reindex of mismatched columns is how a model ends up
scoring inputs that mean something different from what it learned.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluator.config import DIRECTION, REL_DIRECTION, LabelConfig, TargetSpec, default_targets
from evaluator.dataset import build_dataset
from evaluator.features.build import FEATURE_DESCRIPTIONS
from evaluator.features.store import align_to_schema
from evaluator.labels import label_scale, relative_label_scale
from evaluator.model import registry

log = logging.getLogger(__name__)


class ModelNotTrained(FileNotFoundError):
    pass


@dataclass
class LoadedModel:
    booster: lgb.Booster
    metadata: dict

    @property
    def feature_names(self) -> list[str]:
        return self.metadata["feature_names"]

    @property
    def n_classes(self) -> int:
        return len(self.metadata["class_names"])


@lru_cache(maxsize=12)
def load_model(target_name: str) -> LoadedModel:
    # Production only: training overwrites candidates, so serving from there
    # would put every retrain straight into traffic, gate or no gate.
    out_dir = registry.model_dir(target_name, registry.PRODUCTION)
    model_path, meta_path = out_dir / "model.txt", out_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotTrained(
            f"no promoted model for {target_name!r} in {out_dir}. Train it, then run: "
            "python -m scripts.promote --all --apply"
        )
    return LoadedModel(
        booster=lgb.Booster(model_file=str(model_path)),
        metadata=json.loads(meta_path.read_text()),
    )


def available_targets() -> list[str]:
    return registry.available_targets(registry.PRODUCTION)


def _as_proba(raw, n_classes: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if n_classes == 2 and raw.ndim == 1:
        return np.column_stack([1.0 - raw, raw])
    return raw


def top_contributions(
    model: LoadedModel, row: pd.DataFrame, predicted_class: int, k: int = 5
) -> list[dict]:
    contrib = np.asarray(model.booster.predict(row, pred_contrib=True), dtype=float)[0]
    names = model.feature_names
    block = len(names) + 1

    # Binary boosters emit a single contribution block; multiclass emits one per
    # class, each ending in that class's base value.
    if model.n_classes == 2 or contrib.size == block:
        class_contrib = contrib[: len(names)]
        if model.n_classes == 2 and predicted_class == 0:
            class_contrib = -class_contrib
    else:
        start = predicted_class * block
        class_contrib = contrib[start : start + len(names)]

    order = np.argsort(-np.abs(class_contrib))[:k]
    values = row.iloc[0]
    return [
        {
            "feature": names[i],
            "description": FEATURE_DESCRIPTIONS.get(names[i], ""),
            "value": None if pd.isna(values[names[i]]) else round(float(values[names[i]]), 6),
            "shap_contribution": round(float(class_contrib[i]), 6),
            "direction": "increases" if class_contrib[i] > 0 else "decreases",
        }
        for i in order
    ]


def score_target(
    target_name: str, features_row: pd.DataFrame, *, with_attribution: bool = True
) -> dict:
    model = load_model(target_name)
    row = align_to_schema(features_row, model.feature_names, model.metadata["schema"])

    proba = _as_proba(model.booster.predict(row), model.n_classes)[0]
    predicted = int(np.argmax(proba))
    class_names = {int(k): v for k, v in model.metadata["class_names"].items()}
    priors = np.asarray(model.metadata["class_priors"], dtype=float)
    validation = model.metadata["validation"]["pooled_out_of_sample"]

    result = {
        "target": target_name,
        "horizon_days": model.metadata["spec"]["horizon_days"],
        "kind": model.metadata["spec"]["kind"],
        "probabilities": {class_names[c]: round(float(proba[c]), 4) for c in class_names},
        "baseline_probabilities": {class_names[c]: round(float(priors[c]), 4) for c in class_names},
        "lift_over_baseline": {
            class_names[c]: round(float(proba[c] / max(priors[c], 1e-9)), 3) for c in class_names
        },
        "predicted_class": class_names[predicted],
        "skill": {
            "brier_skill": validation.get("brier_skill"),
            "macro_auc": validation.get("macro_auc"),
            "average_precision": validation.get("average_precision"),
            "precision_at_5pct": validation.get("precision_at_5pct"),
            "calibration_error": validation.get("calibration_error"),
        },
    }
    if with_attribution:
        result["top_features"] = top_contributions(model, row, predicted)
    return result


def score_ticker(
    ticker: str,
    *,
    lookback_start: str = "2015-01-01",
    targets: list[str] | None = None,
    use_cache: bool = True,
) -> dict:
    """Score the most recent bar for `ticker` across every trained target."""
    names = targets or available_targets()
    if not names:
        raise ModelNotTrained("no trained models found. Run: python -m scripts.train")

    primary = load_model(names[0])
    label_cfg = LabelConfig(
        horizon_days=primary.metadata["spec"]["horizon_days"],
        threshold_sigmas=primary.metadata["spec"]["threshold_sigmas"],
        vol_lookback=primary.metadata["spec"]["vol_lookback"],
    )

    dataset = build_dataset(
        ticker, lookback_start, None, label_cfg, use_cache=use_cache, targets=default_targets()
    )
    row = dataset.latest_row()
    scale = label_scale(dataset.prices, label_cfg).iloc[-1]
    # A sector-relative call is judged against the volatility of the *excess*
    # return, not the stock's own: the same yardstick its label was built with.
    relative_scale = relative_label_scale(dataset.prices, dataset.sector_prices, label_cfg)

    scored = {}
    for name in names:
        try:
            scored[name] = score_target(name, row)
            spec_kind = scored[name]["kind"]
            horizon = scored[name]["horizon_days"]
            base = relative_scale if spec_kind == REL_DIRECTION else scale
            scored[name]["label_scale"] = (
                None if base is None or pd.isna(base) else float(base * np.sqrt(horizon / label_cfg.horizon_days))
            )
        except Exception as exc:  # noqa: BLE001 - one model must not sink the response
            log.warning("scoring %s failed for %s: %s", name, ticker, exc)

    return {
        "ticker": ticker,
        "as_of": str(dataset.features.index[-1].date()),
        "last_close": round(float(dataset.prices["close"].iloc[-1]), 4),
        "label_scale": None if pd.isna(scale) else float(scale),
        "move_threshold_pct": None if pd.isna(scale) else float(scale * label_cfg.threshold_sigmas),
        "threshold_sigmas": label_cfg.threshold_sigmas,
        "horizon_days": label_cfg.horizon_days,
        "targets": scored,
        "features_row": row,
        "prices": dataset.prices,
        "caveats": primary.metadata["data_caveats"],
    }


def primary_target(scored: dict, kind: str = DIRECTION, horizon: int = 5) -> dict | None:
    return scored.get("targets", {}).get(TargetSpec(horizon, kind).name)
