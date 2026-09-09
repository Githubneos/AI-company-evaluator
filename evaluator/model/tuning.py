"""Hyperparameter search and the XGBoost challenger (spec 3.1, 3.2).

Optuna over the spec's search ranges, scored on purged walk-forward folds --
the same validation the final model is judged by. Tuning against a random split
would select hyperparameters that are good at exploiting leakage, which is worse
than not tuning at all.

The objective is Brier skill against the base rate, not log loss. Log loss can
be driven down by a model that simply learns the class priors well; skill only
improves if the model knows something the priors do not.

The XGBoost challenger is a sanity check rather than an ensemble member (spec
3.1): two GBM implementations given the same data should broadly agree, and a
large divergence in feature importance usually means one of them is fitting an
artefact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluator.config import TargetSpec, ValidationConfig
from evaluator.metrics import evaluate
from evaluator.validation import PurgedWalkForward

log = logging.getLogger(__name__)

MAX_TUNING_ROWS = 600_000


@dataclass
class TuningResult:
    best_params: dict
    best_skill: float
    n_trials: int
    history: list[dict]


def _subsample(X: pd.DataFrame, y: pd.Series, dates: pd.Series, seed: int = 7):
    """Tune on the most recent slice when the panel is large.

    Keeps chronology intact -- a random subsample would break the walk-forward
    structure the folds depend on.
    """
    if len(X) <= MAX_TUNING_ROWS:
        return X, y, dates
    keep = slice(len(X) - MAX_TUNING_ROWS, None)
    return X.iloc[keep], y.iloc[keep], dates.iloc[keep]


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    spec: TargetSpec,
    *,
    n_trials: int = 25,
    validation: ValidationConfig | None = None,
    seed: int = 7,
) -> TuningResult:
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    validation = validation or ValidationConfig(n_splits=3, test_days=252, embargo_days=10)
    X, y, dates = _subsample(X, y, dates, seed)

    splitter = PurgedWalkForward(
        n_splits=validation.n_splits,
        test_size=validation.test_days,
        purge=spec.horizon_days,
        embargo=validation.embargo_days,
    )
    splits = list(splitter.split_panel(dates))
    history: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        params = {
            # Ranges from spec 3.2. max_depth is capped low on purpose: the
            # effective sample size here is regimes, not rows.
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 2000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.9),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 20.0, log=True),
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "num_threads": 4,
        }
        if spec.n_classes == 2:
            params.update({"objective": "binary", "metric": "binary_logloss"})
        else:
            params.update(
                {"objective": "multiclass", "num_class": spec.n_classes, "metric": "multi_logloss"}
            )

        skills = []
        for train_idx, test_idx in splits:
            booster = lgb.train(
                params,
                lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx]),
                num_boost_round=300,
            )
            raw = np.asarray(booster.predict(X.iloc[test_idx]), dtype=float)
            proba = np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw

            priors = np.bincount(y.iloc[train_idx], minlength=spec.n_classes).astype(float)
            priors /= priors.sum()
            metrics = evaluate(y.iloc[test_idx].to_numpy(), proba, priors)
            skills.append(metrics["brier_skill"] or 0.0)

        skill = float(np.mean(skills))
        history.append({"trial": trial.number, "skill": skill, "params": dict(params)})
        return skill

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info("%s: best Brier skill %.4f over %d trials", spec.name, study.best_value, n_trials)
    return TuningResult(study.best_params, float(study.best_value), n_trials, history)


def xgboost_challenger(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    spec: TargetSpec,
    validation: ValidationConfig | None = None,
) -> dict:
    """Train XGBoost on the same folds and compare (spec 3.1).

    Agreement is reassurance; divergence in the top features is a signal that
    one model is fitting something the other does not see, and worth chasing
    before either is trusted.
    """
    import xgboost as xgb

    validation = validation or ValidationConfig(n_splits=3, test_days=252, embargo_days=10)
    splitter = PurgedWalkForward(
        n_splits=validation.n_splits,
        test_size=validation.test_days,
        purge=spec.horizon_days,
        embargo=validation.embargo_days,
    )

    oos_true, oos_proba = [], []
    importances = np.zeros(X.shape[1])

    for train_idx, test_idx in splitter.split_panel(dates):
        params = {
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "nthread": 4,
            "seed": 7,
        }
        if spec.n_classes == 2:
            params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            params.update(
                {
                    "objective": "multi:softprob",
                    "num_class": spec.n_classes,
                    "eval_metric": "mlogloss",
                }
            )

        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        booster = xgb.train(params, dtrain, num_boost_round=300)

        raw = booster.predict(xgb.DMatrix(X.iloc[test_idx]))
        proba = np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw
        oos_true.append(y.iloc[test_idx].to_numpy())
        oos_proba.append(proba)

        scores = booster.get_score(importance_type="gain")
        for i, name in enumerate(X.columns):
            importances[i] += scores.get(name, 0.0)

    priors = np.bincount(y, minlength=spec.n_classes).astype(float)
    priors /= priors.sum()
    metrics = evaluate(np.concatenate(oos_true), np.vstack(oos_proba), priors)

    order = np.argsort(-importances)[:10]
    return {
        "metrics": metrics,
        "top_features": [
            {"feature": X.columns[i], "gain": round(float(importances[i]), 2)} for i in order
        ],
    }


def compare_importances(lgb_features: list[str], xgb_features: list[str], k: int = 10) -> dict:
    """Overlap between the two implementations' top-k features."""
    left, right = set(lgb_features[:k]), set(xgb_features[:k])
    overlap = left & right
    return {
        "overlap_count": len(overlap),
        "overlap": sorted(overlap),
        "lightgbm_only": sorted(left - right),
        "xgboost_only": sorted(right - left),
        "agreement": round(len(overlap) / k, 3),
        "interpretation": (
            "Implementations broadly agree on what drives the model."
            if len(overlap) >= k // 2
            else "Implementations disagree on the main drivers; treat feature "
            "attributions with caution until the divergence is understood."
        ),
    }
