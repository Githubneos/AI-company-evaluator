"""Walk-forward training over the cross-sectional panel (spec 3).

Six models: {direction, magnitude} x {1, 5, 20} days. Each is validated with
date-aware purged, embargoed walk-forward folds and reported both pooled and
per market regime.

THREE DELIBERATE CHOICES
------------------------
No class reweighting. The spec suggests `scale_pos_weight` for rare large
moves, but a volatility-scaled 1-sigma threshold leaves the classes near
16/68/16 (direction) or 32/68 (magnitude) rather than severely skewed, and
reweighting trades away exactly what the downstream LLM layer depends on --
probabilities that mean what they say. Under a stricter threshold, reweight and
then recalibrate (isotonic, fit inside each fold).

Early stopping uses the tail of the training block, never the test block. A test
fold that influenced the stopping iteration is no longer held out.

Per-regime reporting is not decoration. Twenty years of daily bars contains only
a handful of independent market environments, and a model can improve on average
while getting worse in the high-volatility regimes where accuracy actually
matters (spec 3.3, 8.3).
"""

from __future__ import annotations

import gc
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluator.config import ARTIFACT_DIR, TargetSpec, ValidationConfig, default_targets
from evaluator.features.store import FeaturePanel, load_panel
from evaluator.metrics import calibration_bins, evaluate
from evaluator.regimes import regime_for
from evaluator.validation import PurgedWalkForward

log = logging.getLogger(__name__)

MODEL_DIR = Path(ARTIFACT_DIR) / "models"
# MLflow 3 no longer writes to the old filesystem store by default.  SQLite is
# still entirely local but supports the current tracking API and preserves runs
# across retraining cycles.
MLFLOW_DB = Path(ARTIFACT_DIR) / "mlflow.db"
MIN_REGIME_ROWS = 250


def _log_to_mlflow(spec: TargetSpec, cfg: "PanelTrainConfig", metadata: dict) -> None:
    """Experiment tracking (spec 7), local file backend -- no server to run.

    Worth having because this system retrains on a schedule: without a record of
    which data and parameters produced which metrics, a regression six retrains
    later is untraceable.
    """
    try:
        import mlflow
    except ImportError:
        return

    try:
        MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
        mlflow.set_experiment("ai-company-evaluator")

        pooled = metadata["validation"]["pooled_out_of_sample"]
        with mlflow.start_run(run_name=spec.name):
            mlflow.log_params(
                {
                    "target": spec.name,
                    "horizon_days": spec.horizon_days,
                    "kind": spec.kind,
                    "schema": metadata["schema"],
                    "train_rows": metadata["train_rows"],
                    "train_tickers": metadata["train_tickers"],
                    **{k: v for k, v in cfg.lgb_params(spec).items() if not isinstance(v, bool)},
                }
            )
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in pooled.items()
                    if isinstance(v, (int, float)) and v is not None
                }
            )
            for regime, metrics in metadata["validation"].get("by_regime", {}).items():
                if metrics.get("brier_skill") is not None:
                    mlflow.log_metric(f"brier_skill_{regime}", float(metrics["brier_skill"]))
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        log.warning("mlflow logging skipped: %s", exc)


@dataclass
class PanelTrainConfig:
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(n_splits=5, test_days=252, embargo_days=10)
    )
    #: Cap on training rows, enforced by keeping every Nth *date* (whole
    #: cross-sections, never partial days).
    #:
    #: This is a statistical decision as much as a memory one. Labels are
    #: forward-window returns, so a row dated t and one dated t+1 share almost
    #: their entire label window -- at the 20-day horizon, 95% of it. Consecutive
    #: dates are therefore near-duplicates that inflate the row count without
    #: adding independent information, and dropping them costs far less than the
    #: count suggests. It also makes the effective sample size honest: the spec's
    #: own warning is that 20 years contains ~8 independent regimes, not 2.5M
    #: independent observations.
    #:
    #: Whole dates are kept together so every cross-section stays intact and
    #: `split_panel`'s date-boundary guarantee still holds.
    max_train_rows: int = 800_000
    num_boost_round: int = 1500
    early_stopping_rounds: int = 75
    learning_rate: float = 0.03
    num_leaves: int = 63
    max_depth: int = 8
    min_data_in_leaf: int = 300
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    seed: int = 7

    def lgb_params(self, spec: TargetSpec) -> dict:
        params = {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
            "num_threads": 4,
            # 127 bins instead of the default 255 halves the binned dataset.
            # These features are noisy enough that the lost resolution costs
            # nothing measurable, and the memory is what lets six models train.
            "max_bin": 127,
        }
        if spec.n_classes == 2:
            params.update({"objective": "binary", "metric": "binary_logloss"})
        else:
            params.update(
                {"objective": "multiclass", "num_class": spec.n_classes, "metric": "multi_logloss"}
            )
        return params


def _priors(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return counts / counts.sum()


def _as_proba(raw: np.ndarray, n_classes: int) -> np.ndarray:
    """LightGBM returns a 1-D positive-class vector for binary objectives."""
    raw = np.asarray(raw, dtype=float)
    if n_classes == 2 and raw.ndim == 1:
        return np.column_stack([1.0 - raw, raw])
    return raw


def _fit_fold(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    train_idx: np.ndarray,
    spec: TargetSpec,
    cfg: PanelTrainConfig,
    stride: int = 1,
    params_override: dict | None = None,
) -> lgb.Booster:
    """Fit one fold, holding out the tail of the training block for stopping."""
    params = {**cfg.lgb_params(spec), **(params_override or {})}
    train_dates = dates.iloc[train_idx]
    unique = train_dates.unique()

    # Inner validation is the last stretch of *dates*, purged by the horizon so
    # the stopping signal is not contaminated by overlapping label windows.
    # Both windows are in retained-date units, so they carry the same stride
    # rescaling as the outer folds -- an unscaled purge here would let the
    # stopping set overlap the label windows it is meant to be separated from.
    horizon = max(1, math.ceil(spec.horizon_days / stride))
    n_valid_days = min(
        max(1, round(cfg.validation.test_days / stride)), max(len(unique) // 5, 1)
    )
    if len(unique) <= n_valid_days + horizon + 5:
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        return lgb.train(params, dtrain, num_boost_round=cfg.num_boost_round // 3)

    valid_start = unique[-n_valid_days]
    purge_end = unique[-n_valid_days - horizon]

    inner_train = train_idx[(train_dates < purge_end).to_numpy()]
    inner_valid = train_idx[(train_dates >= valid_start).to_numpy()]

    if len(inner_train) < 1000 or len(inner_valid) < 100:
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        return lgb.train(params, dtrain, num_boost_round=cfg.num_boost_round // 3)

    dtrain = lgb.Dataset(X.iloc[inner_train], label=y.iloc[inner_train])
    dvalid = lgb.Dataset(X.iloc[inner_valid], label=y.iloc[inner_valid], reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )


def _thin_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    tickers: pd.Series,
    max_rows: int,
):
    """Keep every Nth date until the panel fits `max_rows`.

    Whole cross-sections are kept or dropped together, so every retained date
    still holds all its tickers. Dropping individual rows instead would break
    the cross-sectional structure the sector-relative features assume, and would
    let `split_panel`'s date-boundary guarantee degrade into a row-based one.

    Returns the thinned data plus the stride actually applied.
    """
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y, dates, tickers, 1

    unique_dates = dates.unique()
    stride = max(2, int(np.ceil(len(X) / max_rows)))
    keep_dates = set(unique_dates[::stride])
    mask = dates.isin(keep_dates).to_numpy()

    return X.loc[mask], y.loc[mask], dates.loc[mask], tickers.loc[mask], stride


def _regime_report(
    y_true: np.ndarray, proba: np.ndarray, dates: pd.Series, priors: np.ndarray
) -> dict:
    """Metrics split by named market regime (spec 3.3)."""
    regimes = regime_for(dates).to_numpy()
    report = {}
    for name in pd.unique(regimes):
        mask = regimes == name
        if mask.sum() < MIN_REGIME_ROWS:
            continue
        if len(np.unique(y_true[mask])) < 2:
            continue
        report[str(name)] = evaluate(y_true[mask], proba[mask], priors)
    return report


@dataclass
class WalkForwardResult:
    """Out-of-fold output of one purged walk-forward run."""

    folds: list[dict]
    best_iterations: list[int]
    y: np.ndarray
    proba: np.ndarray
    dates: pd.Series
    #: One baseline class distribution per row, from that row's training fold.
    priors: np.ndarray


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    spec: TargetSpec,
    cfg: PanelTrainConfig,
    stride: int = 1,
    params_override: dict | None = None,
) -> WalkForwardResult:
    """Purged, embargoed walk-forward evaluation of one target.

    Shared by training and by the baseline ablations, so a baseline is scored
    on exactly the folds, rows and early-stopping scheme the real model was.
    """
    # `split_panel` counts *retained* dates, so every window expressed in trading
    # days has to be rescaled by the stride. Skipping this would silently turn a
    # one-year test fold into a `stride`-year one and shrink the purge gap below
    # the label horizon -- reintroducing exactly the overlap the purge exists to
    # remove, while the fold boundaries still looked correct.
    splitter = PurgedWalkForward(
        n_splits=cfg.validation.n_splits,
        test_size=max(1, round(cfg.validation.test_days / stride)),
        purge=max(1, math.ceil(spec.horizon_days / stride)),
        embargo=max(1, math.ceil(cfg.validation.embargo_days / stride)),
    )

    folds, best_iterations = [], []
    oos_true, oos_proba, oos_dates, oos_priors = [], [], [], []

    for i, (train_idx, test_idx) in enumerate(splitter.split_panel(dates)):
        booster = _fit_fold(X, y, dates, train_idx, spec, cfg, stride, params_override)
        proba = _as_proba(
            booster.predict(X.iloc[test_idx], num_iteration=booster.best_iteration or None),
            spec.n_classes,
        )

        y_test = y.iloc[test_idx].to_numpy()
        priors = _priors(y.iloc[train_idx].to_numpy(), spec.n_classes)

        metrics = evaluate(y_test, proba, priors)
        metrics.update(
            {
                "fold": i,
                "train_start": str(dates.iloc[train_idx[0]].date()),
                "train_end": str(dates.iloc[train_idx[-1]].date()),
                "test_start": str(dates.iloc[test_idx[0]].date()),
                "test_end": str(dates.iloc[test_idx[-1]].date()),
                "n_train": int(len(train_idx)),
                "best_iteration": int(booster.best_iteration or booster.num_trees()),
            }
        )
        folds.append(metrics)
        best_iterations.append(metrics["best_iteration"])
        oos_true.append(y_test)
        oos_proba.append(proba)
        oos_dates.append(dates.iloc[test_idx])
        oos_priors.append(np.tile(priors, (len(y_test), 1)))

        log.info(
            "  %s fold %d (%s..%s) brier_skill=%+.4f",
            spec.name, i, metrics["test_start"], metrics["test_end"],
            metrics["brier_skill"] if metrics["brier_skill"] is not None else float("nan"),
        )

    return WalkForwardResult(
        folds=folds,
        best_iterations=best_iterations,
        y=np.concatenate(oos_true),
        proba=np.vstack(oos_proba),
        dates=pd.concat(oos_dates),
        priors=np.vstack(oos_priors),
    )


def train_target(
    panel: FeaturePanel,
    spec: TargetSpec,
    cfg: PanelTrainConfig,
    *,
    final_init_model: str | None = None,
) -> dict:
    """Walk-forward evaluate then fit a final model for one target."""
    X, y, dates, tickers = panel.target_frame(spec)
    X, y, dates, tickers, stride = _thin_by_date(X, y, dates, tickers, cfg.max_train_rows)
    log.info(
        "%s: %s rows, %s tickers%s",
        spec.name,
        f"{len(X):,}",
        tickers.nunique(),
        f" (kept every {stride} dates)" if stride > 1 else "",
    )

    wf = walk_forward(X, y, dates, spec, cfg, stride)
    folds, best_iterations = wf.folds, wf.best_iterations
    all_true, all_proba, all_dates = wf.y, wf.proba, wf.dates
    overall_priors = _priors(y.to_numpy(), spec.n_classes)

    pooled = evaluate(all_true, all_proba, wf.priors)
    pooled["calibration"] = {
        spec.class_names[c]: calibration_bins(all_true, all_proba, c)
        for c in range(spec.n_classes)
    }

    median_rounds = int(np.median(best_iterations))
    final = lgb.train(
        cfg.lgb_params(spec),
        lgb.Dataset(X, label=y),
        num_boost_round=max(median_rounds, 50),
        init_model=final_init_model,
    )

    out_dir = MODEL_DIR / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out_dir / "model.txt"))
    # The optional fusion model must be fitted on these out-of-fold GBM
    # probabilities, never on predictions from the final model that was trained
    # on the same labels.  Persisting this small validation artifact makes that
    # provenance enforceable instead of relying on a caller convention.
    np.savez_compressed(
        out_dir / "oos_predictions.npz",
        y=all_true,
        probabilities=all_proba,
        dates=all_dates.to_numpy(dtype="datetime64[ns]"),
    )

    metadata = {
        "target": spec.name,
        "spec": asdict(spec),
        "schema": panel.schema,
        "feature_names": panel.feature_names,
        "class_names": {str(k): v for k, v in spec.class_names.items()},
        "class_priors": overall_priors.tolist(),
        "train_rows": int(len(X)),
        "date_stride": int(stride),
        "train_tickers": int(tickers.nunique()),
        "train_start": str(dates.iloc[0].date()),
        "train_end": str(dates.iloc[-1].date()),
        "fit_mode": "incremental" if final_init_model else "full",
        "validation": {
            "folds": folds,
            "pooled_out_of_sample": pooled,
            "by_regime": _regime_report(all_true, all_proba, all_dates, overall_priors),
            "median_best_iteration": median_rounds,
        },
        "data_caveats": [
            "Universe is today's S&P 500 constituents: survivorship-biased. "
            "Companies that failed or were removed are absent, so downside "
            "frequencies are a floor, not an estimate.",
            "Fundamentals are SEC XBRL, joined as of filing date, not Compustat.",
            "Guidance, litigation, dividend and rating events are absent from the "
            "event taxonomy -- 8-K item codes cannot express them.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    _log_to_mlflow(spec, cfg, metadata)
    return metadata


def train_all(
    targets: list[TargetSpec] | None = None,
    cfg: PanelTrainConfig | None = None,
    panel: FeaturePanel | None = None,
    *,
    incremental: bool = False,
) -> dict[str, dict]:
    targets = targets or default_targets()
    cfg = cfg or PanelTrainConfig()
    panel = panel or load_panel()

    initial_models: dict[str, str] = {}
    if incremental:
        # Validate every target before mutating any artifact.  Let the caller
        # fall back to a coherent full run rather than producing a half-updated
        # set of six models and reporting the job as incremental.
        for spec in targets:
            previous_dir = MODEL_DIR / spec.name
            previous_model = previous_dir / "model.txt"
            previous_meta = previous_dir / "metadata.json"
            if not previous_model.exists() or not previous_meta.exists():
                raise RuntimeError(f"cannot incrementally train {spec.name}: no existing model")
            previous_schema = json.loads(previous_meta.read_text()).get("schema")
            if previous_schema != panel.schema:
                raise RuntimeError(
                    f"cannot incrementally train {spec.name}: feature schema changed "
                    f"({previous_schema} -> {panel.schema}); run a full retrain"
                )
            initial_models[spec.name] = str(previous_model)

    results = {}
    for spec in targets:
        try:
            results[spec.name] = train_target(
                panel, spec, cfg, final_init_model=initial_models.get(spec.name)
            )
        except Exception as exc:  # noqa: BLE001 - one target must not sink the rest
            log.exception("training failed for %s: %s", spec.name, exc)
        finally:
            # LightGBM's binned dataset and the fold prediction arrays are both
            # large, and Python will happily hold the freed blocks until the
            # next allocation pressure. On 8 GB that is late enough to get the
            # process OOM-killed partway through the six targets, so collect
            # explicitly between them.
            gc.collect()
    (MODEL_DIR / "index.json").write_text(
        json.dumps(
            {
                name: {
                    "brier_skill": meta["validation"]["pooled_out_of_sample"]["brier_skill"],
                    "macro_auc": meta["validation"]["pooled_out_of_sample"]["macro_auc"],
                }
                for name, meta in results.items()
            },
            indent=2,
        )
    )
    return results
