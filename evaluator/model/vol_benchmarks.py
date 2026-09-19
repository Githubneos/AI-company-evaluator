"""Standard volatility models as benchmarks: HAR-RV and GARCH(1,1).

The baseline study showed these GBMs are mostly volatility forecasters. The
honest comparison for a volatility forecaster is the econometrics literature's
workhorses, not class priors. Both benchmarks here forecast the variance of
the next h days, relative to the trailing variance the label is scaled by:

    rho_hat = forecast mean daily variance over (t, t+h]  /  vol_60(t)^2

That ratio maps directly onto the label, which asks whether the forward
h-day return exceeds ``threshold * vol_60(t) * sqrt(h)``. A per-fold
multinomial logistic calibration on training rows turns ``log(rho_hat)`` into
class probabilities (magnitude or direction) comparable with the models'.

- **HAR-RV** (Corsi 2009): pooled OLS per fold of the future variance ratio on
  daily, weekly (5d) and monthly (22d) realised variance, each divided by the
  same trailing variance so the regression is scale-free across tickers.
- **HAR-RV (range)**: the same regression on the Parkinson range-based
  variance, (ln H/L)^2 / (4 ln 2). A diagnostic showed the intraday range is
  where these models' signal lives (ATR relative to trailing volatility alone
  reaches most of the production model's skill), so close-to-close HAR and
  GARCH are an unfairly weak comparison on their own. This is the strongest
  benchmark here.
- **GARCH(1,1)**: Gaussian maximum likelihood per ticker per fold, on returns
  dated on or before that fold's last training date. The variance recursion is
  linear in r^2 given the parameters, so it runs through
  ``scipy.signal.lfilter`` rather than a Python loop.

Leakage: every quantity at date t uses returns up to and including t. Training
targets (future variance, labels) come only from training rows, which the
purged splitter already separates from each test block.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression

from evaluator.config import TargetSpec
from evaluator.data.sources import _cache_path
from evaluator.features.build import realized_vol
from evaluator.features.store import FeaturePanel
from evaluator.io import atomic_write_json, read_parquet_or_none
from evaluator.metrics import brier_score
from evaluator.model.baselines import (
    _check_alignment,
    _incremental,
    _load_model_artifacts,
    _skill,
    _thin_to_stride,
    paired_fold_stats,
)
from evaluator.model.train import MIN_REGIME_ROWS, MODEL_DIR, PanelTrainConfig, WalkForwardResult, make_splitter
from evaluator.regimes import regime_for

log = logging.getLogger(__name__)

PRICE_START = "2005-01-01"  # the lookback the panel's labels were built from
VOL_LOOKBACK = 60
MIN_GARCH_OBS = 500
EPS = 1e-12
#: Single panel features, each calibrated the same way, to show *where* the
#: skill comes from: close-to-close volatility measures against the intraday
#: range. Values are (label, numerator, denominator or None).
DIAGNOSTIC_FEATURES = {
    "vol_60 level": ("vol_60", None),
    "vol ratio 20/60 (close-to-close)": ("vol_ratio_20_60", None),
    "ATR / vol_60 (intraday range)": ("atr_14_pct", "vol_60"),
}

BENCHMARKS = ("har", "har_range", "garch")
LABELS = {"har": "HAR-RV", "har_range": "HAR-RV (range)", "garch": "GARCH(1,1)"}
PARKINSON = 1.0 / (4.0 * np.log(2.0))


# --------------------------------------------------------------------- GARCH


def garch_filter(returns: np.ndarray, omega: float, alpha: float, beta: float, init: float) -> np.ndarray:
    """One-step-ahead conditional variances.

    ``out[t]`` is the variance for day t+1 forecast at the close of day t, so
    ``out[t]`` depends on ``returns[: t + 1]`` only. The variance used for
    ``returns[t]`` itself is ``out[t - 1]`` (``init`` for t = 0).
    """
    r2 = np.square(returns, dtype=float)
    # sigma2[t+1] = omega + alpha * r2[t] + beta * sigma2[t],  sigma2[0] = init
    out, _ = lfilter([1.0], [1.0, -beta], omega + alpha * r2, zi=[beta * init])
    return out


def _neg_loglik(params: np.ndarray, returns: np.ndarray, init: float) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    ahead = garch_filter(returns, omega, alpha, beta, init)
    sigma2 = np.concatenate([[init], ahead[:-1]])
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
        return 1e12
    return float(0.5 * np.sum(np.log(sigma2) + np.square(returns) / sigma2))


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    init: float
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run(self) -> float:
        return self.omega / max(1e-9, 1.0 - self.persistence)


def fit_garch(returns: np.ndarray) -> GarchFit:
    """Gaussian MLE of GARCH(1,1) on demeaned returns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    r = r - r.mean()
    var = float(np.var(r))
    best = None
    # A few starting points: the likelihood surface is flat along omega/(1-a-b).
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95)):
        x0 = np.array([var * (1 - a0 - b0), a0, b0])
        res = minimize(
            _neg_loglik, x0, args=(r, var), method="L-BFGS-B",
            bounds=[(var * 1e-6, var * 10), (0.0, 0.5), (0.0, 0.998)],
        )
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = (float(v) for v in best.x)
    return GarchFit(omega, alpha, beta, var, bool(best.success) and best.fun < 1e11)


def garch_h_day_variance(ahead: np.ndarray, fit: GarchFit, h: int) -> np.ndarray:
    """Mean daily variance over the next h days, from one-step-ahead forecasts.

    E[sigma2_{t+k}] = L + p^(k-1) (sigma2_{t+1} - L), L = long-run variance.
    """
    p, lr = fit.persistence, fit.long_run
    weights = p ** np.arange(h)  # k = 1..h  ->  p^0 .. p^(h-1)
    return lr + (ahead - lr) * weights.mean()


# ----------------------------------------------------------------------- HAR


def har_components(daily_variance: pd.Series, prefix: str = "rv") -> pd.DataFrame:
    """Daily / weekly / monthly means of a daily variance proxy at t (uses data <= t).

    Pass returns squared for classic HAR-RV, or a range estimator for HAR-range.
    """
    return pd.DataFrame(
        {
            f"{prefix}_d": daily_variance,
            f"{prefix}_w": daily_variance.rolling(5, min_periods=5).mean(),
            f"{prefix}_m": daily_variance.rolling(22, min_periods=22).mean(),
        }
    )


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Range-based daily variance, (ln H/L)^2 / (4 ln 2). Uses only day t's bar."""
    return np.log(high / low).pow(2) * PARKINSON


def future_mean_variance(returns: pd.Series, h: int) -> pd.Series:
    """Mean of r^2 over (t, t+h]: a *training target only* -- it looks forward."""
    r2 = returns.pow(2)
    return r2.rolling(h, min_periods=h).mean().shift(-h)


def fit_har(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS with intercept; returns [b0, b_d, b_w, b_m]."""
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


# ---------------------------------------------------------- per-row features


def _series_for(ticker: str, prices_by_ticker: dict | None) -> pd.DataFrame | None:
    if prices_by_ticker is not None:
        prices = prices_by_ticker.get(ticker)
    else:
        # Read the cache directly: no network, and the same bars the labels used.
        prices = read_parquet_or_none(_cache_path(ticker, PRICE_START, None))
    if prices is None or prices.empty:
        return None
    index = pd.DatetimeIndex(prices.index).normalize()
    close = pd.Series(prices["close"].to_numpy(dtype=float), index=index)
    returns = close.pct_change()
    frame = har_components(returns.pow(2), "rv")
    if {"high", "low"} <= set(prices.columns):
        high = pd.Series(prices["high"].to_numpy(dtype=float), index=index)
        low = pd.Series(prices["low"].to_numpy(dtype=float), index=index)
        with np.errstate(divide="ignore", invalid="ignore"):
            park = parkinson_variance(high, low).where((high > 0) & (low > 0) & (high >= low))
        frame = frame.join(har_components(park, "pk"))
    else:
        for col in ("pk_d", "pk_w", "pk_m"):
            frame[col] = np.nan
    frame["ret"] = returns
    frame["trail_var"] = realized_vol(returns, VOL_LOOKBACK).pow(2)
    return frame


@dataclass
class Rows:
    """The thinned panel rows for one target, with walk-forward folds."""

    tickers: np.ndarray
    dates: pd.Series
    y: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]


def replay_rows(panel: FeaturePanel, spec: TargetSpec, cfg: PanelTrainConfig, stride: int) -> Rows:
    """The model's rows and folds, without fitting anything."""
    narrow = FeaturePanel(panel.frame, panel.feature_names[:1], panel.label_names, panel.schema)
    X, y, dates, tickers = narrow.target_frame(spec)
    keep = X.index
    if stride > 1:
        _, _, kept_dates = _thin_to_stride(X, y, dates, stride)
        keep = kept_dates.index
    y, dates, tickers = y.loc[keep], dates.loc[keep], tickers.loc[keep]
    folds = list(make_splitter(spec, cfg, stride).split_panel(dates))
    return Rows(tickers.to_numpy(), dates.reset_index(drop=True), y.to_numpy(), folds)


# ------------------------------------------------------------------ evaluate


def _calibrated(x_train, y_train, x_test, n_classes: int, priors: np.ndarray) -> np.ndarray:
    """Multinomial logistic on one feature; rows without a forecast get the priors."""
    out = np.tile(priors, (len(x_test), 1))
    ok_train, ok_test = np.isfinite(x_train), np.isfinite(x_test)
    if ok_train.sum() < 100 or len(np.unique(y_train[ok_train])) < n_classes:
        return out
    model = LogisticRegression(C=1e4, max_iter=1000)
    model.fit(x_train[ok_train, None], y_train[ok_train])
    proba = np.zeros((ok_test.sum(), n_classes))
    proba[:, model.classes_.astype(int)] = model.predict_proba(x_test[ok_test, None])
    out[ok_test] = proba
    return out


def evaluate_vol_benchmarks(
    panel: FeaturePanel,
    spec: TargetSpec,
    cfg: PanelTrainConfig,
    *,
    prices_by_ticker: dict | None = None,
) -> dict:
    """Score HAR-RV and GARCH on the trained model's out-of-fold rows. Writes vol_benchmarks.json."""
    meta, saved = _load_model_artifacts(spec)
    stride = int(meta.get("date_stride", 1))
    rows = replay_rows(panel, spec, cfg, stride)
    h, k = spec.horizon_days, spec.n_classes

    # Per-row inputs, looked up from each ticker's daily series.
    n = len(rows.y)
    comp = np.full((n, 3), np.nan)
    comp_range = np.full((n, 3), np.nan)
    trail = np.full(n, np.nan)
    fut = np.full(n, np.nan)
    series: dict[str, pd.DataFrame] = {}
    positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker in pd.unique(rows.tickers):
        idx = np.flatnonzero(rows.tickers == ticker)
        s = _series_for(str(ticker), prices_by_ticker)
        if s is None:
            continue
        loc = s.index.get_indexer(pd.DatetimeIndex(rows.dates.iloc[idx]))
        good = loc >= 0
        idx, loc = idx[good], loc[good]
        series[str(ticker)] = s
        positions[str(ticker)] = (idx, loc)
        comp[idx] = s[["rv_d", "rv_w", "rv_m"]].to_numpy()[loc]
        comp_range[idx] = s[["pk_d", "pk_w", "pk_m"]].to_numpy()[loc]
        trail[idx] = s["trail_var"].to_numpy()[loc]
        fut[idx] = future_mean_variance(s["ret"], h).to_numpy()[loc]
    covered = np.isfinite(trail) & (trail > 0)

    # Scale-free HAR inputs. Rows without a trailing variance yet (a ticker's
    # first weeks) are NaN by design and excluded below.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_y = fut / trail
        har_inputs = {"har": comp / trail[:, None], "har_range": comp_range / trail[:, None]}

    probas = {name: np.zeros((n, k)) for name in BENCHMARKS}
    oos = np.concatenate([test for _, test in rows.folds])
    fold_meta = []
    for i, (train_idx, test_idx) in enumerate(rows.folds):
        priors = np.bincount(rows.y[train_idx].astype(int), minlength=k) / len(train_idx)
        y_train = rows.y[train_idx].astype(int)

        # HAR (close-to-close and range): scale-free pooled OLS on training
        # rows, clipped at the 99.5th percentile against earnings-day outliers.
        coefs = {}
        for name, ratio_x in har_inputs.items():
            usable = covered & np.all(np.isfinite(ratio_x), axis=1)
            tr = train_idx[usable[train_idx] & np.isfinite(ratio_y[train_idx])]
            if len(tr) < 100:
                continue  # e.g. no high/low data: leave this benchmark at the priors
            hi_x = np.quantile(ratio_x[tr], 0.995, axis=0)
            hi_y = np.quantile(ratio_y[tr], 0.995)
            coef = fit_har(np.minimum(ratio_x[tr], hi_x), np.minimum(ratio_y[tr], hi_y))
            with np.errstate(invalid="ignore"):
                rho = coef[0] + np.minimum(ratio_x, hi_x) @ coef[1:]
                x_har = np.log(np.clip(rho, 1e-4, None))
            x_har[~usable] = np.nan
            probas[name][test_idx] = _calibrated(x_har[train_idx], y_train, x_har[test_idx], k, priors)
            coefs[name] = [float(c) for c in coef]
        for name in har_inputs:
            if name not in coefs:
                probas[name][test_idx] = np.tile(priors, (len(test_idx), 1))

        # GARCH: per-ticker MLE on returns up to this fold's last training date.
        train_end = rows.dates.iloc[train_idx].max()
        x_garch = np.full(n, np.nan)
        persist = []
        for ticker, (idx, loc) in positions.items():
            s = series[ticker]
            ret = s["ret"].to_numpy()
            valid = np.isfinite(ret)
            train_mask = valid & (s.index <= train_end)
            if train_mask.sum() < MIN_GARCH_OBS:
                continue
            fit = fit_garch(ret[train_mask])
            if not fit.converged:
                continue
            persist.append(fit.persistence)
            mean = float(np.mean(ret[train_mask]))
            filled = np.where(valid, ret - mean, 0.0)
            ahead = garch_filter(filled, fit.omega, fit.alpha, fit.beta, fit.init)
            var_h = garch_h_day_variance(ahead, fit, h)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_garch[idx] = np.log(np.clip(var_h[loc] / trail[idx], 1e-4, None))
        x_garch[~covered] = np.nan
        probas["garch"][test_idx] = _calibrated(x_garch[train_idx], y_train, x_garch[test_idx], k, priors)
        fold_meta.append(
            {
                "fold": i,
                "har_coef": coefs.get("har"),
                "har_range_coef": coefs.get("har_range"),
                "garch_fits": len(persist),
                "garch_median_persistence": float(np.median(persist)) if persist else None,
            }
        )
        log.info("%s fold %d: HAR %s, HAR-range %s, GARCH fits %d", spec.name, i,
                 np.round(coefs.get("har", []), 3).tolist(), np.round(coefs.get("har_range", []), 3).tolist(),
                 len(persist))

    # Assemble in walk-forward order and prove it is the model's own rows.
    wf = WalkForwardResult(
        folds=[
            {
                "fold": i,
                "n": int(len(test)),
                "test_start": str(rows.dates.iloc[test[0]].date()),
                "test_end": str(rows.dates.iloc[test[-1]].date()),
            }
            for i, (_, test) in enumerate(rows.folds)
        ],
        best_iterations=[],
        y=rows.y[oos].astype(int),
        proba=np.zeros((len(oos), k)),
        dates=rows.dates.iloc[oos].reset_index(drop=True),
        priors=np.vstack(
            [np.tile(np.bincount(rows.y[tr].astype(int), minlength=k) / len(tr), (len(te), 1)) for tr, te in rows.folds]
        ),
    )
    _check_alignment(spec, wf, saved, meta)

    y, priors = wf.y, wf.priors
    model_proba = np.asarray(saved["probabilities"], dtype=float)
    bench = {name: probas[name][oos] for name in BENCHMARKS}
    fold_n = np.array([f["n"] for f in wf.folds])
    bounds = np.concatenate([[0], np.cumsum(fold_n)])
    spans = list(zip(bounds[:-1], bounds[1:]))

    def fold_briers(proba):
        return np.array([brier_score(y[a:b], proba[a:b]) for a, b in spans])

    model_fb = fold_briers(model_proba)
    regimes = regime_for(wf.dates).to_numpy()
    report_bench = {}
    for name, proba in bench.items():
        by_regime = {}
        for regime in sorted(set(regimes)):
            mask = regimes == regime
            if mask.sum() < MIN_REGIME_ROWS or len(np.unique(y[mask])) < 2:
                continue
            by_regime[str(regime)] = _incremental(
                brier_score(y[mask], model_proba[mask]), brier_score(y[mask], proba[mask])
            )
        report_bench[name] = {
            "label": LABELS[name],
            "pooled": _skill(y, proba, priors),
            "model_edge": paired_fold_stats(fold_n, model_fb, fold_briers(proba)),
            "model_edge_by_regime": by_regime,
        }

    diagnostics = _single_feature_diagnostics(panel, rows, oos, priors, k)

    report = {
        "target": spec.name,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_schema": meta.get("schema"),
        "n_rows": int(len(y)),
        "coverage": float(np.mean(covered[oos])),
        "range_coverage": float(np.mean(np.all(np.isfinite(har_inputs["har_range"][oos]), axis=1))),
        "model": {"pooled": _skill(y, model_proba, priors)},
        "benchmarks": report_bench,
        "single_feature_diagnostics": diagnostics,
        "folds": fold_meta,
    }
    atomic_write_json(report, MODEL_DIR / spec.name / "vol_benchmarks.json")
    return report


def _single_feature_diagnostics(panel: FeaturePanel, rows: Rows, oos, priors, k: int) -> dict:
    """Skill of one panel feature at a time, through the same folds and calibration."""
    columns = sorted({c for pair in DIAGNOSTIC_FEATURES.values() for c in pair if c})
    missing = [c for c in columns if c not in panel.feature_names]
    if missing:
        return {}
    frame = panel.frame.set_index(["ticker", "date"])[columns]
    key = pd.MultiIndex.from_arrays([rows.tickers, pd.DatetimeIndex(rows.dates)])
    values = frame.reindex(key)
    y = rows.y.astype(int)
    out = {}
    for label, (num, den) in DIAGNOSTIC_FEATURES.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = values[num].to_numpy(dtype=float)
            if den:
                ratio = ratio / values[den].to_numpy(dtype=float)
            x = np.log(np.where(ratio > 0, ratio, np.nan))
        proba = np.zeros((len(y), k))
        for train_idx, test_idx in rows.folds:
            fold_priors = np.bincount(y[train_idx], minlength=k) / len(train_idx)
            proba[test_idx] = _calibrated(x[train_idx], y[train_idx], x[test_idx], k, fold_priors)
        out[label] = _skill(y[oos], proba[oos], priors)["brier_skill"]
    return out


def load_vol_benchmarks(target: str) -> dict | None:
    path = MODEL_DIR / target / "vol_benchmarks.json"
    return json.loads(path.read_text()) if path.exists() else None


__all__ = [
    "evaluate_vol_benchmarks",
    "fit_garch",
    "fit_har",
    "future_mean_variance",
    "garch_filter",
    "garch_h_day_variance",
    "har_components",
    "parkinson_variance",
    "load_vol_benchmarks",
    "replay_rows",
]
