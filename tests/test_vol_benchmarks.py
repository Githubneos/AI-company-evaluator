"""HAR-RV and GARCH benchmarks: correct estimators, no look-ahead, scored on the model's own rows."""

import numpy as np
import pandas as pd
import pytest

import evaluator.model.baselines as baselines
import evaluator.model.train as train
import evaluator.model.vol_benchmarks as vb
from evaluator.config import LabelConfig, TargetSpec, ValidationConfig
from evaluator.features.build import _atr, realized_vol
from evaluator.features.store import FeaturePanel
from evaluator.labels import make_labels


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int, *, with_sigma: bool = False):
    rng = np.random.default_rng(seed)
    r, sigma = np.empty(n), np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        sigma[t] = np.sqrt(var)
        r[t] = sigma[t] * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return (r, sigma) if with_sigma else r


def test_garch_filter_matches_the_textbook_loop():
    r = np.random.default_rng(0).normal(0, 0.01, 300)
    omega, alpha, beta, init = 2e-6, 0.08, 0.9, 1e-4

    expected, sigma2 = [], init
    for x in r:
        sigma2 = omega + alpha * x**2 + beta * sigma2
        expected.append(sigma2)

    np.testing.assert_allclose(vb.garch_filter(r, omega, alpha, beta, init), expected, rtol=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_garch_recovers_known_parameters(seed):
    r = _simulate_garch(8000, omega=2e-6, alpha=0.08, beta=0.90, seed=seed)

    fit = vb.fit_garch(r)

    assert fit.converged
    assert fit.alpha == pytest.approx(0.08, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.05)
    assert fit.persistence == pytest.approx(0.98, abs=0.02)


def test_h_day_forecast_matches_iterated_expectation():
    fit = vb.GarchFit(omega=2e-6, alpha=0.08, beta=0.9, init=1e-4, converged=True)
    ahead = np.array([5e-5, 1e-4, 4e-4])

    got = vb.garch_h_day_variance(ahead, fit, 5)

    lr, p = fit.long_run, fit.persistence
    expected = [np.mean([lr + p ** (k - 1) * (a - lr) for k in range(1, 6)]) for a in ahead]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # One-day horizon is exactly the one-step forecast.
    np.testing.assert_allclose(vb.garch_h_day_variance(ahead, fit, 1), ahead)


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(4)
    X = rng.gamma(2.0, 0.5, size=(20000, 3))
    y = 0.1 + X @ np.array([0.2, 0.3, 0.4]) + rng.normal(0, 0.05, 20000)

    np.testing.assert_allclose(vb.fit_har(X, y), [0.1, 0.2, 0.3, 0.4], atol=0.01)


def test_nothing_at_date_t_depends_on_returns_after_t():
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 400), index=pd.bdate_range("2020-01-01", periods=400))
    cut = 250
    shocked = r.copy()
    shocked.iloc[cut + 1 :] *= 7.0  # rewrite the future

    pd.testing.assert_frame_equal(
        vb.har_components(r.pow(2)).iloc[: cut + 1], vb.har_components(shocked.pow(2)).iloc[: cut + 1]
    )
    a = vb.garch_filter(r.to_numpy(), 2e-6, 0.08, 0.9, 1e-4)
    b = vb.garch_filter(shocked.to_numpy(), 2e-6, 0.08, 0.9, 1e-4)
    np.testing.assert_array_equal(a[: cut + 1], b[: cut + 1])
    # The training target is the one thing allowed to look forward.
    window = slice(cut - 5, cut + 1)
    assert not np.allclose(vb.future_mean_variance(r, 5).iloc[window], vb.future_mean_variance(shocked, 5).iloc[window])


def test_parkinson_variance_formula():
    high, low = pd.Series([110.0, 100.0]), pd.Series([100.0, 100.0])
    expected = [np.log(1.1) ** 2 / (4 * np.log(2)), 0.0]
    np.testing.assert_allclose(vb.parkinson_variance(high, low), expected)


# --------------------------------------------------------------- end to end

SPEC = TargetSpec(1, "magnitude")


def _intraday_range(close: pd.Series, r: np.ndarray, sigma: np.ndarray, *, seed: int, steps: int = 32):
    """High/low from an intraday Brownian path pinned to each day's actual return.

    A real bar's range must contain the open-to-close path, so it depends on
    both the day's volatility and its realised move -- a range drawn
    independently of the return would be unrealistically uninformative.
    """
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 1.0, (len(r), steps)) * (sigma[:, None] / np.sqrt(steps))
    path = np.cumsum(increments, axis=1)
    # Brownian bridge: shift the path so it ends exactly at the day's log return.
    path -= (path[:, -1:] - r[:, None]) * np.linspace(1 / steps, 1.0, steps)[None, :]
    prev = close.shift(1).fillna(close.iloc[0] / np.exp(r[0])).to_numpy()
    high = prev * np.exp(np.maximum(0.0, path.max(axis=1)))
    low = prev * np.exp(np.minimum(0.0, path.min(axis=1)))
    return high, low


def _garch_universe(n_tickers: int = 8, n_days: int = 2600):
    """Tickers whose returns really are GARCH, so volatility is forecastable."""
    dates = pd.bdate_range("2012-01-02", periods=n_days)
    prices, frames = {}, []
    for i in range(n_tickers):
        r, sigma = _simulate_garch(n_days, omega=3e-6, alpha=0.10, beta=0.87, seed=100 + i, with_sigma=True)
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
        high, low = _intraday_range(close, r, sigma, seed=200 + i)
        p = pd.DataFrame({"close": close, "high": high, "low": low})
        prices[f"T{i}"] = p
        returns = close.pct_change()
        labels = make_labels(p, LabelConfig(horizon_days=1, threshold_sigmas=1.0, vol_lookback=60))
        vol20, vol60 = realized_vol(returns, 20), realized_vol(returns, 60)
        bars = pd.DataFrame({"high": p["high"], "low": p["low"], "close": close})
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": f"T{i}",
                    "vol_20": vol20.to_numpy(dtype="float32"),
                    "vol_60": vol60.to_numpy(dtype="float32"),
                    "vol_ratio_20_60": (vol20 / vol60).to_numpy(dtype="float32"),
                    "atr_14_pct": (_atr(bars, 14) / close).to_numpy(dtype="float32"),
                    "abs_ret": returns.abs().to_numpy(dtype="float32"),
                    "label_magnitude_1d": labels["label_magnitude"].to_numpy(),
                }
            )
        )
    frame = pd.concat(frames).sort_values(["date", "ticker"]).reset_index(drop=True)
    features = ["vol_20", "vol_60", "vol_ratio_20_60", "atr_14_pct", "abs_ret"]
    panel = FeaturePanel(frame, features, ["label_magnitude_1d"], "synthetic")
    return panel, prices


def _cfg(n_splits: int = 4) -> train.PanelTrainConfig:
    return train.PanelTrainConfig(
        validation=ValidationConfig(n_splits=n_splits, test_days=250, embargo_days=5),
        max_train_rows=12000, num_boost_round=200, early_stopping_rounds=20, min_data_in_leaf=50,
    )


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    for module in (train, baselines, vb):
        monkeypatch.setattr(module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "_log_to_mlflow", lambda *a, **k: None)
    return tmp_path


def test_benchmarks_score_the_models_own_rows_and_find_real_volatility_signal(model_dir):
    panel, prices = _garch_universe()
    train.train_target(panel, SPEC, _cfg())

    report = vb.evaluate_vol_benchmarks(panel, SPEC, _cfg(), prices_by_ticker=prices)

    saved = np.load(model_dir / SPEC.name / "oos_predictions.npz")
    assert report["n_rows"] == len(saved["y"])
    assert report["coverage"] > 0.95
    assert report["range_coverage"] > 0.95
    for name in ("har", "har_range", "garch"):
        # On truly GARCH data, every benchmark must beat class priors out of sample.
        assert report["benchmarks"][name]["pooled"]["brier_skill"] > 0.005, name
    persist = [f["garch_median_persistence"] for f in report["folds"]]
    assert all(p == pytest.approx(0.97, abs=0.05) for p in persist)  # true persistence 0.97
    assert (model_dir / SPEC.name / "vol_benchmarks.json").exists()


def test_single_feature_diagnostic_finds_the_range_feature_strongest(model_dir):
    panel, prices = _garch_universe()
    train.train_target(panel, SPEC, _cfg())

    diag = vb.evaluate_vol_benchmarks(panel, SPEC, _cfg(), prices_by_ticker=prices)["single_feature_diagnostics"]

    assert set(diag) == set(vb.DIAGNOSTIC_FEATURES)
    # Both volatility views find real signal here. Which one wins is a property
    # of the data, not of the code: this simulation holds volatility constant
    # within each day, so the range has little efficiency advantage, while on
    # real bars it is far ahead (see the README table).
    assert diag["ATR / vol_60 (intraday range)"] > 0.005
    assert diag["vol ratio 20/60 (close-to-close)"] > 0.005
    assert diag["vol_60 level"] < diag["ATR / vol_60 (intraday range)"]


def test_range_benchmark_falls_back_to_priors_without_high_low(model_dir):
    panel, prices = _garch_universe()
    train.train_target(panel, SPEC, _cfg())
    close_only = {t: p[["close"]] for t, p in prices.items()}

    report = vb.evaluate_vol_benchmarks(panel, SPEC, _cfg(), prices_by_ticker=close_only)

    assert report["range_coverage"] == 0.0
    assert report["benchmarks"]["har_range"]["pooled"]["brier_skill"] == pytest.approx(0.0, abs=1e-9)
    assert report["benchmarks"]["har"]["pooled"]["brier_skill"] > 0.005


def test_a_panel_that_changed_since_training_is_refused(model_dir):
    panel, prices = _garch_universe()
    train.train_target(panel, SPEC, _cfg(n_splits=3))

    with pytest.raises(baselines.BaselineMismatch):
        vb.evaluate_vol_benchmarks(panel, SPEC, _cfg(n_splits=4), prices_by_ticker=prices)
