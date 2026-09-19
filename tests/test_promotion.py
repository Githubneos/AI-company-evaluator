"""The promotion gate: what may replace a model that is serving traffic."""

import json

import numpy as np
import pandas as pd
import pytest

import evaluator.model.predict as predict
import evaluator.model.promotion as promotion
import evaluator.model.registry as registry


def _metadata(skill: float, regimes: dict | None = None) -> dict:
    return {
        "target": "magnitude_1d",
        "validation": {
            "pooled_out_of_sample": {"brier_skill": skill},
            "by_regime": {k: {"brier_skill": v} for k, v in (regimes or {}).items()},
        },
    }


def _write_model(stage: str, target: str, metadata: dict, *, oos: dict | None = None) -> None:
    d = registry.model_dir(target, stage)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.txt").write_text("booster")
    (d / "metadata.json").write_text(json.dumps(metadata))
    if oos is not None:
        np.savez_compressed(d / "oos_predictions.npz", **oos)


def _oos(tickers, dates, y, p_large) -> dict:
    return {
        "y": np.asarray(y),
        "probabilities": np.column_stack([1 - np.asarray(p_large), np.asarray(p_large)]),
        "dates": pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]"),
        "tickers": np.asarray(tickers, dtype="U12"),
    }


@pytest.fixture
def stages(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "CANDIDATE_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "PRODUCTION_DIR", tmp_path / "production")
    predict.load_model.cache_clear()
    return tmp_path


def test_promotion_invalidates_the_serving_cache(stages, monkeypatch):
    cleared = []
    monkeypatch.setattr(predict.load_model, "cache_clear", lambda: cleared.append(True))
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.04))

    promotion.swap_into_production("magnitude_1d")

    assert cleared == [True]


def test_serving_reads_production_never_candidates(stages):
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.04))

    assert predict.available_targets() == []
    with pytest.raises(predict.ModelNotTrained, match="scripts.promote"):
        predict.load_model("magnitude_1d")

    promotion.promote("magnitude_1d", apply=True)
    assert predict.available_targets() == ["magnitude_1d"]


def test_a_candidate_without_skill_is_never_promoted(stages):
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(-0.01))

    verdict = promotion.promote("magnitude_1d", apply=True)

    assert not verdict["promote"] and "no measured skill" in verdict["reason"]
    assert not registry.model_dir("magnitude_1d", registry.PRODUCTION).exists()


def test_promotion_copies_the_whole_directory_including_reports(stages):
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.04))
    d = registry.model_dir("magnitude_1d", registry.CANDIDATE)
    (d / "baselines.json").write_text('{"verdict": "v"}')
    (d / "ablations.json").write_text('{"sets": {}}')

    assert promotion.promote("magnitude_1d", apply=True)["applied"]

    assert registry.load_report("magnitude_1d", "baselines") == {"verdict": "v"}
    assert registry.load_report("magnitude_1d", "ablations") == {"sets": {}}


def test_a_failed_swap_leaves_production_intact(stages, monkeypatch):
    _write_model(registry.PRODUCTION, "magnitude_1d", _metadata(0.04))
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.05))
    monkeypatch.setattr(promotion.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        promotion.swap_into_production("magnitude_1d")

    assert registry.load_metadata("magnitude_1d")["validation"]["pooled_out_of_sample"]["brier_skill"] == 0.04
    assert [p.name for p in registry.PRODUCTION_DIR.iterdir()] == ["magnitude_1d"]  # no leftovers


def test_regime_regression_refuses_an_otherwise_better_candidate(stages):
    incumbent = _metadata(0.03, {"recent": 0.02, "rate_hikes": 0.01})
    candidate = _metadata(0.04, {"recent": 0.05, "rate_hikes": -0.02})

    verdict = promotion.compare(candidate, incumbent)

    assert not verdict["promote"]
    assert "rate_hikes" in verdict["reason"] and "regresses" in verdict["reason"]


def test_shared_rows_decide_when_both_sides_have_them(stages):
    dates = pd.bdate_range("2024-01-01", periods=60).repeat(20)
    tickers = [f"T{i}" for i in range(20)] * 60
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, len(dates))
    # The incumbent is informative; the candidate is noise on those same rows.
    _write_model(registry.PRODUCTION, "magnitude_1d", _metadata(0.03),
                 oos=_oos(tickers, dates, y, np.where(y == 1, 0.8, 0.2)))
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.09),  # flattering pooled skill
                 oos=_oos(tickers, dates, y, rng.uniform(0.3, 0.7, len(dates))))

    verdict = promotion.promote("magnitude_1d", apply=True)

    assert not verdict["promote"]
    assert "rows both models scored" in verdict["reason"]
    assert verdict["common_rows"]["n"] == 1200 and verdict["common_rows"]["edge"] < 0
    # Pooled skill alone would have promoted it; production is untouched.
    assert registry.load_metadata("magnitude_1d")["validation"]["pooled_out_of_sample"]["brier_skill"] == 0.03


def test_a_genuinely_better_candidate_is_promoted_on_shared_rows(stages):
    dates = pd.bdate_range("2024-01-01", periods=60).repeat(20)
    tickers = [f"T{i}" for i in range(20)] * 60
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, len(dates))
    _write_model(registry.PRODUCTION, "magnitude_1d", _metadata(0.03),
                 oos=_oos(tickers, dates, y, np.where(y == 1, 0.6, 0.4)))
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.05),
                 oos=_oos(tickers, dates, y, np.where(y == 1, 0.85, 0.15)))

    verdict = promotion.promote("magnitude_1d", apply=True)

    assert verdict["promote"] and verdict["applied"]
    assert verdict["common_rows"]["edge"] > 0 and verdict["common_rows"]["ci90"][0] > 0
    assert registry.load_metadata("magnitude_1d")["validation"]["pooled_out_of_sample"]["brier_skill"] == 0.05


def test_partly_overlapping_universes_compare_only_on_the_overlap(stages):
    dates = pd.bdate_range("2024-01-01", periods=60)
    old_t = [f"T{i}" for i in range(20)]  # 20 x 60 = 1,200 shared rows, over the gate's minimum
    new_t = old_t + [f"N{i}" for i in range(10)]  # candidate saw more names
    rng = np.random.default_rng(2)

    def rows(tickers):
        d = pd.DatetimeIndex(np.repeat(dates.to_numpy(), len(tickers)))
        t = tickers * len(dates)
        y = rng.integers(0, 2, len(d))
        return t, d, y

    t_old, d_old, y_old = rows(old_t)
    _write_model(registry.PRODUCTION, "magnitude_1d", _metadata(0.03),
                 oos=_oos(t_old, d_old, y_old, np.where(y_old == 1, 0.7, 0.3)))
    shared = pd.Series(np.where(y_old == 1, 0.9, 0.1), index=pd.MultiIndex.from_arrays([t_old, d_old]))
    t_new, d_new, _ = rows(new_t)
    key = pd.MultiIndex.from_arrays([t_new, d_new])
    p_new = shared.reindex(key).to_numpy()
    y_new = np.where(np.isnan(p_new), rng.integers(0, 2, len(key)), (p_new > 0.5).astype(int))
    p_new = np.where(np.isnan(p_new), 0.5, p_new)
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.04),
                 oos=_oos(t_new, d_new, y_new, p_new))

    verdict = promotion.promote("magnitude_1d")

    assert verdict["common_rows"]["n"] == len(old_t) * len(dates)  # only the overlap
    assert verdict["promote"] and verdict["common_rows"]["edge"] > 0


def test_missing_tickers_fall_back_to_pooled_comparison(stages):
    _write_model(registry.PRODUCTION, "magnitude_1d", _metadata(0.05))  # no npz at all
    _write_model(registry.CANDIDATE, "magnitude_1d", _metadata(0.04))

    verdict = promotion.promote("magnitude_1d")

    assert not verdict["promote"] and "does not beat incumbent" in verdict["reason"]
    assert verdict["common_rows"]["available"] is False
