"""Is a model trustworthy in today's market, on its own measured record?"""

import pandas as pd
import pytest

import evaluator.fusion as fusion
from evaluator.regimes import ANALOGOUS_REGIMES, current_regimes, volatility_state


@pytest.mark.parametrize(
    ("vix", "state"),
    [(9.0, "calm"), (14.9, "calm"), (15.0, "normal"), (21.9, "normal"),
     (22.0, "stressed"), (29.9, "stressed"), (30.0, "crisis"), (80.0, "crisis"),
     (None, "unknown"), (float("nan"), "unknown")],
)
def test_volatility_state_bands(vix, state):
    assert volatility_state(vix) == state


def test_todays_regimes_combine_the_calendar_and_the_backdrop():
    calm = current_regimes(12.0, today=pd.Timestamp("2026-09-19"))
    stressed = current_regimes(26.0, today=pd.Timestamp("2026-09-19"))

    assert calm["calendar"] == "recent" and calm["state"] == "calm"
    assert calm["relevant"][0] == "recent"  # where we are in the split
    assert "rate_hikes" in stressed["relevant"]  # what this market resembles
    assert stressed["relevant"].count("recent") == 1  # no duplicates


def test_unknown_volatility_falls_back_to_the_calendar_regime_only():
    context = current_regimes(None, today=pd.Timestamp("2026-09-19"))

    assert context["state"] == "unknown"
    assert context["relevant"] == ["recent"]
    assert set(ANALOGOUS_REGIMES) == {"calm", "normal", "stressed", "crisis"}


def _metadata(by_regime: dict) -> dict:
    return {"validation": {"by_regime": {k: {"brier_skill": v} for k, v in by_regime.items()}}}


@pytest.fixture
def measured(monkeypatch):
    """Two models: one that held up in stressed markets, one that did not."""
    records = {
        "magnitude_1d": _metadata({"recent": 0.03, "rate_hikes": 0.02, "gfc": 0.01}),
        "direction_5d": _metadata({"recent": 0.02, "rate_hikes": -0.032, "gfc": 0.01}),
    }
    monkeypatch.setattr("evaluator.model.registry.load_metadata", lambda t, *a, **k: records.get(t))
    return records


def test_a_model_that_lost_in_similar_conditions_is_flagged(measured, monkeypatch):
    monkeypatch.setattr("evaluator.regimes.current_vix", lambda: 26.0)  # stressed, like 2022-23

    guard = fusion._regime_guard({"magnitude_1d": {}, "direction_5d": {}})

    assert guard["state"] == "stressed"
    assert guard["unreliable_now"] == ["direction_5d"]
    flagged = guard["per_target"]["direction_5d"]
    assert flagged["reliable_now"] is False
    assert flagged["worst_regime"] == "rate_hikes" and flagged["worst_skill"] == -0.032
    assert "no signal" in flagged["interpretation"]
    assert guard["per_target"]["magnitude_1d"]["reliable_now"] is True


def test_the_same_model_is_fine_when_the_market_is_calm(measured, monkeypatch):
    monkeypatch.setattr("evaluator.regimes.current_vix", lambda: 13.0)

    guard = fusion._regime_guard({"direction_5d": {}})

    # Its rate-hike failure is real but not relevant to a calm tape.
    assert guard["unreliable_now"] == []
    assert guard["per_target"]["direction_5d"]["worst_regime"] in ("recent", "recovery", "covid_recovery")


def test_a_model_without_a_record_for_todays_regimes_says_so(monkeypatch):
    monkeypatch.setattr("evaluator.model.registry.load_metadata", lambda t, *a, **k: _metadata({"gfc": 0.05}))
    monkeypatch.setattr("evaluator.regimes.current_vix", lambda: 13.0)

    guard = fusion._regime_guard({"magnitude_1d": {}})
    entry = guard["per_target"]["magnitude_1d"]

    assert entry["reliable_now"] is True  # unmeasured is not the same as failed
    assert entry["worst_skill"] is None
    assert "unmeasured" in entry["interpretation"]


def test_the_llm_is_told_to_defer_to_the_guard():
    from evaluator.llm.evaluator import SYSTEM_PROMPT

    assert "regime_guard.unreliable_now" in SYSTEM_PROMPT
    assert "unusable right now" in SYSTEM_PROMPT
