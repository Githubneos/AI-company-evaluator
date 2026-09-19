"""Post-mortem tagging tests (spec 6.2).

The design property under test: rules run before the LLM, and the LLM is only
consulted for what the rules cannot explain. That ordering is what keeps the
tags trustworthy -- a rule firing on a measured VIX jump is evidence, whereas a
model asked to explain a move it cannot observe will always produce a plausible
story. UNEXPLAINED staying a real, reachable outcome is part of the same point.
"""

import pandas as pd
import pytest

from evaluator.feedback import postmortem as pm
from evaluator.feedback.postmortem import TAGS, rule_tags, tag_predictions, tag_summary
from evaluator.feedback.store import log_prediction, resolve_pending
from evaluator.llm.provider import LLMResponse


@pytest.fixture
def db(tmp_path):
    return tmp_path / "predictions.db"


@pytest.fixture
def prices():
    dates = pd.bdate_range("2024-01-01", periods=60, name="date")
    return pd.DataFrame({"close": pd.Series(range(100, 160), index=dates, dtype=float)})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Neutralise the VIX and EDGAR lookups unless a test opts in."""
    monkeypatch.setattr(pm, "_vix_move", lambda *a, **k: None)
    monkeypatch.setattr(pm, "_untagged_events", lambda *a, **k: False)


def _row(**overrides) -> dict:
    row = {
        "ticker": "TEST",
        "as_of": "2024-03-01",
        "horizon_days": 5,
        "predicted_class": "NEUTRAL",
        "actual_class": "SPIKE",
        "actual_return": 0.09,
        "actual_z": 2.5,
        "proba_drop": 0.10,
        "proba_neutral": 0.80,
        "proba_spike": 0.10,
        "sentiment_score": None,
    }
    row.update(overrides)
    return row


class TestRules:
    def test_low_conviction_is_tagged_model_uncertain(self):
        assert "MODEL_UNCERTAIN" in rule_tags(
            _row(proba_drop=0.34, proba_neutral=0.34, proba_spike=0.32)
        )

    def test_confident_prediction_is_not_tagged_uncertain(self):
        assert "MODEL_UNCERTAIN" not in rule_tags(_row())

    def test_vix_shock_is_tagged_regime_shift(self, monkeypatch):
        monkeypatch.setattr(pm, "_vix_move", lambda *a, **k: 0.45)
        assert "REGIME_SHIFT" in rule_tags(_row())

    def test_calm_vix_is_not_a_regime_shift(self, monkeypatch):
        monkeypatch.setattr(pm, "_vix_move", lambda *a, **k: 0.03)
        assert "REGIME_SHIFT" not in rule_tags(_row())

    def test_flat_sentiment_during_a_big_move_is_tagged_lagged(self):
        assert "SENTIMENT_LAGGED" in rule_tags(_row(sentiment_score=0.01, actual_z=2.5))

    def test_flat_sentiment_during_a_small_move_is_not_tagged(self):
        assert "SENTIMENT_LAGGED" not in rule_tags(_row(sentiment_score=0.01, actual_z=0.6))

    def test_untagged_event_in_window_is_flagged(self, monkeypatch):
        monkeypatch.setattr(pm, "_untagged_events", lambda *a, **k: True)
        assert "EVENT_NOT_IN_TAXONOMY" in rule_tags(_row())


class StubProvider:
    def __init__(self, text="UNEXPLAINED"):
        self.text = text
        self.calls = 0

    def available(self):
        return True

    def complete(self, system, user, **kwargs):
        self.calls += 1
        return LLMResponse(self.text, "stub", "stub")


def _seed(db, prices, **overrides):
    log_prediction(
        {
            "ticker": "TEST",
            "as_of": str(prices.index[10].date()),
            "horizon_days": 5,
            "threshold_sigmas": 1.0,
            "label_scale": 0.001,  # tiny scale so the move resolves as a large one
            "last_close": 100.0,
            "probabilities": {"DROP": 0.1, "NEUTRAL": 0.8, "SPIKE": 0.1},
            "predicted_class": "DROP",
            **overrides,
        },
        db_path=db,
    )
    resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db)


class TestTagging:
    def test_llm_is_only_called_when_rules_find_nothing(self, db, prices, monkeypatch):
        provider = StubProvider()
        monkeypatch.setattr(pm, "_vix_move", lambda *a, **k: 0.50)  # rule fires

        _seed(db, prices)
        result = tag_predictions(db_path=db, provider=provider)

        assert result["tagged"] == 1
        assert "REGIME_SHIFT" in result["by_tag"]
        assert provider.calls == 0

    def test_llm_is_consulted_when_no_rule_fires(self, db, prices):
        provider = StubProvider("SENTIMENT_LAGGED")
        _seed(db, prices)
        result = tag_predictions(db_path=db, provider=provider)

        assert provider.calls == 1
        assert result["by_tag"] == {"SENTIMENT_LAGGED": 1}

    def test_an_invalid_llm_tag_falls_back_to_unexplained(self, db, prices):
        _seed(db, prices)
        result = tag_predictions(db_path=db, provider=StubProvider("THE MOON WAS FULL"))
        assert result["by_tag"] == {"UNEXPLAINED": 1}

    def test_unavailable_llm_yields_unexplained_not_a_crash(self, db, prices):
        _seed(db, prices)
        result = tag_predictions(db_path=db, use_llm=False)
        assert result["by_tag"] == {"UNEXPLAINED": 1}

    def test_tagging_is_idempotent(self, db, prices):
        _seed(db, prices)
        assert tag_predictions(db_path=db, use_llm=False)["tagged"] == 1
        assert tag_predictions(db_path=db, use_llm=False)["tagged"] == 0

    def test_correct_predictions_are_not_tagged(self, db, prices):
        _seed(db, prices, predicted_class="SPIKE")
        assert tag_predictions(db_path=db, use_llm=False)["candidates"] == 0

    def test_summary_counts_by_tag(self, db, prices):
        _seed(db, prices)
        tag_predictions(db_path=db, use_llm=False)
        assert tag_summary(db_path=db) == {"UNEXPLAINED": 1}

    def test_every_emitted_tag_is_in_the_documented_vocabulary(self, db, prices):
        _seed(db, prices)
        result = tag_predictions(db_path=db, provider=StubProvider("REGIME_SHIFT"))
        assert set(result["by_tag"]) <= set(TAGS)
