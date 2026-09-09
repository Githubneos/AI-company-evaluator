"""Feedback-loop tests.

The property that matters: a resolved outcome must be scored against the
volatility scale that was current when the prediction was made, so that
"was this right?" is answered on the same terms the model was trained on.
"""

import numpy as np
import pandas as pd
import pytest

from evaluator.feedback.store import feedback_context, log_prediction, resolve_pending


@pytest.fixture
def db(tmp_path):
    return tmp_path / "predictions.db"


@pytest.fixture
def prices():
    dates = pd.bdate_range("2024-01-01", periods=60, name="date")
    close = pd.Series(np.linspace(100.0, 130.0, len(dates)), index=dates)
    return pd.DataFrame({"close": close})


def _prediction(as_of: str, predicted: str = "NEUTRAL", scale: float = 0.02) -> dict:
    return {
        "ticker": "TEST",
        "as_of": as_of,
        "horizon_days": 5,
        "threshold_sigmas": 1.0,
        "label_scale": scale,
        "last_close": 100.0,
        "probabilities": {"DROP": 0.15, "NEUTRAL": 0.70, "SPIKE": 0.15},
        "predicted_class": predicted,
        "top_features": [{"feature": "rsi_14", "shap_contribution": 0.1}],
    }


def test_open_windows_are_not_resolved(db, prices):
    """A prediction whose horizon has not elapsed must stay unresolved."""
    last_date = str(prices.index[-1].date())
    log_prediction(_prediction(last_date), db_path=db)

    assert resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db) == 0
    assert feedback_context("TEST", db_path=db)["resolved_predictions"] == 0
    assert feedback_context("TEST", db_path=db)["open_predictions"] == 1


def test_resolution_uses_the_scale_recorded_at_prediction_time(db, prices):
    as_of = prices.index[10]
    log_prediction(_prediction(str(as_of.date()), scale=0.02), db_path=db)

    assert resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db) == 1

    context = feedback_context("TEST", db_path=db)
    resolved = context["recent"][0]

    start, end = prices["close"].iloc[10], prices["close"].iloc[15]
    assert resolved["actual_return"] == pytest.approx(end / start - 1.0, abs=1e-4)
    # 5-day move on this rising series far exceeds 1 * 0.02, so it is a SPIKE.
    assert resolved["actual"] == "SPIKE"
    assert resolved["predicted"] == "NEUTRAL"


def test_hit_rate_counts_only_resolved_predictions(db, prices):
    log_prediction(_prediction(str(prices.index[5].date()), predicted="SPIKE"), db_path=db)
    log_prediction(_prediction(str(prices.index[6].date()), predicted="DROP"), db_path=db)
    log_prediction(_prediction(str(prices.index[-1].date())), db_path=db)

    resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db)
    context = feedback_context("TEST", db_path=db)

    assert context["resolved_predictions"] == 2
    assert context["open_predictions"] == 1
    assert context["hit_rate"] == pytest.approx(0.5)


def test_missed_moves_surface_neutral_calls_that_moved(db, prices):
    log_prediction(_prediction(str(prices.index[8].date()), predicted="NEUTRAL"), db_path=db)
    resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db)

    missed = feedback_context("TEST", db_path=db)["missed_moves"]
    assert len(missed) == 1
    assert missed[0]["predicted"] == "NEUTRAL"
    assert missed[0]["actual"] == "SPIKE"


def test_empty_track_record_is_stated_not_implied(db):
    context = feedback_context("NEVERSEEN", db_path=db)
    assert context["resolved_predictions"] == 0
    assert "no track record" in context["note"].lower()


def test_resolution_is_idempotent(db, prices):
    log_prediction(_prediction(str(prices.index[10].date())), db_path=db)

    assert resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db) == 1
    assert resolve_pending(prices_by_ticker={"TEST": prices}, db_path=db) == 0
