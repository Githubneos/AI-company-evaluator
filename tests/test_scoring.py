"""Nightly universe scoring and the leaderboard it feeds."""

import json

import pandas as pd
import pytest

import evaluator.scoring as scoring


def _scored(ticker: str, probability: float, as_of: str = "2026-09-18") -> dict:
    return {
        "ticker": ticker,
        "as_of": as_of,
        "last_close": 100.0,
        "move_threshold_pct": 0.02,
        "targets": {
            "magnitude_1d": {
                "kind": "magnitude",
                "horizon_days": 1,
                "predicted_class": "QUIET",
                "probabilities": {"QUIET": 1 - probability, "LARGE_MOVE": probability},
                "baseline_probabilities": {"QUIET": 0.74, "LARGE_MOVE": 0.26},
                "lift_over_baseline": {"QUIET": 1.0, "LARGE_MOVE": probability / 0.26},
            }
        },
    }


@pytest.fixture
def scores_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "SCORES_DIR", tmp_path)
    monkeypatch.setattr(scoring, "LATEST_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(scoring, "available_targets", lambda: ["magnitude_1d"])
    monkeypatch.setattr(
        scoring,
        "load_universe",
        lambda: pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "sector": ["Energy", "Energy", "Utilities"]}),
    )
    return tmp_path


def test_one_failing_ticker_does_not_sink_the_run(scores_dir, monkeypatch):
    def score_ticker(ticker, **kwargs):
        if ticker == "BBB":
            raise ValueError("no price data")
        return _scored(ticker, 0.3 if ticker == "AAA" else 0.5)

    monkeypatch.setattr(scoring, "score_ticker", score_ticker)
    summary = scoring.score_universe()

    assert summary["tickers"] == 2 and summary["rows"] == 2
    assert list(summary["failures"]) == ["BBB"]
    assert "no price data" in summary["failures"]["BBB"]
    assert json.loads((scores_dir / "latest.json").read_text())["as_of"] == "2026-09-18"


def test_scoring_nothing_at_all_is_an_error_not_an_empty_table(scores_dir, monkeypatch):
    monkeypatch.setattr(scoring, "score_ticker", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    with pytest.raises(RuntimeError, match="scored nothing"):
        scoring.score_universe()


def test_scoring_without_promoted_models_refuses(scores_dir, monkeypatch):
    monkeypatch.setattr(scoring, "available_targets", list)

    with pytest.raises(scoring.ModelNotTrained, match="promote"):
        scoring.score_universe()


def _write_table(directory, rows, as_of="2026-09-18"):
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / f"{as_of}.parquet", index=False)
    (directory / "latest.json").write_text(
        json.dumps({"as_of": as_of, "tickers": frame["ticker"].nunique(), "targets": ["magnitude_1d"], "failures": {}})
    )


def test_leaderboard_ranks_and_filters(scores_dir):
    _write_table(
        scores_dir,
        [
            {"ticker": "AAA", "sector": "Energy", "target": "magnitude_1d", "probability": 0.30, "lift": 1.2},
            {"ticker": "BBB", "sector": "Energy", "target": "magnitude_1d", "probability": 0.55, "lift": 2.1},
            {"ticker": "CCC", "sector": "Utilities", "target": "magnitude_1d", "probability": 0.45, "lift": 1.7},
            {"ticker": "DDD", "sector": "Energy", "target": "magnitude_5d", "probability": 0.99, "lift": 3.0},
        ],
    )

    board = scoring.leaderboard("magnitude_1d", limit=2)
    assert [r["ticker"] for r in board["rows"]] == ["BBB", "CCC"]  # ranked by probability
    assert board["sectors"] == ["Energy", "Utilities"]

    energy = scoring.leaderboard("magnitude_1d", sector="Energy")
    assert [r["ticker"] for r in energy["rows"]] == ["BBB", "AAA"]  # other sectors excluded
    assert all(r["target"] == "magnitude_1d" for r in energy["rows"])  # other targets excluded


def test_leaderboard_flags_a_stale_table(scores_dir, monkeypatch):
    _write_table(scores_dir, [{"ticker": "AAA", "sector": "Energy", "target": "magnitude_1d", "probability": 0.3}])

    fresh = pd.Timestamp("2026-09-18 20:00")
    monkeypatch.setattr(scoring.pd.Timestamp, "now", staticmethod(lambda *a: fresh))
    assert scoring.leaderboard()["stale"] is False

    week_later = pd.Timestamp("2026-09-25 20:00")
    monkeypatch.setattr(scoring.pd.Timestamp, "now", staticmethod(lambda *a: week_later))
    stale = scoring.leaderboard()
    assert stale["stale"] is True and stale["stale_trading_days"] == 5


def test_leaderboard_without_a_run_says_so(scores_dir):
    board = scoring.leaderboard()

    assert board["available"] is False
    assert "score_universe" in board["reason"] and board["rows"] == []
