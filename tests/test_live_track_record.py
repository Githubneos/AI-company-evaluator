"""The live track record: migration, resolution per target kind, and measured skill."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from evaluator.feedback.store import connect, live_skill, log_predictions, resolve_pending

ORIGINAL_SCHEMA = """
CREATE TABLE predictions (
    prediction_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, created_at TEXT NOT NULL, as_of TEXT NOT NULL,
    horizon_days INTEGER NOT NULL, threshold_sigmas REAL NOT NULL, label_scale REAL, last_close REAL,
    proba_drop REAL NOT NULL, proba_neutral REAL NOT NULL, proba_spike REAL NOT NULL,
    predicted_class TEXT NOT NULL, sentiment_score REAL, top_features TEXT, resolved_at TEXT,
    actual_return REAL, actual_z REAL, actual_class TEXT, error_type TEXT, post_mortem_tag TEXT);
"""


def _prediction(
    ticker="AAPL", target="magnitude_1d", kind="magnitude", horizon=1, probabilities=None, as_of="2026-01-05"
):
    return {
        "ticker": ticker,
        "as_of": as_of,
        "target": target,
        "kind": kind,
        "horizon_days": horizon,
        "threshold_sigmas": 1.0,
        "label_scale": 0.02,
        "last_close": 100.0,
        "probabilities": probabilities or {"QUIET": 0.7, "LARGE_MOVE": 0.3},
        "predicted_class": "QUIET",
    }


def _prices(dates, closes) -> pd.DataFrame:
    return pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(dates))


def test_an_original_database_migrates_without_losing_a_row(tmp_path):
    path = tmp_path / "predictions.db"
    raw = sqlite3.connect(path)
    raw.executescript(ORIGINAL_SCHEMA)
    raw.execute(
        "INSERT INTO predictions VALUES ('id1','AAPL','t','2026-01-02',5,1.0,0.02,100.0,"
        "0.1,0.7,0.2,'NEUTRAL',NULL,'[]',NULL,NULL,NULL,NULL,NULL,NULL)"
    )
    raw.commit()
    raw.close()

    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM predictions").fetchall()
        info = conn.execute("PRAGMA table_info(predictions)").fetchall()
        constrained = [r[1] for r in info if r[3] and r[1].startswith("proba_")]

    assert len(rows) == 1
    assert rows[0]["prediction_id"] == "id1"
    assert (rows[0]["target"], rows[0]["kind"]) == ("direction_5d", "direction")
    assert json.loads(rows[0]["probabilities"]) == {"DROP": 0.1, "NEUTRAL": 0.7, "SPIKE": 0.2}
    assert constrained == []  # a magnitude row has no DROP probability to store


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "predictions.db"
    log_predictions([_prediction()], db_path=path)

    for _ in range(3):
        with connect(path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert "predictions_migrated" not in {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
    assert rows == 1


def test_magnitude_outcomes_resolve_on_absolute_move(tmp_path):
    path = tmp_path / "predictions.db"
    dates = pd.bdate_range("2026-01-05", periods=3)
    log_predictions(
        [
            _prediction(ticker="BIG"),  # will move 4% = 2 sigma
            _prediction(ticker="CALM"),  # will move 0.5% = 0.25 sigma
        ],
        db_path=path,
    )

    resolved = resolve_pending(
        prices_by_ticker={
            "BIG": _prices(dates, [100.0, 104.0, 104.0]),
            "CALM": _prices(dates, [100.0, 100.5, 100.5]),
        },
        db_path=path,
    )

    with connect(path) as conn:
        rows = conn.execute("SELECT ticker, actual_class FROM predictions").fetchall()
        outcomes = {r["ticker"]: r["actual_class"] for r in rows}
    assert resolved == 2
    assert outcomes == {"BIG": "LARGE_MOVE", "CALM": "QUIET"}


def test_direction_outcomes_keep_their_sign(tmp_path):
    path = tmp_path / "predictions.db"
    dates = pd.bdate_range("2026-01-05", periods=3)
    log_predictions(
        [
            _prediction(ticker="UP", target="direction_1d", kind="direction",
                        probabilities={"DROP": 0.2, "NEUTRAL": 0.5, "SPIKE": 0.3}),
            _prediction(ticker="DOWN", target="direction_1d", kind="direction",
                        probabilities={"DROP": 0.2, "NEUTRAL": 0.5, "SPIKE": 0.3}),
        ],
        db_path=path,
    )

    resolve_pending(
        prices_by_ticker={
            "UP": _prices(dates, [100.0, 104.0, 104.0]),
            "DOWN": _prices(dates, [100.0, 96.0, 96.0]),
        },
        db_path=path,
    )

    with connect(path) as conn:
        rows = conn.execute("SELECT ticker, actual_class FROM predictions").fetchall()
        outcomes = {r["ticker"]: r["actual_class"] for r in rows}
    assert outcomes == {"UP": "SPIKE", "DOWN": "DROP"}


def test_a_sector_relative_call_is_judged_against_its_sector(tmp_path, monkeypatch):
    path = tmp_path / "predictions.db"
    dates = pd.bdate_range("2026-01-05", periods=3)
    monkeypatch.setattr("evaluator.data.universe.sector_map", lambda: {"STOCK": "XLI"})
    log_predictions(
        [
            _prediction(ticker="STOCK", target="rel_direction_1d", kind="rel_direction",
                        probabilities={"DROP": 0.3, "NEUTRAL": 0.4, "SPIKE": 0.3})
        ],
        db_path=path,
    )

    # The stock falls 3%, but its sector falls 6%: relative, that is a spike.
    resolved = resolve_pending(
        prices_by_ticker={
            "STOCK": _prices(dates, [100.0, 97.0, 97.0]),
            "XLI": _prices(dates, [100.0, 94.0, 94.0]),
        },
        db_path=path,
    )

    with connect(path) as conn:
        row = conn.execute("SELECT actual_class, actual_return FROM predictions").fetchone()
    assert resolved == 1
    assert row["actual_class"] == "SPIKE"  # not DROP, which the raw move would give
    assert row["actual_return"] == pytest.approx(0.03, abs=1e-9)  # -3% minus -6%


def test_windows_are_not_resolved_early(tmp_path):
    path = tmp_path / "predictions.db"
    dates = pd.bdate_range("2026-01-05", periods=3)
    log_predictions([_prediction(ticker="SLOW", target="magnitude_20d", horizon=20)], db_path=path)

    assert resolve_pending(prices_by_ticker={"SLOW": _prices(dates, [100.0, 101.0, 102.0])}, db_path=path) == 0


def test_live_skill_measures_against_the_realised_base_rate(tmp_path):
    path = tmp_path / "predictions.db"
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-01-05", periods=300)
    truth = rng.random(300) < 0.3
    # An informative model and, for contrast, one that always says the base rate.
    log_predictions(
        [
            _prediction(ticker=f"T{i}", as_of=str(d.date()),
                        probabilities={"QUIET": 0.2 if hit else 0.9, "LARGE_MOVE": 0.8 if hit else 0.1})
            for i, (d, hit) in enumerate(zip(dates, truth))
        ],
        db_path=path,
    )
    with connect(path) as conn:  # resolve them by hand, as if the windows had closed
        for (pid,), hit in zip(conn.execute("SELECT prediction_id FROM predictions ORDER BY as_of"), truth):
            conn.execute(
                "UPDATE predictions SET resolved_at='t', actual_class=?, actual_z=? WHERE prediction_id=?",
                ("LARGE_MOVE" if hit else "QUIET", 2.0 if hit else 0.1, pid),
            )

    report = live_skill(db_path=path, min_rows=200)
    entry = report["targets"]["magnitude_1d"]

    assert entry["n"] == 300 and entry["sufficient"] is True
    assert entry["brier_skill"] > 0.5  # the model knew the answer
    assert entry["base_rates"]["LARGE_MOVE"] == pytest.approx(truth.mean(), abs=0.01)
    assert report["any_sufficient"] is True


def test_too_few_resolved_predictions_is_stated_not_scored(tmp_path):
    path = tmp_path / "predictions.db"
    log_predictions([_prediction()], db_path=path)
    with connect(path) as conn:
        conn.execute("UPDATE predictions SET resolved_at='t', actual_class='QUIET'")

    report = live_skill(db_path=path, min_rows=200)
    entry = report["targets"]["magnitude_1d"]

    assert entry["sufficient"] is False
    assert "too few" in entry["note"]
    assert report["any_sufficient"] is False
