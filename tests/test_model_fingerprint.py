"""A prediction is only comparable with another from the same trained model."""

import json

import numpy as np
import pandas as pd
import pytest

from evaluator.feedback.store import connect, live_skill, log_prediction


def _score(target="direction_5d", fp="aaa111", cls="SPIKE"):
    return {
        "ticker": "T", "as_of": "2026-01-05", "target": target, "kind": "direction",
        "horizon_days": 5, "threshold_sigmas": 1.0, "label_scale": 0.02, "last_close": 100.0,
        "probabilities": {"DROP": 0.2, "NEUTRAL": 0.5, "SPIKE": 0.3},
        "predicted_class": cls, "model_fingerprint": fp,
    }


def test_fingerprint_is_recorded_and_round_trips(tmp_path):
    db = tmp_path / "p.db"
    pid = log_prediction(_score(), db_path=db)
    with connect(db) as conn:
        row = conn.execute("SELECT model_fingerprint FROM predictions WHERE prediction_id=?", (pid,)).fetchone()
    assert row["model_fingerprint"] == "aaa111"


def test_migration_from_an_older_schema_keeps_every_row_and_nulls_the_new_column(tmp_path):
    db = tmp_path / "old.db"
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE predictions (
            prediction_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, created_at TEXT NOT NULL,
            as_of TEXT NOT NULL, horizon_days INTEGER NOT NULL, threshold_sigmas REAL NOT NULL,
            label_scale REAL, last_close REAL, proba_drop REAL NOT NULL, proba_neutral REAL NOT NULL,
            proba_spike REAL NOT NULL, predicted_class TEXT NOT NULL, sentiment_score REAL,
            top_features TEXT, resolved_at TEXT, actual_return REAL, actual_z REAL,
            actual_class TEXT, error_type TEXT, post_mortem_tag TEXT
        )""")
    for i in range(3):
        conn.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"id{i}", "T", "2025-01-01T00:00:00", "2025-01-01", 5, 1.0, 0.02, 100.0,
             0.2, 0.5, 0.3, "SPIKE", None, "[]", None, None, None, None, None, None),
        )
    conn.commit()
    conn.close()

    with connect(db) as c:
        n = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        cols = {r[1] for r in c.execute("PRAGMA table_info(predictions)")}
        fps = [r[0] for r in c.execute("SELECT model_fingerprint FROM predictions")]

    assert n == 3
    assert "model_fingerprint" in cols
    assert fps == [None, None, None]


def test_live_skill_splits_by_model_version_not_just_by_target(tmp_path, monkeypatch):
    db = tmp_path / "p.db"
    now = "2026-02-01T00:00:00"
    rows = [
        dict(prediction_id="a", ticker="T", created_at=now, as_of="2026-01-01", target="direction_5d",
             kind="direction", horizon_days=5, threshold_sigmas=1.0, label_scale=0.02, last_close=100.0,
             probabilities=json.dumps({"DROP": 0.2, "NEUTRAL": 0.5, "SPIKE": 0.3}),
             predicted_class="SPIKE", model_fingerprint="old_model",
             resolved_at=now, actual_class="SPIKE"),
        dict(prediction_id="b", ticker="T", created_at=now, as_of="2026-01-08", target="direction_5d",
             kind="direction", horizon_days=5, threshold_sigmas=1.0, label_scale=0.02, last_close=100.0,
             probabilities=json.dumps({"DROP": 0.5, "NEUTRAL": 0.3, "SPIKE": 0.2}),
             predicted_class="DROP", model_fingerprint="new_model",
             resolved_at=now, actual_class="DROP"),
    ]
    with connect(db) as conn:
        for r in rows:
            cols = ", ".join(r)
            placeholders = ", ".join("?" for _ in r)
            conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({placeholders})", tuple(r.values()))

    result = live_skill(db_path=db, min_rows=1)
    entry = result["targets"]["direction_5d"]

    assert set(entry["model_versions"]) == {"old_model", "new_model"}
    assert len(entry["per_model"]) == 2
    assert "note_versions" in entry
    assert "2 model versions" in entry["note_versions"]
