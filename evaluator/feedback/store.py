"""Prediction log and outcome resolution (spec 6.1, 6.4).

Built now rather than with the rest of Phase 6 because the log is the only part
of the feedback loop that cannot be backfilled: a prediction that was never
recorded at the moment it was made cannot be scored honestly later. Everything
downstream -- post-mortem tagging, retraining triggers, the `feedback_context`
the LLM evaluator reads -- is reconstructable from this table. The table itself
is not.

Resolution compares the realised move against the same volatility scale that
was current when the prediction was made, so a resolved outcome is directly
comparable to the label the model was trained on.

SQLite because the write rate is one row per prediction and the read pattern is
"recent rows for one ticker". Postgres when this becomes multi-tenant.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluator.config import ARTIFACT_DIR, CLASS_NAMES, DROP, NEUTRAL, SPIKE

DB_PATH = Path(ARTIFACT_DIR) / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     TEXT PRIMARY KEY,
    ticker            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    as_of             TEXT NOT NULL,
    horizon_days      INTEGER NOT NULL,
    threshold_sigmas  REAL NOT NULL,
    label_scale       REAL,
    last_close        REAL,
    proba_drop        REAL NOT NULL,
    proba_neutral     REAL NOT NULL,
    proba_spike       REAL NOT NULL,
    predicted_class   TEXT NOT NULL,
    sentiment_score   REAL,
    top_features      TEXT,
    resolved_at       TEXT,
    actual_return     REAL,
    actual_z          REAL,
    actual_class      TEXT,
    error_type        TEXT,
    post_mortem_tag   TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker, as_of);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(resolved_at);
"""


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(score: dict, *, db_path: Path | str = DB_PATH) -> str:
    """Record a prediction at the moment it is issued. Returns its id."""
    prediction_id = str(uuid.uuid4())
    probabilities = score["probabilities"]

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                prediction_id, ticker, created_at, as_of, horizon_days,
                threshold_sigmas, label_scale, last_close,
                proba_drop, proba_neutral, proba_spike, predicted_class,
                sentiment_score, top_features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                score["ticker"],
                datetime.now(timezone.utc).isoformat(),
                score["as_of"],
                score["horizon_days"],
                score["threshold_sigmas"],
                score.get("label_scale"),
                score.get("last_close"),
                probabilities[CLASS_NAMES[DROP]],
                probabilities[CLASS_NAMES[NEUTRAL]],
                probabilities[CLASS_NAMES[SPIKE]],
                score["predicted_class"],
                score.get("sentiment_score"),
                json.dumps(score.get("top_features", [])),
            ),
        )
    return prediction_id


def _classify(z: float, threshold: float) -> str:
    if z >= threshold:
        return CLASS_NAMES[SPIKE]
    if z <= -threshold:
        return CLASS_NAMES[DROP]
    return CLASS_NAMES[NEUTRAL]


def resolve_pending(
    *,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
    db_path: Path | str = DB_PATH,
) -> int:
    """Fill in outcomes for predictions whose forward window has now closed.

    `prices_by_ticker` is injectable so this is testable without a network call;
    when omitted, prices are fetched per ticker.
    """
    with connect(db_path) as conn:
        pending = conn.execute(
            "SELECT * FROM predictions WHERE resolved_at IS NULL"
        ).fetchall()

        if not pending:
            return 0

        tickers = {row["ticker"] for row in pending}
        prices = dict(prices_by_ticker or {})
        for ticker in tickers - prices.keys():
            from evaluator.data.sources import load_prices

            prices[ticker] = load_prices(ticker, "2015-01-01", None, use_cache=False)

        resolved = 0
        for row in pending:
            frame = prices.get(row["ticker"])
            if frame is None or frame.empty:
                continue

            close = frame["close"]
            as_of = pd.Timestamp(row["as_of"])
            if as_of not in close.index:
                continue

            start = close.index.get_loc(as_of)
            end = start + row["horizon_days"]
            if end >= len(close):
                continue  # window still open

            actual_return = float(close.iloc[end] / close.iloc[start] - 1.0)
            scale = row["label_scale"]
            actual_z = float(actual_return / scale) if scale else None
            actual_class = (
                _classify(actual_z, row["threshold_sigmas"]) if actual_z is not None else None
            )
            error_type = (
                None
                if actual_class is None
                else ("correct" if actual_class == row["predicted_class"] else "incorrect")
            )

            conn.execute(
                """
                UPDATE predictions
                   SET resolved_at = ?, actual_return = ?, actual_z = ?,
                       actual_class = ?, error_type = ?
                 WHERE prediction_id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    actual_return,
                    actual_z,
                    actual_class,
                    error_type,
                    row["prediction_id"],
                ),
            )
            resolved += 1

    return resolved


def feedback_context(
    ticker: str, *, limit: int = 20, db_path: Path | str = DB_PATH
) -> dict:
    """Track record on this ticker, shaped for the LLM evaluator's context.

    This is the `feedback_context` of spec 5 and 6.4: what the system has
    previously claimed about this name and how those claims turned out. It is
    the only channel through which the reasoning layer learns between retrains.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT as_of, predicted_class, actual_class, error_type,
                   actual_return, actual_z, proba_drop, proba_neutral, proba_spike
              FROM predictions
             WHERE ticker = ? AND resolved_at IS NOT NULL
             ORDER BY as_of DESC
             LIMIT ?
            """,
            (ticker, limit),
        ).fetchall()
        open_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE ticker = ? AND resolved_at IS NULL",
            (ticker,),
        ).fetchone()[0]

    if not rows:
        return {
            "resolved_predictions": 0,
            "open_predictions": int(open_count),
            "note": (
                "No resolved predictions for this ticker yet. The system has no "
                "track record here; treat the model output on its validation "
                "metrics alone."
            ),
        }

    correct = sum(1 for r in rows if r["error_type"] == "correct")
    missed_moves = [
        {
            "as_of": r["as_of"],
            "predicted": r["predicted_class"],
            "actual": r["actual_class"],
            "actual_return": round(r["actual_return"], 4),
        }
        for r in rows
        if r["predicted_class"] == "NEUTRAL" and r["actual_class"] != "NEUTRAL"
    ]

    return {
        "resolved_predictions": len(rows),
        "open_predictions": int(open_count),
        "hit_rate": round(correct / len(rows), 4),
        "missed_moves": missed_moves[:5],
        "recent": [
            {
                "as_of": r["as_of"],
                "predicted": r["predicted_class"],
                "actual": r["actual_class"],
                "actual_return": round(r["actual_return"], 4),
            }
            for r in rows[:5]
        ],
    }
