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
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evaluator.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    DIRECTION,
    DROP,
    LARGE_MOVE,
    MAGNITUDE,
    MAGNITUDE_NAMES,
    NEUTRAL,
    QUIET,
    REL_DIRECTION,
    SPIKE,
)

log = logging.getLogger(__name__)

DB_PATH = Path(ARTIFACT_DIR) / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     TEXT PRIMARY KEY,
    ticker            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    as_of             TEXT NOT NULL,
    target            TEXT NOT NULL DEFAULT 'direction_5d',
    kind              TEXT NOT NULL DEFAULT 'direction',
    horizon_days      INTEGER NOT NULL,
    threshold_sigmas  REAL NOT NULL,
    label_scale       REAL,
    last_close        REAL,
    probabilities     TEXT NOT NULL DEFAULT '{}',
    proba_drop        REAL,
    proba_neutral     REAL,
    proba_spike       REAL,
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
"""

#: Created after the migration: an index cannot reference a column that an
#: older database has not been given yet.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker, as_of);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(resolved_at);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target, as_of);
"""



@contextmanager
def connect(db_path: Path | str | None = None):
    # Late-bound: a default argument would capture DB_PATH at import time, so a
    # test (or a second database) could not redirect it.
    path = Path(db_path if db_path is not None else DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.executescript(INDEXES)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database to the current shape, once, without losing rows.

    The first schema hard-coded three direction probability columns as NOT
    NULL, which a magnitude prediction cannot satisfy. SQLite cannot relax a
    constraint in place, so the table is rebuilt and copied: the log is the one
    thing here that cannot be reconstructed, so it is never dropped.
    """
    info = {row[1]: row for row in conn.execute("PRAGMA table_info(predictions)")}
    # Rebuild when a column is missing *or* when the old NOT NULL constraints
    # are still there: a magnitude prediction has no DROP probability to store,
    # and a half-migrated database fails only at the first insert.
    required = {"target", "kind", "probabilities"}
    constrained = any(info[c][3] for c in ("proba_drop", "proba_neutral", "proba_spike") if c in info)
    if required <= set(info) and not constrained:
        return

    log.info("migrating the prediction log to multi-target shape")
    # The source table may be the original shape or a half-migrated one, so the
    # three new columns are read only where they exist.
    fallback_probabilities = "json_object('DROP', proba_drop, 'NEUTRAL', proba_neutral, 'SPIKE', proba_spike)"
    target_expr = "'direction_' || horizon_days || 'd'"
    kind_expr = "'direction'"
    probabilities_expr = fallback_probabilities
    if "target" in info:
        target_expr = f"COALESCE(NULLIF(target, ''), {target_expr})"
    if "kind" in info:
        kind_expr = f"COALESCE(NULLIF(kind, ''), {kind_expr})"
    if "probabilities" in info:
        probabilities_expr = (
            f"CASE WHEN probabilities IS NULL OR probabilities IN ('', '{{}}') "
            f"THEN {fallback_probabilities} ELSE probabilities END"
        )

    conn.execute("DROP TABLE IF EXISTS predictions_migrated")
    conn.executescript(SCHEMA.replace("predictions", "predictions_migrated"))
    conn.execute(
        f"""
        INSERT INTO predictions_migrated (
            prediction_id, ticker, created_at, as_of, target, kind, horizon_days,
            threshold_sigmas, label_scale, last_close, probabilities,
            proba_drop, proba_neutral, proba_spike, predicted_class,
            sentiment_score, top_features, resolved_at, actual_return, actual_z,
            actual_class, error_type, post_mortem_tag
        )
        SELECT
            prediction_id, ticker, created_at, as_of,
            {target_expr}, {kind_expr}, horizon_days,
            threshold_sigmas, label_scale, last_close, {probabilities_expr},
            proba_drop, proba_neutral, proba_spike, predicted_class,
            sentiment_score, top_features, resolved_at, actual_return, actual_z,
            actual_class, error_type, post_mortem_tag
          FROM predictions
        """
    )
    moved = conn.execute("SELECT COUNT(*) FROM predictions_migrated").fetchone()[0]
    original = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if moved != original:  # pragma: no cover - refuse to drop rows
        raise RuntimeError(f"migration would lose rows: {original} -> {moved}")
    conn.execute("DROP TABLE predictions")
    conn.execute("ALTER TABLE predictions_migrated RENAME TO predictions")
    log.info("prediction log migrated: %d rows kept", moved)


def log_prediction(score: dict, *, db_path: Path | str | None = None) -> str:
    """Record a prediction at the moment it is issued. Returns its id."""
    with connect(db_path) as conn:
        return _insert(conn, score)


def log_predictions(scores: list[dict], *, db_path: Path | str | None = None) -> list[str]:
    """Record many predictions in one transaction (the nightly run)."""
    with connect(db_path) as conn:
        return [_insert(conn, score) for score in scores]


def _insert(conn: sqlite3.Connection, score: dict) -> str:
    prediction_id = str(uuid.uuid4())
    probabilities = score["probabilities"]
    horizon = int(score["horizon_days"])
    kind = score.get("kind", DIRECTION)
    conn.execute(
        """
        INSERT INTO predictions (
            prediction_id, ticker, created_at, as_of, target, kind, horizon_days,
            threshold_sigmas, label_scale, last_close, probabilities,
            proba_drop, proba_neutral, proba_spike, predicted_class,
            sentiment_score, top_features
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            score["ticker"],
            datetime.now(UTC).isoformat(),
            score["as_of"],
            score.get("target", f"{kind}_{horizon}d"),
            kind,
            horizon,
            score["threshold_sigmas"],
            score.get("label_scale"),
            score.get("last_close"),
            json.dumps(probabilities),
            # The three direction columns predate targets; they stay filled for
            # the classes they describe and null for magnitude.
            probabilities.get(CLASS_NAMES[DROP]),
            probabilities.get(CLASS_NAMES[NEUTRAL]),
            probabilities.get(CLASS_NAMES[SPIKE]),
            score["predicted_class"],
            score.get("sentiment_score"),
            json.dumps(score.get("top_features", [])),
        ),
    )
    return prediction_id


def _classify(z: float, threshold: float, kind: str = DIRECTION) -> str:
    """The realised class, in the same terms the model predicted."""
    if kind == MAGNITUDE:
        return MAGNITUDE_NAMES[LARGE_MOVE] if abs(z) >= threshold else MAGNITUDE_NAMES[QUIET]
    if z >= threshold:
        return CLASS_NAMES[SPIKE]
    if z <= -threshold:
        return CLASS_NAMES[DROP]
    return CLASS_NAMES[NEUTRAL]


def resolve_pending(
    *,
    prices_by_ticker: dict[str, pd.DataFrame] | None = None,
    db_path: Path | str | None = None,
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

        # Sector-relative predictions are judged against the sector, so the
        # ETF's own move has to be available too.
        sectors: dict[str, pd.DataFrame] = {}
        if any(row["kind"] == REL_DIRECTION for row in pending):
            from evaluator.data.universe import sector_map

            mapping = sector_map()
            for ticker in tickers:
                etf = mapping.get(ticker)
                if etf is None:
                    continue
                if etf in prices:
                    sectors[ticker] = prices[etf]
                    continue
                try:
                    from evaluator.data.sources import load_prices

                    sectors[ticker] = load_prices(etf, "2015-01-01", None, use_cache=True)
                except Exception as exc:  # noqa: BLE001 - one missing ETF is not fatal
                    log.warning("sector prices unavailable for %s (%s): %s", ticker, etf, exc)

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
            if row["kind"] == REL_DIRECTION:
                sector = sectors.get(row["ticker"])
                if sector is None or as_of not in sector.index:
                    continue  # cannot judge a relative call without the sector
                sector_close = sector["close"]
                s_start = sector_close.index.get_loc(as_of)
                s_end = s_start + row["horizon_days"]
                if s_end >= len(sector_close):
                    continue
                actual_return -= float(sector_close.iloc[s_end] / sector_close.iloc[s_start] - 1.0)
            scale = row["label_scale"]
            actual_z = float(actual_return / scale) if scale else None
            actual_class = (
                _classify(actual_z, row["threshold_sigmas"], row["kind"]) if actual_z is not None else None
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
                    datetime.now(UTC).isoformat(),
                    actual_return,
                    actual_z,
                    actual_class,
                    error_type,
                    row["prediction_id"],
                ),
            )
            resolved += 1

    return resolved


def _mean_brier(records: list[dict], classes: list[str], forecasts: list[dict]) -> float:
    """Mean multi-class Brier score of `forecasts` against the realised classes."""
    total = 0.0
    for record, probabilities in zip(records, forecasts):
        total += sum((probabilities.get(c, 0.0) - (record["actual_class"] == c)) ** 2 for c in classes)
    return total / len(records)


#: Below this many resolved predictions per target, live skill is noise.
MIN_LIVE_ROWS = 200


def live_skill(*, db_path: Path | str | None = None, min_rows: int = MIN_LIVE_ROWS) -> dict:
    """Out-of-sample skill on predictions the system actually issued.

    The backtest is a claim about the past; this is the record. Each resolved
    prediction is scored against the class priors of its own target, taken from
    the predictions themselves, so "skill" means the same thing it does in
    training: better than guessing the base rate.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT target, kind, as_of, probabilities, actual_class, predicted_class
              FROM predictions
             WHERE resolved_at IS NOT NULL AND actual_class IS NOT NULL
             ORDER BY as_of
            """
        ).fetchall()

    by_target: dict[str, list[dict]] = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(dict(row))

    out = {}
    for target, records in sorted(by_target.items()):
        classes = sorted({c for r in records for c in json.loads(r["probabilities"])})
        if not classes:
            continue
        priors = {c: sum(r["actual_class"] == c for r in records) / len(records) for c in classes}
        model = _mean_brier(records, classes, [json.loads(r["probabilities"]) for r in records])
        baseline = _mean_brier(records, classes, [priors] * len(records))
        entry = {
            "target": target,
            "kind": records[0]["kind"],
            "n": len(records),
            "first": records[0]["as_of"],
            "last": records[-1]["as_of"],
            "brier": round(model, 5),
            "brier_baseline": round(baseline, 5),
            "brier_skill": round(1 - model / baseline, 5) if baseline > 0 else None,
            "accuracy": round(sum(r["actual_class"] == r["predicted_class"] for r in records) / len(records), 4),
            "base_rates": {c: round(p, 4) for c, p in priors.items()},
            "sufficient": len(records) >= min_rows,
        }
        if not entry["sufficient"]:
            entry["note"] = (
                f"{len(records)} resolved predictions is too few to measure live skill; "
                f"{min_rows} is the floor. Treat this as a count, not a result."
            )
        out[target] = entry
    return {
        "targets": out,
        "min_rows": min_rows,
        "resolved_total": len(rows),
        "any_sufficient": any(e["sufficient"] for e in out.values()),
    }


def feedback_context(
    ticker: str, *, limit: int = 20, db_path: Path | str | None = None
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
