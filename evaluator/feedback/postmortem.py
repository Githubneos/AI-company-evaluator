"""Automated post-mortem tagging (spec 6.2).

For resolved predictions that were badly wrong, work out *why*. Rules run first
and an LLM is consulted only for what the rules cannot explain -- which keeps
cost down, but more importantly keeps the tags trustworthy: a rule that fires on
a measured VIX jump is evidence, while an LLM asked to explain a move it cannot
verify will always produce a plausible story.

Tags follow the spec's categories:

  EVENT_NOT_IN_TAXONOMY   An event type the labelling scheme cannot express
                          (guidance, litigation, ratings) preceded the move.
  REGIME_SHIFT            Market conditions moved faster than the regime
                          features captured.
  SENTIMENT_LAGGED        Sentiment was stale or flat while the price moved.
  ANALOG_POOR_MATCH       Retrieved analogs turned out to resemble nothing.
  MODEL_UNCERTAIN         The model itself signalled low conviction; the error
                          is honest rather than diagnostic.
  UNEXPLAINED             Nothing identifiable. Kept as a real category rather
                          than forcing every miss into a story.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.feedback.store import DB_PATH, connect

log = logging.getLogger(__name__)

LARGE_ERROR_SIGMAS = 1.5
VIX_SHOCK_THRESHOLD = 0.25
STALE_SENTIMENT_HOURS = 24

TAGS = [
    "EVENT_NOT_IN_TAXONOMY",
    "REGIME_SHIFT",
    "SENTIMENT_LAGGED",
    "ANALOG_POOR_MATCH",
    "MODEL_UNCERTAIN",
    "UNEXPLAINED",
]


def _vix_move(as_of: pd.Timestamp, horizon: int) -> float | None:
    """Fractional VIX change across the prediction window."""
    try:
        from evaluator.data.sources import load_prices

        vix = load_prices("^VIX", "2005-01-01", None)["close"]
    except Exception as exc:  # noqa: BLE001
        log.warning("VIX unavailable for post-mortem: %s", exc)
        return None

    vix.index = pd.DatetimeIndex(vix.index)
    window = vix[vix.index >= as_of]
    if len(window) < 2:
        return None
    end = window.iloc[: horizon + 1]
    return float(end.iloc[-1] / end.iloc[0] - 1.0)


def _untagged_events(ticker: str, as_of: pd.Timestamp, horizon: int) -> bool:
    """Did an 8-K land in the window whose item code the taxonomy cannot express?"""
    from evaluator.data.universe import cik_for
    from evaluator.events import UNPOPULATED_TYPES, classify_item, parse_items

    cik = cik_for(ticker)
    if cik is None:
        return False
    try:
        from evaluator.data.edgar import EdgarClient

        filings = EdgarClient().filings(cik, ticker, use_cache=True)
    except Exception:  # noqa: BLE001
        return False
    if filings.empty:
        return False

    window = filings[
        (pd.to_datetime(filings["filing_date"]) >= as_of)
        & (pd.to_datetime(filings["filing_date"]) <= as_of + pd.Timedelta(days=horizon * 2))
        & (filings["form"] == "8-K")
    ]
    for row in window.itertuples(index=False):
        for code in parse_items(row.items):
            event_type, _ = classify_item(code)
            # Item 8.01 "other events" is the bucket the unmappable categories
            # land in -- guidance, litigation, dividends all arrive as DISCLOSURE.
            # Those are exactly the UNPOPULATED_TYPES the taxonomy cannot express,
            # so an 8.01 in the window is the best available evidence that the
            # model was blind to a real catalyst.
            if event_type in ("DISCLOSURE", "UNCLASSIFIED_8K"):
                return True
    return False


def rule_tags(row: dict) -> list[str]:
    """Tags derivable from measured quantities alone."""
    tags = []
    as_of = pd.Timestamp(row["as_of"])
    horizon = int(row["horizon_days"])

    probabilities = [row["proba_drop"], row["proba_neutral"], row["proba_spike"]]
    if max(probabilities) < 0.5:
        tags.append("MODEL_UNCERTAIN")

    vix_move = _vix_move(as_of, horizon)
    if vix_move is not None and abs(vix_move) >= VIX_SHOCK_THRESHOLD:
        tags.append("REGIME_SHIFT")

    try:
        if _untagged_events(row["ticker"], as_of, horizon):
            tags.append("EVENT_NOT_IN_TAXONOMY")
    except Exception as exc:  # noqa: BLE001
        log.debug("event check failed: %s", exc)

    sentiment = row.get("sentiment_score")
    actual_z = row.get("actual_z")
    if sentiment is not None and actual_z is not None:
        if abs(sentiment) < 0.1 and abs(actual_z) >= LARGE_ERROR_SIGMAS:
            tags.append("SENTIMENT_LAGGED")

    return tags


LLM_PROMPT = """\
You are tagging why a market-risk prediction was wrong. You are given the \
prediction, what actually happened, and the checks that have already run.

Choose exactly ONE tag from this list and reply with only that tag:

EVENT_NOT_IN_TAXONOMY - a company event the model's event taxonomy cannot \
express (guidance change, litigation, dividend change, credit rating action)
REGIME_SHIFT - broad market conditions changed faster than the model tracked
SENTIMENT_LAGGED - news sentiment failed to reflect what was happening
MODEL_UNCERTAIN - the model expressed low conviction and was simply wrong
UNEXPLAINED - no identifiable cause

Rules: you have no information beyond what is given. Do not invent a news event \
you cannot see evidence for. If the data does not support a specific cause, \
answer UNEXPLAINED -- that is a correct and useful answer, not a failure."""


def llm_tag(row: dict, provider=None) -> str | None:
    """Ask an LLM only when the rules found nothing."""
    from evaluator.llm.provider import GEMINI_TAGGER_MODEL, LLMUnavailable, get_provider

    provider = provider or get_provider(model=GEMINI_TAGGER_MODEL)
    context = {
        "ticker": row["ticker"],
        "as_of": row["as_of"],
        "horizon_days": row["horizon_days"],
        "predicted_class": row["predicted_class"],
        "actual_class": row["actual_class"],
        "actual_return": row["actual_return"],
        "actual_z": row["actual_z"],
        "model_probabilities": {
            "DROP": row["proba_drop"],
            "NEUTRAL": row["proba_neutral"],
            "SPIKE": row["proba_spike"],
        },
        "rule_checks_found_nothing": True,
    }

    try:
        response = provider.complete(LLM_PROMPT, json.dumps(context, indent=2, default=str))
    except LLMUnavailable as exc:
        log.info("LLM tagging unavailable: %s", exc)
        return None

    if response.refused or not response.text:
        return None
    tag = response.text.strip().split()[0].upper().strip(".,")
    return tag if tag in TAGS else "UNEXPLAINED"


def tag_predictions(
    *,
    db_path: Path | str = DB_PATH,
    use_llm: bool = True,
    limit: int = 200,
    provider=None,
) -> dict:
    """Weekly job: tag resolved predictions that were materially wrong."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
             WHERE resolved_at IS NOT NULL
               AND post_mortem_tag IS NULL
               AND error_type = 'incorrect'
             ORDER BY as_of DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()

        counts: dict[str, int] = {}
        tagged = 0

        for record in rows:
            row = dict(record)
            if row.get("actual_z") is not None and abs(row["actual_z"]) < 0.5:
                continue  # a near-miss on a quiet window is not worth a story

            tags = rule_tags(row)
            tag = tags[0] if tags else (llm_tag(row, provider) if use_llm else None)
            tag = tag or "UNEXPLAINED"

            conn.execute(
                "UPDATE predictions SET post_mortem_tag = ? WHERE prediction_id = ?",
                (tag, row["prediction_id"]),
            )
            counts[tag] = counts.get(tag, 0) + 1
            tagged += 1

    return {"candidates": len(rows), "tagged": tagged, "by_tag": counts}


def tag_summary(ticker: str | None = None, db_path: Path | str = DB_PATH) -> dict:
    """What has gone wrong, and how often -- the input to fixing it."""
    query = "SELECT post_mortem_tag, COUNT(*) AS n FROM predictions WHERE post_mortem_tag IS NOT NULL"
    params: tuple = ()
    if ticker:
        query += " AND ticker = ?"
        params = (ticker,)
    query += " GROUP BY post_mortem_tag ORDER BY n DESC"

    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return {row["post_mortem_tag"]: int(row["n"]) for row in rows}
