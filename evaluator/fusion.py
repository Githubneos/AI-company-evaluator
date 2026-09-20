"""Fusion layer (spec 5): assemble the structured payload the LLM evaluator reads.

Deliberately rules-based rather than a learned model. The spec offers a shallow
meta-model as an option and `evaluator/fusion_model.py` implements it, but it is
only preferred when it demonstrably beats the rules out of sample -- combining
signals is not automatically better than reporting them.

Two principles hold the design together:

Absent signals are represented explicitly, `available: false` with a reason,
never defaulted to zero. A sentiment score of 0.0 and a dead news feed are very
different states, and an LLM handed the first will reason about neutral coverage
that nobody measured.

Model quality is stated, not implied. `model_quality.interpretation` says in
plain language whether the model beats its base rate, because the reasoning
layer cannot be trusted to infer that from a skill score it has never been
calibrated against, and the honest answer changes the entire tone of the output.
"""

from __future__ import annotations

import logging

import numpy as np

from evaluator.data.universe import cik_for
from evaluator.feedback.store import feedback_context
from evaluator.model.analogs import load_index
from evaluator.model.predict import score_ticker

log = logging.getLogger(__name__)

SKILL_NOISE_FLOOR = 0.01


def _quality_verdict(skill: float | None, auc: float | None) -> str:
    """Plain-language read on whether a model beats guessing the base rate."""
    if skill is None:
        return "Model skill could not be measured."
    if skill <= 0:
        return (
            "This model does NOT beat predicting the historical base rates. Its "
            "probabilities carry no demonstrated information about the forward "
            "move and must not be presented as a signal."
        )
    if skill < SKILL_NOISE_FLOOR:
        return (
            "This model beats the base rate by a margin too small to be "
            "distinguished from noise on this sample. Treat it as no signal."
        )
    auc_text = f" Out-of-sample AUC {auc:.3f}." if auc else ""
    return (
        "This model beats base-rate prediction out of sample by a measurable "
        f"margin (Brier skill {skill:+.4f}).{auc_text}"
    )


def _model_quality(targets: dict) -> dict:
    """Per-target skill plus an overall verdict driven by the best model."""
    per_target = {}
    best_skill, best_name, best_auc = None, None, None

    for name, result in targets.items():
        skill = result["skill"].get("brier_skill")
        per_target[name] = {
            **result["skill"],
            "interpretation": _quality_verdict(skill, result["skill"].get("macro_auc")),
        }
        if skill is not None and (best_skill is None or skill > best_skill):
            best_skill, best_name, best_auc = skill, name, result["skill"].get("macro_auc")

    return {
        "per_target": per_target,
        "best_target": best_name,
        "best_brier_skill": best_skill,
        "interpretation": _quality_verdict(best_skill, best_auc),
        "any_target_has_skill": bool(best_skill is not None and best_skill > SKILL_NOISE_FLOOR),
    }


def _sentiment_block(ticker: str, enabled: bool) -> dict:
    if not enabled:
        return {
            "available": False,
            "reason": "Sentiment scoring was disabled for this request.",
        }
    try:
        from evaluator.sentiment.aggregate import score_ticker_sentiment

        return score_ticker_sentiment(ticker, cik_for(ticker)).as_payload()
    except Exception as exc:  # noqa: BLE001 - a dead feed is a reported state, not a 500
        log.warning("sentiment unavailable for %s: %s", ticker, exc)
        return {"available": False, "reason": f"Sentiment pipeline failed: {exc}"}


def _analog_block(scored: dict, enabled: bool, k: int = 5) -> dict:
    if not enabled:
        return {"available": False, "reason": "Analog retrieval was disabled for this request."}

    index = load_index()
    if index is None:
        return {
            "available": False,
            "reason": "No analog index has been built. Run: python -m scripts.build_analogs",
        }
    try:
        import pandas as pd

        matches = index.query(
            scored["features_row"],
            as_of=pd.Timestamp(scored["as_of"]),
            k=k,
            exclude_ticker=scored["ticker"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("analog retrieval failed: %s", exc)
        return {"available": False, "reason": f"Analog retrieval failed: {exc}"}

    if not matches:
        return {"available": False, "reason": "No sufficiently similar historical situations found."}
    return {
        "available": True,
        "note": (
            "Retrieved by feature similarity, not causal resemblance. These are "
            "what similar-looking situations did next, not what this one must do."
        ),
        "matches": matches,
    }


def _divergence_block(scored: dict, sentiment: dict) -> dict:
    from evaluator.sentiment.aggregate import detect_divergence

    prices = scored.get("prices")
    if prices is None or len(prices) < 25:
        return {"available": False, "reason": "Insufficient price history to measure divergence."}

    close = prices["close"]
    recent_return = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) > 6 else None
    vol = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(5))
    return detect_divergence(sentiment, recent_return, vol if vol > 0 else None)


def build_payload(
    ticker: str,
    *,
    lookback_start: str = "2015-01-01",
    use_cache: bool = True,
    with_sentiment: bool = True,
    with_analogs: bool = True,
    analog_k: int = 5,
) -> dict:
    """Everything the reasoning layer is allowed to see, and nothing it is not."""
    ticker = ticker.upper()
    scored = score_ticker(ticker, lookback_start=lookback_start, use_cache=use_cache)
    targets = scored["targets"]

    sentiment = _sentiment_block(ticker, with_sentiment)
    threshold_pct = abs(scored["move_threshold_pct"] or 0.0) * 100

    payload = {
        "ticker": ticker,
        "as_of": scored["as_of"],
        "last_close": scored["last_close"],
        "horizon_days": scored["horizon_days"],
        "threshold_sigmas": scored["threshold_sigmas"],
        "label_scale": scored["label_scale"],
        "question": (
            f"Probability of a forward move larger than "
            f"{scored['threshold_sigmas']:.1f} trailing sigma (about "
            f"{threshold_pct:.1f}%) over each modelled horizon. "
            "'direction' models predict drop/neutral/spike; 'magnitude' models "
            "predict only whether a large move occurs, in either direction; "
            "'rel_direction' models predict the move relative to the stock's "
            "sector ETF, with the sector's own move removed."
        ),
        "gbm": {"available": bool(targets), "targets": targets},
        "model_quality": _model_quality(targets),
        "sentiment": sentiment,
        "divergence": _divergence_block(scored, sentiment),
        "historical_analogs": _analog_block(scored, with_analogs, analog_k),
        "feedback_context": feedback_context(ticker),
        "data_caveats": scored["caveats"],
    }
    return payload
