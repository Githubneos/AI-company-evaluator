"""FastAPI service (spec 7).

    uvicorn evaluator.serving.app:app --reload

Serving reuses `build_dataset`, the same code path training uses, so the
features behind a prediction are computed identically to the features the model
was fitted on. Reimplementing them here for speed is the usual origin of
train/serve skew (spec 2.5).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from evaluator.data.universe import cik_for, load_universe
from evaluator.feedback.postmortem import tag_summary
from evaluator.feedback.store import feedback_context, log_prediction, resolve_pending
from evaluator.fusion import build_payload
from evaluator.llm.evaluator import evaluate as llm_evaluate
from evaluator.llm.provider import LLMUnavailable
from evaluator.model.ablations import load_ablations
from evaluator.model.baselines import load_baselines
from evaluator.model.predict import ModelNotTrained, available_targets, load_model, score_ticker
from evaluator.monitoring import system_report

log = logging.getLogger(__name__)

app = FastAPI(
    title="AI Company Evaluator",
    description=(
        "Forward price-movement risk scoring from market data, SEC events, and "
        "news sentiment. Research tooling, not investment advice."
    ),
    version="0.2.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Make browsers revalidate dashboard assets (cheap 304s) so edits show up on reload."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# Must match the lookback `score_ticker` uses, so the price chart reads the same
# cached bars scoring already fetched instead of writing a second cache file.
PRICE_LOOKBACK_START = "2015-01-01"


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """The browser dashboard. Everything it shows comes from the JSON endpoints below."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "trained_targets": available_targets()}


@app.get("/universe")
def universe() -> dict:
    frame = load_universe()
    return {
        "count": int(len(frame)),
        "sectors": frame["sector"].value_counts().to_dict(),
        "tickers": frame[["ticker", "name", "sector"]].to_dict(orient="records"),
        "caveat": (
            "Today's S&P 500 constituents. Survivorship-biased: firms that "
            "failed or were removed are absent, so downside frequencies are a floor."
        ),
    }


@app.get("/prices/{ticker}")
def prices(ticker: str, days: int = Query(260, ge=5, le=2520)) -> dict:
    """Recent daily closes, for charting. Not an input to any model."""
    from evaluator.data.sources import load_prices

    ticker = ticker.upper()
    try:
        frame = load_prices(ticker, PRICE_LOOKBACK_START)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"price fetch failed: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"no price history for {ticker}")
    close = frame["close"].dropna().iloc[-days:]
    return {
        "ticker": ticker,
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "close": [round(float(v), 4) for v in close],
    }


@app.get("/score/{ticker}")
def score(ticker: str, lookback_start: str = "2015-01-01") -> dict:
    """Raw model output across every trained target."""
    try:
        result = score_ticker(ticker.upper(), lookback_start=lookback_start)
    except ModelNotTrained as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Drop the frames used internally; they are not JSON-serialisable payload.
    return {k: v for k, v in result.items() if k not in ("features_row", "prices")}


@app.get("/payload/{ticker}")
def payload(
    ticker: str,
    lookback_start: str = "2015-01-01",
    sentiment: bool = True,
    analogs: bool = True,
) -> dict:
    """The fusion payload the reasoning layer would see. No LLM call, no cost."""
    try:
        return build_payload(
            ticker.upper(),
            lookback_start=lookback_start,
            with_sentiment=sentiment,
            with_analogs=analogs,
        )
    except ModelNotTrained as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sentiment/{ticker}")
def sentiment_only(ticker: str) -> dict:
    """News sentiment alone, including the staleness flag."""
    from evaluator.sentiment.aggregate import score_ticker_sentiment

    ticker = ticker.upper()
    try:
        return score_ticker_sentiment(ticker, cik_for(ticker)).as_payload()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"sentiment pipeline failed: {exc}") from exc


@app.get("/analogs/{ticker}")
def analogs_only(ticker: str, k: int = Query(5, ge=1, le=25)) -> dict:
    """Closest historical situations and what followed them."""
    try:
        result = build_payload(
            ticker.upper(), with_sentiment=False, with_analogs=True, analog_k=k
        )
    except ModelNotTrained as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result["historical_analogs"]


@app.get("/evaluate/{ticker}")
def evaluate(
    ticker: str,
    lookback_start: str = "2015-01-01",
    log_prediction_row: bool = True,
    sentiment: bool = True,
    analogs: bool = True,
) -> dict:
    """Full pipeline: features -> models -> fusion -> written evaluation.

    The prediction is logged *before* the LLM call, so the record exists whether
    or not the reasoning layer succeeds. An unlogged prediction cannot be scored
    later, and that is the one loss this system cannot recover from.
    """
    ticker = ticker.upper()
    try:
        fusion_payload = build_payload(
            ticker, lookback_start=lookback_start,
            with_sentiment=sentiment, with_analogs=analogs,
        )
    except ModelNotTrained as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction_id = None
    if log_prediction_row:
        try:
            prediction_id = log_prediction(_prediction_row(fusion_payload))
        except Exception as exc:  # noqa: BLE001 - logging must not block the response
            log.exception("failed to log prediction for %s: %s", ticker, exc)

    try:
        result = llm_evaluate(fusion_payload)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=502, detail=f"reasoning layer unavailable: {exc}") from exc

    return {"prediction_id": prediction_id, "payload": fusion_payload, **result}


def _prediction_row(payload: dict) -> dict:
    """Flatten the fusion payload into the prediction-log schema."""
    from evaluator.config import DIRECTION, TargetSpec

    primary_name = TargetSpec(payload["horizon_days"], DIRECTION).name
    targets = payload["gbm"]["targets"]
    primary = targets.get(primary_name) or next(iter(targets.values()))
    sentiment = payload.get("sentiment", {})

    return {
        "ticker": payload["ticker"],
        "as_of": payload["as_of"],
        "horizon_days": int(primary["horizon_days"]),
        "threshold_sigmas": float(payload["threshold_sigmas"]),
        "label_scale": payload["label_scale"],
        "last_close": payload["last_close"],
        "probabilities": primary["probabilities"],
        "predicted_class": primary["predicted_class"],
        "sentiment_score": sentiment.get("sentiment_score") if sentiment.get("available") else None,
        "top_features": primary.get("top_features", []),
    }


@app.get("/model/{target}/validation")
def validation(target: str) -> dict:
    try:
        metadata = load_model(target).metadata
    except ModelNotTrained as exc:
        raise _not_found(exc) from exc
    return {
        "target": metadata["target"],
        "schema": metadata["schema"],
        "train_start": metadata["train_start"],
        "train_end": metadata["train_end"],
        "train_rows": metadata["train_rows"],
        "train_tickers": metadata["train_tickers"],
        "validation": metadata["validation"],
        # Skill over simple volatility / earnings-cycle signals on the same
        # out-of-sample rows; null until scripts.evaluate_baselines has run.
        "baselines": load_baselines(target),
        # Edge of each feature group over volatility alone; null until
        # scripts.evaluate_ablations has run.
        "ablations": load_ablations(target),
        "caveats": metadata["data_caveats"],
    }


@app.get("/feedback/{ticker}")
def feedback(ticker: str) -> dict:
    return feedback_context(ticker.upper())


@app.post("/feedback/resolve")
def resolve() -> dict:
    """Close out predictions whose forward window has elapsed."""
    return {"resolved": resolve_pending()}


@app.get("/monitoring")
def monitoring() -> dict:
    """Feedback-loop health, prediction distribution, and post-mortem tags."""
    report = system_report()
    report["post_mortem_tags"] = tag_summary()
    return report
