"""Per-ticker sentiment aggregation (spec 4.3, 4.4).

Each article's weight is the product of three factors:

  credibility  Source tier. A wire service and an SEO content farm should not
               count equally.
  recency      Exponential decay, ~8h half-life. News-driven signal decays fast;
               a headline from three days ago is not describing today.
  conviction   FinBERT's own distance from neutral. A hedged headline the model
               is unsure about should not move the aggregate much.

Confidence is deliberately *not* the mean of article confidences. It rises with
independent corroboration and falls when sources disagree, because five outlets
agreeing is a stronger signal than one outlet being certain -- and disagreement
is genuine uncertainty rather than something to average away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC

from evaluator.sentiment.finbert import Score, score_texts
from evaluator.sentiment.news import Article, NewsBatch

RECENCY_HALF_LIFE_HOURS = 8.0
MAX_HEADLINES = 8


@dataclass
class SentimentResult:
    score: float
    confidence: float
    article_count: int
    stale: bool
    newest_age_hours: float | None
    top_headlines: list[dict]
    sources_ok: dict[str, bool]
    reason: str | None = None

    def as_payload(self) -> dict:
        """Fusion-layer shape. `available` is false when there is nothing to say."""
        if self.article_count == 0 or self.stale:
            return {
                "available": False,
                "reason": self.reason
                or (
                    "News feed returned no recent articles; sentiment would be "
                    "indistinguishable from a broken feed."
                ),
                "article_count": self.article_count,
                "stale": self.stale,
                "newest_age_hours": self.newest_age_hours,
                "sources_ok": self.sources_ok,
            }
        return {
            "available": True,
            "sentiment_score": round(self.score, 4),
            "sentiment_confidence": round(self.confidence, 4),
            "article_count": self.article_count,
            "newest_age_hours": self.newest_age_hours,
            "top_headlines": self.top_headlines,
            "sources_ok": self.sources_ok,
        }


def _recency_weight(age_hours: float) -> float:
    return 0.5 ** (max(age_hours, 0.0) / RECENCY_HALF_LIFE_HOURS)


def _age_hours(article: Article, now) -> float:
    return (now - article.published).total_seconds() / 3600.0


def aggregate(batch: NewsBatch, *, use_cache: bool = True) -> SentimentResult:
    from datetime import datetime

    now = datetime.now(UTC)

    if not batch.articles:
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            article_count=0,
            stale=True,
            newest_age_hours=batch.newest_age_hours,
            top_headlines=[],
            sources_ok=batch.sources_ok,
            reason="No articles returned by any configured news source.",
        )

    scores: list[Score] = score_texts([a.text for a in batch.articles], use_cache=use_cache)

    weighted_sum = 0.0
    total_weight = 0.0
    scored: list[tuple[Article, Score, float]] = []

    for article, score in zip(batch.articles, scores):
        weight = (
            article.credibility
            * _recency_weight(_age_hours(article, now))
            * max(score.confidence, 0.05)
        )
        weighted_sum += weight * score.polarity
        total_weight += weight
        scored.append((article, score, weight))

    # Multiple syndicated headlines from Reuters are not multiple independent
    # confirmations.  Keep the strongest evidence per publisher for aggregation
    # and confidence; different sources agreeing are what should raise
    # confidence under spec 4.3.
    by_source: dict[str, tuple[Article, Score, float]] = {}
    for item in scored:
        key = item[0].source.strip().casefold() or "unknown"
        if key not in by_source or item[2] > by_source[key][2]:
            by_source[key] = item
    independent = list(by_source.values())
    weighted_sum = sum(weight * score.polarity for _, score, weight in independent)
    total_weight = sum(weight for _, _, weight in independent)
    sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Confidence: more weight and more agreement both raise it.
    polarities = [s.polarity for _, s, _ in independent]
    mean_polarity = sum(polarities) / len(polarities)
    variance = sum((p - mean_polarity) ** 2 for p in polarities) / len(polarities)
    agreement = 1.0 / (1.0 + variance)
    volume = 1.0 - math.exp(-total_weight)
    confidence = max(0.0, min(1.0, agreement * volume))

    scored.sort(key=lambda item: item[2], reverse=True)
    headlines = [
        {
            "title": article.title[:200],
            "source": article.source or "unknown",
            "kind": article.kind,
            "published": article.published.isoformat(),
            "polarity": round(score.polarity, 3),
        }
        for article, score, _ in scored[:MAX_HEADLINES]
    ]

    return SentimentResult(
        score=sentiment,
        confidence=confidence,
        article_count=len(batch.articles),
        stale=batch.stale,
        newest_age_hours=batch.newest_age_hours,
        top_headlines=headlines,
        sources_ok=batch.sources_ok,
    )


def score_ticker_sentiment(ticker: str, cik: int | None = None) -> SentimentResult:
    from evaluator.sentiment.news import collect_news

    return aggregate(collect_news(ticker, cik))


def detect_divergence(sentiment: dict, recent_return: float | None, vol: float | None) -> dict:
    """Sentiment-vs-price divergence (spec 4.4).

    Flags the case where the tape and the coverage disagree: strongly negative
    news with the price holding up, or the reverse. The move is measured in
    volatility units so "flat" means flat relative to how much this name
    normally moves.
    """
    if not sentiment.get("available"):
        return {
            "available": False,
            "reason": "Divergence detection needs a sentiment score; none is available.",
        }
    if recent_return is None or vol is None or vol <= 0:
        return {"available": False, "reason": "Insufficient price history to measure divergence."}

    score = sentiment["sentiment_score"]
    move_sigmas = recent_return / vol

    diverging = (score <= -0.25 and move_sigmas >= 0.5) or (score >= 0.25 and move_sigmas <= -0.5)
    if diverging:
        direction = "negative news, price holding up" if score < 0 else "positive news, price falling"
    else:
        direction = "sentiment and price action broadly agree"

    return {
        "available": True,
        "divergence_flag": bool(diverging),
        "sentiment_score": round(score, 4),
        "recent_move_sigmas": round(float(move_sigmas), 3),
        "interpretation": direction,
    }
