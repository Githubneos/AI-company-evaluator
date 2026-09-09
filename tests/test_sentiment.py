"""Sentiment aggregation tests (spec 4.3, 4.4).

FinBERT is stubbed throughout: these cover the aggregation logic, which is where
the design decisions live. Nothing here validates FinBERT's own accuracy, which
would need a labelled financial-news set.

The property that matters most is that a dead feed produces `available: false`
rather than a confident neutral reading. A broken news API that silently reports
"sentiment 0.0, all quiet" is worse than one that reports nothing, because the
reasoning layer downstream will faithfully explain the calm.
"""

from datetime import datetime, timedelta, timezone

import pytest

from evaluator.sentiment import aggregate as agg
from evaluator.sentiment.aggregate import aggregate, detect_divergence
from evaluator.sentiment.finbert import Score
from evaluator.sentiment.news import Article, NewsBatch


def now() -> datetime:
    return datetime.now(timezone.utc)


def article(title="Company beats expectations", hours_ago=1.0, source="Reuters", kind="news"):
    return Article(
        title=title,
        summary="",
        published=now() - timedelta(hours=hours_ago),
        source=source,
        kind=kind,
    )


def batch(articles, stale=False, age=1.0) -> NewsBatch:
    return NewsBatch(
        ticker="TEST",
        articles=articles,
        stale=stale,
        newest_age_hours=age,
        sources_ok={"yfinance": True},
    )


@pytest.fixture
def stub_scores(monkeypatch):
    """Map each text to a fixed score by keyword."""

    def fake(texts, **kwargs):
        out = []
        for text in texts:
            lowered = text.lower()
            if "beats" in lowered or "surge" in lowered:
                out.append(Score(positive=0.9, negative=0.05, neutral=0.05))
            elif "misses" in lowered or "plunge" in lowered:
                out.append(Score(positive=0.05, negative=0.9, neutral=0.05))
            else:
                out.append(Score(positive=0.1, negative=0.1, neutral=0.8))
        return out

    monkeypatch.setattr(agg, "score_texts", fake)
    return fake


def test_empty_feed_is_unavailable_not_neutral(stub_scores):
    payload = aggregate(batch([])).as_payload()
    assert payload["available"] is False
    assert "no articles" in payload["reason"].lower()


def test_stale_feed_is_unavailable(stub_scores):
    result = aggregate(batch([article()], stale=True, age=100.0))
    assert result.as_payload()["available"] is False


def test_positive_and_negative_headlines_move_the_score(stub_scores):
    positive = aggregate(batch([article("Company beats expectations")])).score
    negative = aggregate(batch([article("Company misses expectations")])).score
    assert positive > 0.5
    assert negative < -0.5


def test_recent_news_outweighs_old_news(stub_scores):
    """An 8-hour half-life means yesterday's story barely counts against today's."""
    mixed = batch(
        [
            article("Company beats expectations", hours_ago=0.5),
            article("Company misses expectations", hours_ago=48.0),
        ]
    )
    assert aggregate(mixed).score > 0.4


def test_credible_sources_outweigh_low_tier_ones(stub_scores):
    mixed = batch(
        [
            article("Company beats expectations", source="Reuters"),
            article("Company misses expectations", source="Motley Fool"),
        ]
    )
    assert aggregate(mixed).score > 0


def test_filings_carry_the_highest_credibility():
    assert article(source="SEC EDGAR").credibility == 1.0
    assert article(source="Reuters").credibility > article(source="Motley Fool").credibility


def test_unknown_source_is_not_given_the_benefit_of_the_doubt():
    assert article(source="Some Blog").credibility < article(source="CNBC").credibility


def test_disagreement_lowers_confidence(stub_scores):
    agreeing = batch([article("Company beats expectations") for _ in range(4)])
    conflicting = batch(
        [article("Company beats expectations"), article("Company misses expectations")] * 2
    )
    assert aggregate(agreeing).confidence > aggregate(conflicting).confidence


def test_more_corroborating_sources_raises_confidence(stub_scores):
    one = aggregate(batch([article("Company beats expectations", source="Reuters")]))
    many = aggregate(
        batch(
            [
                article("Company beats expectations", source=source)
                for source in ("Reuters", "Bloomberg", "Associated Press", "CNBC")
            ]
        )
    )
    assert many.confidence > one.confidence


def test_repeated_headlines_from_one_source_do_not_create_false_corroboration(stub_scores):
    one = aggregate(batch([article("Company beats expectations", source="Reuters")]))
    repeated = aggregate(
        batch([article("Company beats expectations", source="Reuters") for _ in range(6)])
    )
    assert repeated.confidence == pytest.approx(one.confidence)


def test_neutral_coverage_barely_moves_the_score(stub_scores):
    result = aggregate(batch([article("Company holds annual meeting") for _ in range(3)]))
    assert abs(result.score) < 0.1


class TestDivergence:
    def test_unavailable_without_sentiment(self):
        result = detect_divergence({"available": False}, 0.02, 0.01)
        assert result["available"] is False

    def test_negative_news_with_price_holding_up_is_flagged(self):
        result = detect_divergence({"available": True, "sentiment_score": -0.6}, 0.02, 0.01)
        assert result["divergence_flag"] is True
        assert "holding up" in result["interpretation"]

    def test_positive_news_with_price_falling_is_flagged(self):
        result = detect_divergence({"available": True, "sentiment_score": 0.6}, -0.02, 0.01)
        assert result["divergence_flag"] is True

    def test_agreement_is_not_flagged(self):
        result = detect_divergence({"available": True, "sentiment_score": 0.6}, 0.02, 0.01)
        assert result["divergence_flag"] is False

    def test_move_is_measured_in_volatility_units(self):
        """A 2% move is large for a utility and noise for a biotech."""
        quiet = detect_divergence({"available": True, "sentiment_score": -0.6}, 0.02, 0.005)
        volatile = detect_divergence({"available": True, "sentiment_score": -0.6}, 0.02, 0.10)
        assert quiet["divergence_flag"] is True
        assert volatile["divergence_flag"] is False

    def test_missing_price_history_is_reported(self):
        assert detect_divergence({"available": True, "sentiment_score": 0.5}, None, None)[
            "available"
        ] is False
