"""News and filing feeds for the sentiment pipeline (spec 4.2).

Two sources with very different reliability, kept distinct rather than pooled:

  yfinance headlines  Broad but noisy, variable provider quality, and the free
                      tier returns only ~10 recent items.
  EDGAR 8-K feed      Narrow but authoritative. A filing is a fact, not a
                      report about a fact, so it carries the highest credibility
                      weight in aggregation.

STALENESS IS AN OUTPUT, NOT AN EXCEPTION
----------------------------------------
Spec 8.2 is explicit that a dead news feed must not fail silently. If the feed
returns nothing, the pipeline reports `stale=True` with the age of the newest
item it did find, and the fusion layer passes that to the reasoning layer. The
failure mode being designed against is a broken API producing a confident
"neutral sentiment" reading forever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger(__name__)

STALE_AFTER_HOURS = 48

#: Source credibility tiers (spec 4.3). Wire services and filings outrank
#: aggregators and blogs; unknown providers get the low tier rather than the
#: benefit of the doubt.
SOURCE_TIERS = {
    "sec edgar": 1.0,
    "reuters": 0.95,
    "bloomberg": 0.95,
    "associated press": 0.9,
    "the wall street journal": 0.9,
    "financial times": 0.9,
    "cnbc": 0.8,
    "barron's": 0.8,
    "marketwatch": 0.7,
    "yahoo finance": 0.6,
    "benzinga": 0.6,
    "investor's business daily": 0.6,
    "motley fool": 0.4,
    "simply wall st.": 0.4,
    "zacks": 0.4,
}
DEFAULT_TIER = 0.3


@dataclass
class Article:
    title: str
    summary: str
    published: datetime
    source: str
    url: str = ""
    kind: str = "news"

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()

    @property
    def credibility(self) -> float:
        return SOURCE_TIERS.get(self.source.strip().lower(), DEFAULT_TIER)


@dataclass
class NewsBatch:
    ticker: str
    articles: list[Article] = field(default_factory=list)
    stale: bool = False
    newest_age_hours: float | None = None
    sources_ok: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "article_count": len(self.articles),
            "stale": self.stale,
            "newest_age_hours": self.newest_age_hours,
            "sources_ok": self.sources_ok,
        }


def _parse_time(value) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    return stamp.to_pydatetime()


def fetch_yfinance_news(ticker: str) -> list[Article]:
    """Recent headlines. yfinance nests the payload under `content`."""
    import yfinance as yf

    raw = yf.Ticker(ticker).news or []
    articles = []
    for item in raw:
        content = item.get("content", item)
        published = _parse_time(content.get("pubDate") or content.get("displayTime"))
        if published is None:
            continue
        provider = (content.get("provider") or {}).get("displayName", "") or ""
        articles.append(
            Article(
                title=content.get("title", "") or "",
                summary=content.get("summary") or content.get("description") or "",
                published=published,
                source=provider,
                url=((content.get("canonicalUrl") or {}) or {}).get("url", "") or "",
                kind="news",
            )
        )
    return [a for a in articles if a.title]


def fetch_recent_filings(ticker: str, cik: int, *, days: int = 14) -> list[Article]:
    """Recent 8-K filings as first-class sentiment inputs (spec 4.2).

    Rendered as human-readable text so the same sentiment model can score them,
    with the event taxonomy name carrying the actual information.
    """
    from evaluator.data.edgar import EdgarClient
    from evaluator.events import classify_item, parse_items

    # Sentiment needs the filing feed to be live.  Reusing a permanent EDGAR
    # cache here made scheduled polling miss every filing issued after its first
    # run while still reporting the source as healthy.
    filings = EdgarClient().filings(cik, ticker, use_cache=True, refresh=True)
    if filings.empty:
        return []

    cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=days)
    recent = filings[
        (filings["form"] == "8-K")
        & (pd.to_datetime(filings["filing_date"]).dt.tz_localize("UTC") >= cutoff)
    ]

    articles = []
    for row in recent.itertuples(index=False):
        for code in parse_items(row.items):
            event_type, subtype = classify_item(code)
            published = _parse_time(row.acceptance_datetime) or _parse_time(row.filing_date)
            if published is None:
                continue
            articles.append(
                Article(
                    title=f"{ticker} filed an 8-K reporting {event_type.replace('_', ' ').lower()}",
                    summary=f"SEC Form 8-K, item {code} ({subtype.replace('_', ' ').lower()}).",
                    published=published,
                    source="SEC EDGAR",
                    kind="filing",
                )
            )
    return articles


def collect_news(ticker: str, cik: int | None = None) -> NewsBatch:
    """Gather every feed, recording which ones answered."""
    batch = NewsBatch(ticker=ticker)

    try:
        batch.articles.extend(fetch_yfinance_news(ticker))
        batch.sources_ok["yfinance"] = True
    except Exception as exc:  # noqa: BLE001 - a dead feed is a reported state
        log.warning("yfinance news failed for %s: %s", ticker, exc)
        batch.sources_ok["yfinance"] = False

    if cik is not None:
        try:
            batch.articles.extend(fetch_recent_filings(ticker, cik))
            batch.sources_ok["edgar"] = True
        except Exception as exc:  # noqa: BLE001
            log.warning("EDGAR feed failed for %s: %s", ticker, exc)
            batch.sources_ok["edgar"] = False

    now = datetime.now(timezone.utc)
    if batch.articles:
        newest = max(a.published for a in batch.articles)
        batch.newest_age_hours = round((now - newest).total_seconds() / 3600, 2)
        batch.stale = now - newest > timedelta(hours=STALE_AFTER_HOURS)
    else:
        batch.stale = True

    return batch
