"""SEC EDGAR filing metadata (spec 1.1, 1.3).

Free, no key, and the only event source in this build with genuine historical
depth. We take filing *metadata* only -- form type, dates, and 8-K item codes --
never document text. The item code is itself a structured label assigned by the
filer under SEC rules, so the event taxonomy needs no parsing and no inference.

TIMING, WHICH IS THE WHOLE GAME
-------------------------------
`acceptanceDateTime` is captured alongside `filingDate` because they are not
interchangeable for modelling. A filing accepted at 16:35 is public *after* the
close, so the first trading session that could react to it is the next one.
Treating it as known on its filing date leaks several hours of hindsight into
every earnings event -- which, given that earnings are the largest scheduled
source of large moves, would be enough on its own to fake a skilful model.
`evaluator/events.py` applies the shift; this module only records the facts.

SEC requires a descriptive User-Agent with contact info and throttles at 10
requests/second. Both are honoured here; ignoring either gets the client blocked.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from evaluator.config import CACHE_DIR

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"
DEFAULT_USER_AGENT = "AI-company-evaluator research karumudikeerthan@gmail.com"

FILING_COLUMNS = [
    "cik",
    "ticker",
    "form",
    "filing_date",
    "acceptance_datetime",
    "report_date",
    "items",
    "accession",
]


@dataclass
class EdgarClient:
    """Rate-limited EDGAR reader.

    SEC's published ceiling is 10 requests/second; `min_interval` defaults below
    that deliberately. Being throttled costs far more time than being polite.
    """

    user_agent: str = DEFAULT_USER_AGENT
    min_interval: float = 0.12
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR / "edgar")
    _last_request: float = field(default=0.0, init=False, repr=False)

    def _get_json(self, url: str) -> dict | None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if exc.code == 404:
                log.warning("EDGAR 404 for %s", url)
                return None
            raise
        except (URLError, TimeoutError) as exc:
            log.warning("EDGAR request failed for %s: %s", url, exc)
            return None
        finally:
            self._last_request = time.monotonic()

    def filings(
        self, cik: int, ticker: str, *, use_cache: bool = True, refresh: bool = False
    ) -> pd.DataFrame:
        """All filings for one company, recent chunk plus paginated history."""
        cache_path = self.cache_dir / f"{ticker}.parquet"
        if use_cache and cache_path.exists() and not refresh:
            return pd.read_parquet(cache_path)

        payload = self._get_json(SUBMISSIONS_URL.format(cik=cik))
        if payload is None:
            return pd.DataFrame(columns=FILING_COLUMNS)

        blocks = [payload.get("filings", {}).get("recent", {})]

        # Companies with long histories page their older filings into extra files.
        for extra in payload.get("filings", {}).get("files", []):
            older = self._get_json(ARCHIVE_URL.format(name=extra["name"]))
            if older:
                blocks.append(older)

        frame = pd.concat(
            [self._block_to_frame(b, cik, ticker) for b in blocks if b],
            ignore_index=True,
        )
        if frame.empty:
            return pd.DataFrame(columns=FILING_COLUMNS)

        frame = (
            frame.drop_duplicates(subset=["accession"])
            .sort_values("filing_date")
            .reset_index(drop=True)
        )

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path)
        return frame

    @staticmethod
    def _block_to_frame(block: dict, cik: int, ticker: str) -> pd.DataFrame:
        forms = block.get("form", [])
        if not forms:
            return pd.DataFrame(columns=FILING_COLUMNS)

        n = len(forms)

        def column(key: str) -> list:
            values = block.get(key, [])
            return list(values) + [None] * (n - len(values))

        return pd.DataFrame(
            {
                "cik": cik,
                "ticker": ticker,
                "form": forms,
                "filing_date": pd.to_datetime(column("filingDate"), errors="coerce"),
                "acceptance_datetime": pd.to_datetime(
                    column("acceptanceDateTime"), errors="coerce", utc=True
                ),
                "report_date": pd.to_datetime(column("reportDate"), errors="coerce"),
                "items": column("items"),
                "accession": column("accessionNumber"),
            }
        )


def load_filings(
    tickers: list[str],
    ciks: dict[str, int],
    *,
    client: EdgarClient | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Filings for many companies, concatenated. Failures are skipped, not fatal."""
    client = client or EdgarClient()
    frames = []
    for i, ticker in enumerate(tickers, start=1):
        cik = ciks.get(ticker)
        if cik is None:
            continue
        try:
            frames.append(client.filings(cik, ticker, use_cache=use_cache))
        except Exception as exc:  # noqa: BLE001 - one bad company must not stop the backfill
            log.warning("filings failed for %s: %s", ticker, exc)
        if i % 50 == 0:
            log.info("fetched filings for %d/%d companies", i, len(tickers))

    if not frames:
        return pd.DataFrame(columns=FILING_COLUMNS)
    return pd.concat(frames, ignore_index=True)
