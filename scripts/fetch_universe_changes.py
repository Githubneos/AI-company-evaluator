"""CLI: fetch the S&P 500 change history and check it in (spec 1.2).

    python -m scripts.fetch_universe_changes --apply

Wikipedia's "Historical components of the S&P 500" lists every addition and
removal with an effective date and a reason. That history is what turns a
survivorship-biased snapshot of today's index into membership intervals: which
names were in the index, and when.

The result is written to data/universe/sp500_changes.csv and committed, with a
sidecar recording when it was fetched. A training run must not silently pick up
this month's index changes, exactly as with data/universe/sp500.csv.
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
from datetime import UTC, datetime

import pandas as pd

from evaluator.data.universe import CHANGES_META_PATH, CHANGES_PATH
from evaluator.io import atomic_write_json

log = logging.getLogger("fetch_universe_changes")

SOURCE_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
USER_AGENT = "AI-company-evaluator research karumudikeerthan@gmail.com"
COLUMNS = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]


def fetch_changes(url: str = SOURCE_URL) -> pd.DataFrame:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
        html = response.read().decode()
    return parse_changes(html)


def parse_changes(html: str) -> pd.DataFrame:
    """The widest table on the page: one row per index change."""
    tables = pd.read_html(io.StringIO(html))
    candidates = [t for t in tables if len(t.columns) >= 5 and len(t) > 50]
    if not candidates:
        raise RuntimeError("no change table found; the page layout has changed")
    frame = max(candidates, key=len).copy()
    frame.columns = [
        "_".join(str(part) for part in col).lower() if isinstance(col, tuple) else str(col).lower()
        for col in frame.columns
    ]

    def column(*needles: str) -> pd.Series:
        for name in frame.columns:
            if all(needle in name for needle in needles):
                return frame[name]
        raise RuntimeError(f"column {needles} missing; the page layout has changed")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(column("effective", "date"), errors="coerce"),
            "added_ticker": column("added", "ticker"),
            "added_name": column("added", "security"),
            "removed_ticker": column("removed", "ticker"),
            "removed_name": column("removed", "security"),
            "reason": column("reason"),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("added_ticker", "removed_ticker"):
        out[col] = out[col].astype("string").str.strip().str.upper().replace({"": pd.NA})
    return out[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write data/universe/sp500_changes.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    changes = fetch_changes()
    print(f"{len(changes):,} changes, {changes['date'].min().date()} -> {changes['date'].max().date()}")
    adds, drops = changes["added_ticker"].notna().sum(), changes["removed_ticker"].notna().sum()
    print(f"additions: {adds:,}   removals: {drops:,}")
    print(changes.tail(3).to_string(index=False))

    if args.apply:
        CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
        changes.to_csv(CHANGES_PATH, index=False, date_format="%Y-%m-%d")
        atomic_write_json(
            {
                "source": SOURCE_URL,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "rows": int(len(changes)),
                "first_change": str(changes["date"].min().date()),
                "last_change": str(changes["date"].max().date()),
            },
            CHANGES_META_PATH,
        )
        print(f"\nwrote {CHANGES_PATH}")
    else:
        print("\n(dry run -- pass --apply to write the CSV)")


if __name__ == "__main__":
    main()
