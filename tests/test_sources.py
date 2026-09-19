"""Live-cache and filing-refresh behaviour (specs 4.2 and 7)."""

import pandas as pd

from evaluator.data import sources
from evaluator.data.edgar import EdgarClient


def _bars(start: str) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=2, name="date")
    return pd.DataFrame(
        {"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [1, 2]},
        index=index,
    )


def test_live_cache_fetches_only_the_missing_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
    path = sources._cache_path("TEST", "2024-01-01", None)
    cached = _bars("2024-01-01")
    cached.to_parquet(path)
    calls = []

    class FakeYF:
        @staticmethod
        def download(ticker, **kwargs):
            calls.append(kwargs["start"])
            return _bars(kwargs["start"])

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    monkeypatch.setattr(sources, "_expected_latest_bar", lambda: pd.Timestamp("2024-01-05"))

    result = sources.load_prices("TEST", "2024-01-01")
    assert calls == ["2024-01-03"]
    assert result.index.max() == pd.Timestamp("2024-01-04")


def test_unreadable_price_cache_is_refetched_and_rewritten_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
    path = sources._cache_path("TEST", "2024-01-01", None)
    # What two interleaved in-place writers left behind in production.
    path.write_bytes(b"PAR1 half-written garbage")
    calls = []

    class FakeYF:
        @staticmethod
        def download(ticker, **kwargs):
            calls.append(kwargs["start"])
            return _bars(kwargs["start"])

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)

    result = sources.load_prices("TEST", "2024-01-01")

    assert calls == ["2024-01-01"]  # full refetch, not a crash
    assert len(result) == 2
    pd.testing.assert_frame_equal(pd.read_parquet(path), result, check_freq=False)
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp files left


def test_edgar_refresh_bypasses_an_existing_cache(tmp_path, monkeypatch):
    client = EdgarClient(cache_dir=tmp_path)
    cached = pd.DataFrame(
        columns=["cik", "ticker", "form", "filing_date", "acceptance_datetime", "report_date", "items", "accession"]
    )
    cached.to_parquet(tmp_path / "TEST.parquet")
    monkeypatch.setattr(client, "_get_json", lambda url: None)
    assert client.filings(1, "TEST", use_cache=True, refresh=False).empty
    # refresh=True attempts the provider rather than silently treating the old
    # cache as fresh; this mocked request returns no data, so the result is empty.
    assert client.filings(1, "TEST", use_cache=True, refresh=True).empty
