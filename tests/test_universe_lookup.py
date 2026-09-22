"""Universe lookups tolerate the way people actually type tickers."""

from evaluator.data.universe import cik_for, tickers


def test_cik_lookup_ignores_case_and_whitespace():
    ticker = tickers()[0]

    expected = cik_for(ticker)
    assert expected is not None
    assert cik_for(ticker.lower()) == expected
    assert cik_for(f"  {ticker.lower()} ") == expected


def test_cik_lookup_of_an_unknown_ticker_is_none():
    assert cik_for("NOT-A-TICKER") is None
