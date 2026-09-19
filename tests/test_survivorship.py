"""Historical index membership, and which recovered names can be trusted."""

import pandas as pd
import pytest

import evaluator.data.delisted as delisted
import evaluator.data.universe as universe


@pytest.fixture
def history(monkeypatch):
    """A small index: one constant name, one added late, one removed, one that did both."""
    current = pd.DataFrame(
        {
            "ticker": ["OLD", "NEW", "BACK"],
            "name": ["Old Co", "New Co", "Back Co"],
            "sector": ["Industrials"] * 3,
            "date_added": pd.to_datetime(["1990-01-02", "2015-06-01", "2018-03-01"]),
            "cik": [1, 2, 3],
            "sector_etf": ["XLI"] * 3,
        }
    )
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-05-10", "2015-06-01", "2016-09-15", "2018-03-01"]),
            "added_ticker": pd.array([None, "NEW", None, "BACK"], dtype="string"),
            "added_name": [None, "New Co", None, "Back Co"],
            "removed_ticker": pd.array(["GONE", "DROPPED", "BACK", None], dtype="string"),
            "removed_name": ["Gone Co", "Dropped Co", "Back Co", None],
            "reason": [
                "Acquired by Old Co.",
                "Market capitalization changes.",
                "Market capitalization changes.",
                "Re-added after index reshuffle.",
            ],
        }
    )
    # Replace the loaders outright: patching through an lru_cache wrapper would
    # leave the real data in place, silently testing the live universe.
    for cached in (universe.load_universe, universe.load_changes, universe.membership_intervals):
        cached.cache_clear()
    monkeypatch.setattr(universe, "load_universe", lambda: current)
    monkeypatch.setattr(universe, "load_changes", lambda: changes)
    universe.membership_intervals.cache_clear()
    yield
    universe.membership_intervals.cache_clear()


def test_intervals_reconstruct_when_each_name_was_a_member(history):
    intervals = universe.membership_intervals()

    assert intervals["OLD"] == [(pd.Timestamp("1990-01-02"), None)]
    assert intervals["NEW"] == [(pd.Timestamp("2015-06-01"), None)]
    assert intervals["GONE"] == [(pd.Timestamp.min, pd.Timestamp("2010-05-10"))]
    assert intervals["DROPPED"] == [(pd.Timestamp.min, pd.Timestamp("2015-06-01"))]
    # Removed in 2016, re-added in 2018: two spells, no overlap.
    assert intervals["BACK"] == [(pd.Timestamp.min, pd.Timestamp("2016-09-15")), (pd.Timestamp("2018-03-01"), None)]


def test_membership_mask_excludes_before_joining_and_after_leaving(history):
    dates = pd.to_datetime(["2014-01-02", "2015-07-01", "2017-01-03", "2019-01-02"])

    assert list(universe.was_member("NEW", dates)) == [False, True, True, True]
    assert list(universe.was_member("GONE", dates)) == [False, False, False, False]
    assert list(universe.was_member("BACK", dates)) == [True, True, False, True]
    assert list(universe.was_member("NEVER_HEARD_OF_IT", dates)) == [False] * 4


def test_removed_tickers_lists_only_names_gone_from_the_index(history):
    removed = universe.removed_tickers()

    assert set(removed) == {"GONE", "DROPPED"}  # BACK is in the index again
    assert removed["GONE"][0] == pd.Timestamp("2010-05-10")
    assert universe.is_delisting(removed["GONE"][1]) and not universe.is_delisting(removed["DROPPED"][1])


def test_delisting_words_separate_acquisitions_from_reshuffles():
    assert universe.is_delisting("Acquired by Foo Inc.")
    assert universe.is_delisting("Bar filed for bankruptcy and was delisted.")
    assert universe.is_delisting("Taken private by a consortium.")
    assert not universe.is_delisting("Market capitalization changes.")
    assert not universe.is_delisting("")


def _prices(start, end) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    return pd.DataFrame({"close": 100.0}, index=index)


def test_a_symbol_still_trading_long_after_a_delisting_is_rejected(history):
    window = (pd.Timestamp("2005-01-03"), pd.Timestamp("2010-05-10"))
    # The real company's tape stops at the acquisition...
    genuine = delisted.classify("GONE", _prices("2005-01-03", "2010-05-07"), window[1], "Acquired by Old Co.", window)
    # ...so a series running to today belongs to whoever has the symbol now.
    recycled = delisted.classify("GONE", _prices("2005-01-03", "2026-01-02"), window[1], "Acquired by Old Co.", window)

    assert genuine["status"] == delisted.OK
    assert recycled["status"] == delisted.RECYCLED
    assert recycled["overrun_days"] > delisted.TAIL_TOLERANCE_DAYS


def test_a_name_dropped_for_market_cap_may_keep_trading(history):
    window = (pd.Timestamp("2005-01-03"), pd.Timestamp("2015-06-01"))

    result = delisted.classify(
        "DROPPED", _prices("2005-01-03", "2026-01-02"), window[1], "Market capitalization changes.", window
    )

    assert result["status"] == delisted.OK  # still listed: a continuing series is expected


def test_too_little_history_in_the_membership_window_is_refused(history):
    window = (pd.Timestamp("2005-01-03"), pd.Timestamp("2015-06-01"))

    result = delisted.classify(
        "DROPPED", _prices("2014-01-02", "2015-06-01"), window[1], "Market capitalization changes.", window
    )

    assert result["status"] == delisted.TOO_SHORT
    assert result["coverage"] < delisted.MIN_COVERAGE


def test_survey_reports_every_removed_name(history):
    def loader(ticker):
        return _prices("2005-01-03", "2010-05-07") if ticker == "GONE" else None

    out = delisted.survey("2005-01-01", loader=loader)

    assert set(out["ticker"]) == {"GONE", "DROPPED"}
    assert delisted.usable_tickers(out) == ["GONE"]
    assert out.loc[out["ticker"] == "DROPPED", "status"].iloc[0] == delisted.NO_DATA


def test_membership_window_is_clipped_to_the_panel_span(history):
    assert delisted.membership_window("GONE", "2005-01-01") == (
        pd.Timestamp("2005-01-01"),
        pd.Timestamp("2010-05-10"),
    )
    # Membership ended before the panel starts: nothing usable.
    assert delisted.membership_window("GONE", "2012-01-01") is None
