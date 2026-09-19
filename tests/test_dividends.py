"""Dividend events: derived from payments, dated when they were observable."""

import numpy as np
import pandas as pd
import pytest

from evaluator.data import dividends as div
from evaluator.dataset import build_dataset
from evaluator.features.build import NO_EVENT_SENTINEL, build_event_features

QUARTER = 91


def _series(amounts, start="2020-01-15", step=QUARTER) -> pd.Series:
    dates = pd.date_range(start, periods=len(amounts), freq=f"{step}D")
    return pd.Series(amounts, index=dates, dtype="float64")


def test_first_payment_initiates():
    events = div.dividend_events("T", _series([0.5, 0.5, 0.5]), as_of=pd.Timestamp("2020-08-01"))

    assert list(events["event_type"]) == ["DIVIDEND_INITIATE"]
    assert events["event_date"].iloc[0] == pd.Timestamp("2020-01-15")


def test_raises_and_cuts_need_more_than_a_rounding_change():
    # 0.500 -> 0.505 is under the 1% tolerance; 0.505 -> 0.60 is a raise; then a cut.
    # as_of just after the last payment: not yet late enough to look suspended.
    events = div.dividend_events("T", _series([0.50, 0.505, 0.60, 0.30]), as_of=pd.Timestamp("2020-11-01"))
    kinds = list(events["event_type"])

    assert kinds == ["DIVIDEND_INITIATE", "DIVIDEND_RAISE", "DIVIDEND_CUT"]
    assert events["event_subtype"].iloc[2] == "-50.0%"


def test_suspension_is_dated_when_the_payment_failed_to_arrive():
    paid = pd.concat([_series([0.5, 0.5, 0.5]), pd.Series([0.5], index=[pd.Timestamp("2022-06-01")])])

    events = div.dividend_events("T", paid, as_of=pd.Timestamp("2023-01-01"))
    suspension = events[events["event_type"] == "DIVIDEND_SUSPEND"].iloc[0]
    last_payment = pd.Timestamp("2020-07-15")

    # Detected 1.5 quarters after the last payment, not at the last payment,
    # and strictly before the resumption that eventually revealed the gap.
    assert suspension["event_date"] > last_payment
    assert suspension["event_date"] < pd.Timestamp("2022-06-01")
    assert list(events["event_type"]).count("DIVIDEND_INITIATE") == 2  # resumed


def test_a_company_that_simply_stops_gets_a_suspension_once_the_date_passes():
    paid = _series([0.5, 0.5, 0.5])  # last payment 2020-07-15

    early = div.dividend_events("T", paid, as_of=pd.Timestamp("2020-09-01"))
    later = div.dividend_events("T", paid, as_of=pd.Timestamp("2021-01-01"))

    assert "DIVIDEND_SUSPEND" not in set(early["event_type"])  # not yet late
    assert "DIVIDEND_SUSPEND" in set(later["event_type"])


def test_no_event_is_dated_before_the_payment_that_implies_it():
    paid = _series([0.5, 0.5, 0.6, 0.6, 0.2])
    events = div.dividend_events("T", paid, as_of=pd.Timestamp("2022-01-01"))

    for row in events.itertuples():
        if row.event_type in ("DIVIDEND_RAISE", "DIVIDEND_CUT", "DIVIDEND_INITIATE"):
            assert row.event_date in set(paid.index)  # dated on an actual ex-date
        else:  # a suspension is dated after the payment it followed, never before
            assert row.event_date > paid.index[paid.index < row.event_date].max()


def test_an_empty_or_missing_series_produces_no_events():
    assert div.dividend_events("T", pd.Series(dtype="float64")).empty
    assert div.dividend_events("T", None).empty


def test_cached_series_is_reused_without_a_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(div, "DIVIDEND_DIR", tmp_path)
    paid = _series([0.5, 0.6])
    div.atomic_write_parquet(
        pd.DataFrame({"ex_date": paid.index, "dividend": paid.to_numpy()}), div._cache_path("T"), index=False
    )

    def explode(*a, **k):
        raise AssertionError("should not hit the network")

    monkeypatch.setitem(__import__("sys").modules, "yfinance", type("M", (), {"Ticker": explode}))

    pd.testing.assert_series_equal(div.load_dividends("T"), paid, check_names=False, check_freq=False)


def test_features_track_cuts_and_raises():
    index = pd.bdate_range("2020-01-01", periods=400)
    events = div.dividend_events("T", _series([0.5, 0.5, 0.2, 0.2]), as_of=pd.Timestamp("2021-06-01"))

    features = build_event_features(index, events)

    cut_date = events.loc[events["event_type"] == "DIVIDEND_CUT", "event_date"].iloc[0]
    before, after = cut_date - pd.Timedelta(days=1), cut_date + pd.Timedelta(days=30)
    assert features.loc[:before, "days_since_dividend_cut"].iloc[-1] == NO_EVENT_SENTINEL
    assert features.loc[after, "days_since_dividend_cut"] == 30
    assert features.loc[after, "dividend_cuts_365d"] == 1
    assert features.loc[:before, "dividend_cuts_365d"].iloc[-1] == 0


def test_build_dataset_joins_dividend_events(monkeypatch):
    dates = pd.bdate_range("2021-01-04", periods=300)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
    prices = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=dates,
    )
    monkeypatch.setattr("evaluator.dataset.load_prices", lambda *a, **k: prices)
    monkeypatch.setattr("evaluator.dataset.load_benchmark", lambda *a, **k: prices)
    monkeypatch.setattr("evaluator.dataset.load_macro", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("evaluator.dataset.cik_for", lambda ticker: None)
    monkeypatch.setattr("evaluator.dataset.sector_map", dict)
    paid = _series([0.5, 0.5, 0.25, 0.25], start="2021-02-01")

    dataset = build_dataset("T", "2021-01-01", dividends=paid, with_fundamentals=False)

    features = dataset.features
    assert features["dividend_cuts_365d"].max() == 1
    cut_day = features.index[features["dividend_cuts_365d"].to_numpy().argmax()]
    assert cut_day >= pd.Timestamp("2021-08-01")  # the third payment, halved
    assert features.loc[dates[0], "days_since_dividend_cut"] == NO_EVENT_SENTINEL


def test_dividend_change_is_no_longer_listed_as_underivable():
    from evaluator.events import UNPOPULATED_TYPES

    assert "DIVIDEND_CHANGE" not in UNPOPULATED_TYPES
    assert set(div.DIVIDEND_EVENT_TYPES) == {
        "DIVIDEND_INITIATE", "DIVIDEND_RAISE", "DIVIDEND_CUT", "DIVIDEND_SUSPEND",
    }


@pytest.mark.parametrize("amount", [-1.0, 0.0])
def test_non_positive_payments_are_ignored(amount):
    assert div.dividend_events("T", pd.Series([amount], index=[pd.Timestamp("2020-01-01")])).empty
