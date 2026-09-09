"""Historical analog retrieval tests (spec 3.5).

The property that matters: retrieved analogs must predate the query. A
"historical" analog drawn from after the query date is a look-ahead leak wearing
a convincing disguise -- the outcomes look plausible, the distances look right,
and the reasoning layer will present next month's data as precedent.
"""

import numpy as np
import pandas as pd
import pytest

from evaluator.model.analogs import ANALOG_FEATURES, build_index


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=600)
    rows = []
    for date in dates:
        for ticker in ("AAA", "BBB", "CCC"):
            row = {
                "ticker": ticker,
                "date": date,
                "forward_return_5d": float(rng.normal(0, 0.03)),
                "forward_return_20d": float(rng.normal(0, 0.06)),
                "label_direction_5d": float(rng.integers(0, 3)),
            }
            for feature in ANALOG_FEATURES:
                row[feature] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def index(panel):
    return build_index(panel, max_rows=100_000)


def _query_row(panel) -> pd.DataFrame:
    return panel.iloc[[-1]][ANALOG_FEATURES]


class TestAnalogRetrieval:
    def test_no_analog_is_dated_at_or_after_the_query(self, index, panel):
        as_of = pd.Timestamp("2016-06-01")
        matches = index.query(_query_row(panel), as_of=as_of, k=5)
        assert matches
        for match in matches:
            assert pd.Timestamp(match["date"]) < as_of

    def test_the_query_ticker_can_be_excluded(self, index, panel):
        matches = index.query(
            _query_row(panel), as_of=pd.Timestamp("2017-01-01"), k=5, exclude_ticker="AAA"
        )
        assert matches
        assert all(m["ticker"] != "AAA" for m in matches)

    def test_returns_at_most_k(self, index, panel):
        matches = index.query(_query_row(panel), as_of=pd.Timestamp("2017-01-01"), k=3)
        assert len(matches) <= 3

    def test_ranks_all_eligible_history_not_only_a_future_neighbour_pool(self, panel):
        """Nearby rows after as_of cannot crowd valid older analogs out."""
        # The final 200 rows are near the query but all future to this as_of;
        # an over-fetch-then-filter implementation returns none.
        frame = panel.copy()
        for feature in ANALOG_FEATURES:
            frame[feature] = 100.0
        frame.loc[frame.index[:20], ANALOG_FEATURES] = 0.0
        frame.loc[frame.index[-1], ANALOG_FEATURES] = 0.0
        index = build_index(frame, max_rows=100_000)

        matches = index.query(
            frame.iloc[[-1]][ANALOG_FEATURES],
            as_of=frame["date"].iloc[20],
            k=5,
        )
        assert matches
        assert all(pd.Timestamp(match["date"]) < frame["date"].iloc[20] for match in matches)

    def test_earliest_query_date_yields_no_analogs(self, index, panel):
        """Nothing precedes the start of history -- report none, do not reach forward."""
        first_date = panel["date"].min()
        matches = index.query(_query_row(panel), as_of=first_date, k=5)
        assert matches == []

    def test_matches_carry_realised_outcomes(self, index, panel):
        matches = index.query(_query_row(panel), as_of=pd.Timestamp("2017-01-01"), k=5)
        for match in matches:
            assert "forward_return_5d" in match
            assert match["outcome_5d"] in {"DROP", "NEUTRAL", "SPIKE", None}

    def test_distances_are_non_negative_and_ordered(self, index, panel):
        matches = index.query(_query_row(panel), as_of=pd.Timestamp("2018-01-01"), k=5)
        distances = [m["distance"] for m in matches]
        assert all(d >= 0 for d in distances)
        assert distances == sorted(distances)

    def test_nan_features_do_not_break_the_query(self, index, panel):
        row = _query_row(panel).copy()
        row.iloc[0, 0] = np.nan
        assert index.query(row, as_of=pd.Timestamp("2018-01-01"), k=3)

    def test_scaler_is_fit_on_the_index_not_the_query(self, index):
        """Fitting the scaler on query data would leak its distribution in."""
        assert index.scaler.n_samples_seen_ == len(index.reference)
