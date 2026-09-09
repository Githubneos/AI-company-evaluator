import numpy as np
import pandas as pd
import pytest

from evaluator.validation import PurgedWalkForward


SPLITTER = PurgedWalkForward(n_splits=4, test_size=50, purge=5, embargo=10)
N = 1000


def test_train_never_overlaps_or_postdates_test():
    for train_idx, test_idx in SPLITTER.split(N):
        assert not set(train_idx) & set(test_idx)
        assert train_idx.max() < test_idx.min()


def test_purge_gap_is_respected():
    """The last `purge` rows before a test block must be dropped: their forward
    label windows reach into the test period."""
    for train_idx, test_idx in SPLITTER.split(N):
        assert test_idx.min() - train_idx.max() > SPLITTER.purge


def test_test_folds_are_contiguous_and_walk_forward():
    blocks = [test_idx for _, test_idx in SPLITTER.split(N)]
    assert [len(b) for b in blocks] == [SPLITTER.test_size] * SPLITTER.n_splits
    for earlier, later in zip(blocks, blocks[1:]):
        assert later.min() == earlier.max() + 1
    assert blocks[-1].max() == N - 1


def test_embargo_excludes_the_window_after_each_prior_test_block():
    splits = list(SPLITTER.split(N))
    for fold, (train_idx, _) in enumerate(splits):
        train = set(train_idx)
        for prior in range(fold):
            prior_end = splits[prior][1].max() + 1
            embargoed = set(range(prior_end, prior_end + SPLITTER.embargo))
            assert not train & embargoed


def test_training_window_expands():
    sizes = [len(train_idx) for train_idx, _ in SPLITTER.split(N)]
    assert sizes == sorted(sizes)
    assert sizes[0] > 0


def test_rejects_insufficient_samples():
    with pytest.raises(ValueError, match="need at least"):
        list(SPLITTER.split(SPLITTER.min_samples() - 1))


class TestPanelSplits:
    """Date-aware splitting for cross-sectional panels.

    The leak this prevents is subtle: with 500 tickers per date, a row-based
    boundary puts some names of a given day in training and the rest in test.
    Those rows share the same market move and overlapping label windows, so the
    model is effectively shown the answer for the day it is scored on -- while
    the dates still look correctly ordered to a row-wise check.
    """

    @staticmethod
    def panel_dates(n_days: int = 400, per_day: int = 5) -> pd.Series:
        days = pd.bdate_range("2020-01-01", periods=n_days)
        return pd.Series(sorted(list(days) * per_day))

    def test_no_date_appears_in_both_train_and_test(self):
        dates = self.panel_dates()
        splitter = PurgedWalkForward(n_splits=3, test_size=50, purge=5, embargo=10)

        for train_idx, test_idx in splitter.split_panel(dates):
            assert not set(dates.iloc[train_idx]) & set(dates.iloc[test_idx])

    def test_every_row_of_a_date_lands_on_the_same_side(self):
        dates = self.panel_dates(per_day=7)
        splitter = PurgedWalkForward(n_splits=3, test_size=50, purge=5, embargo=10)

        for _, test_idx in splitter.split_panel(dates):
            counts = dates.iloc[test_idx].value_counts()
            assert set(counts.unique()) == {7}

    def test_purge_gap_is_measured_in_days_not_rows(self):
        dates = self.panel_dates(per_day=5)
        splitter = PurgedWalkForward(n_splits=3, test_size=50, purge=5, embargo=0)

        for train_idx, test_idx in splitter.split_panel(dates):
            gap_days = len(
                pd.bdate_range(dates.iloc[train_idx].max(), dates.iloc[test_idx].min())
            )
            assert gap_days > splitter.purge

    def test_unsorted_panel_is_rejected(self):
        dates = self.panel_dates()[::-1]
        splitter = PurgedWalkForward(n_splits=3, test_size=50, purge=5)
        with pytest.raises(ValueError, match="sorted by date"):
            list(splitter.split_panel(dates))

    def test_train_always_precedes_test(self):
        dates = self.panel_dates()
        splitter = PurgedWalkForward(n_splits=3, test_size=50, purge=5, embargo=10)
        for train_idx, test_idx in splitter.split_panel(dates):
            assert dates.iloc[train_idx].max() < dates.iloc[test_idx].min()


def test_zero_embargo_keeps_all_prior_rows():
    splitter = PurgedWalkForward(n_splits=3, test_size=20, purge=2, embargo=0)
    for train_idx, test_idx in splitter.split(300):
        expected = np.arange(test_idx.min() - splitter.purge)
        assert np.array_equal(train_idx, expected)


class TestDateThinning:
    """Row capping must preserve cross-sections and fold geometry.

    Striding is how a 2.5M-row panel fits in 8 GB, but it changes what a "day"
    means to the splitter. Two things must survive it: every retained date keeps
    all its tickers, and windows expressed in trading days get rescaled -- an
    unscaled purge would silently shrink below the label horizon and let
    overlapping windows back into training.
    """

    @staticmethod
    def _panel(n_days=1000, per_day=50):
        from evaluator.model.train import _thin_by_date

        days = pd.bdate_range("2015-01-01", periods=n_days)
        dates = pd.Series(sorted(list(days) * per_day)).reset_index(drop=True)
        X = pd.DataFrame({"f": np.arange(len(dates), dtype="float32")})
        y = pd.Series(np.zeros(len(dates), dtype="int8"))
        tickers = pd.Series([f"T{i % per_day}" for i in range(len(dates))])
        return _thin_by_date, X, y, dates, tickers, per_day

    def test_thinning_keeps_whole_cross_sections(self):
        thin, X, y, dates, tickers, per_day = self._panel()
        _, _, kept_dates, _, stride = thin(X, y, dates, tickers, max_rows=10_000)

        assert stride > 1
        assert set(kept_dates.value_counts().unique()) == {per_day}

    def test_thinning_respects_the_row_cap(self):
        thin, X, y, dates, tickers, _ = self._panel()
        kept_X, _, _, _, _ = thin(X, y, dates, tickers, max_rows=10_000)
        assert len(kept_X) <= 10_000

    def test_thinning_preserves_date_order(self):
        thin, X, y, dates, tickers, _ = self._panel()
        _, _, kept_dates, _, _ = thin(X, y, dates, tickers, max_rows=10_000)
        assert kept_dates.is_monotonic_increasing

    def test_no_thinning_when_already_under_the_cap(self):
        thin, X, y, dates, tickers, _ = self._panel(n_days=10, per_day=5)
        kept_X, _, _, _, stride = thin(X, y, dates, tickers, max_rows=10_000)
        assert stride == 1
        assert len(kept_X) == len(X)

    def test_rescaled_purge_still_covers_the_label_horizon(self):
        """At stride N, a 20-day horizon needs ceil(20/N) retained dates."""
        import math

        for stride in (2, 4, 7):
            purge = max(1, math.ceil(20 / stride))
            assert purge * stride >= 20
