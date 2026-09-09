"""Purged, embargoed walk-forward splitting.

Random k-fold on this data leaks the future into the past twice over: once
because a fold's training rows can postdate its test rows, and once because
overlapping label windows mean a training row dated t-1 and a test row dated t
share most of the same forward return. Both inflate scores dramatically, and
the resulting model looks excellent right up until it is asked to predict a
day it has not already seen.

Two guards, following Lopez de Prado, *Advances in Financial Machine
Learning* ch. 7:

purge   Drop training rows whose forward label window reaches into the test
        block. With an H-day horizon, a row dated t is only safe to train on
        if t + H is strictly before the test block starts, so the last H rows
        before each test block are dropped.

embargo Drop training rows in a buffer immediately *after* a test block, for
        every later fold that would otherwise include them. Their features
        overlap the test block's label windows, so serial correlation carries
        test information into training even though the dates run the right way.

Folds walk forward with an expanding training window, which is what a model
that gets retrained on a schedule actually experiences in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForward:
    n_splits: int
    test_size: int
    purge: int
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge and embargo must be non-negative")

    def min_samples(self) -> int:
        """Smallest sample count that leaves every fold a non-empty training set."""
        return self.n_splits * self.test_size + self.purge + 1

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) over rows assumed sorted by time, ascending."""
        if n_samples < self.min_samples():
            raise ValueError(
                f"need at least {self.min_samples()} samples for "
                f"n_splits={self.n_splits}, test_size={self.test_size}, "
                f"purge={self.purge}; got {n_samples}"
            )

        first_test_start = n_samples - self.n_splits * self.test_size
        embargoed: list[tuple[int, int]] = []

        for fold in range(self.n_splits):
            test_start = first_test_start + fold * self.test_size
            test_stop = test_start + self.test_size

            train_stop = test_start - self.purge
            candidates = np.arange(train_stop)

            mask = np.ones(train_stop, dtype=bool)
            for lo, hi in embargoed:
                mask[lo:hi] = False

            yield candidates[mask], np.arange(test_start, test_stop)

            embargoed.append((test_stop, min(test_stop + self.embargo, n_samples)))

    def split_panel(self, dates) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Split a cross-sectional panel by *date*, not by row.

        In a panel, hundreds of tickers share each date. Splitting on row
        position would cut through a single trading day, putting some names of
        day X in training and the rest in test. Those rows share the same market
        move, the same macro state, and overlapping label windows, so the model
        would effectively be told the answer for the day it is being tested on --
        a leak that row-wise purging cannot see, because the dates look
        correctly ordered.

        `test_size`, `purge` and `embargo` are therefore interpreted in trading
        days here, and every row belonging to a date lands on the same side.
        """
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        if not dates.is_monotonic_increasing:
            raise ValueError("panel rows must be sorted by date before splitting")

        unique_dates = dates.unique()
        positions = np.searchsorted(unique_dates, dates)

        for train_days, test_days in self.split(len(unique_dates)):
            train_mask = np.isin(positions, train_days)
            test_mask = np.isin(positions, test_days)
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
