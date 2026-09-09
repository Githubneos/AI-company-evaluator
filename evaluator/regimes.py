"""Named market regimes and NBER recession dating (spec 2.4, 3.3).

Hardcoded rather than fetched from FRED: these are settled historical facts,
they change at most once every several years, and a training run should not
depend on a network call to reproduce.

The regime list exists because of the spec's warning about effective sample
size. Twenty years of daily bars looks like 5,000 observations per name, but it
contains only a handful of genuinely independent market environments. A model
can post a good aggregate score while being actively harmful in the one regime
where accuracy matters -- so validation reports per regime, not just pooled.
"""

from __future__ import annotations

import pandas as pd

#: NBER business-cycle contractions (peak month -> trough month), inclusive.
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

#: Named environments for per-regime validation. Contiguous and non-overlapping;
#: chosen so each contains a distinct volatility and rate backdrop.
REGIMES = [
    ("pre_gfc", "2005-01-01", "2007-11-30"),
    ("gfc", "2007-12-01", "2009-06-30"),
    ("recovery", "2009-07-01", "2019-12-31"),
    ("covid_crash", "2020-01-01", "2020-04-30"),
    ("covid_recovery", "2020-05-01", "2021-12-31"),
    ("rate_hikes", "2022-01-01", "2023-07-31"),
    ("recent", "2023-08-01", "2100-01-01"),
]


def recession_flag(index: pd.DatetimeIndex) -> pd.Series:
    """1.0 during an NBER contraction, else 0.0.

    Known only in arrears in real life -- NBER dates recessions months after the
    fact -- so treat this as a regime *label* for analysis, not as a feature a
    live model could legitimately have used. It is included in training because
    the spec asks for it; its SHAP attributions should be read with that caveat.
    """
    flag = pd.Series(0.0, index=index)
    for start, end in NBER_RECESSIONS:
        flag[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))] = 1.0
    return flag


def regime_for(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """Map each date to its named regime."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    labels = pd.Series("unknown", index=range(len(dates)), dtype=object)
    for name, start, end in REGIMES:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        labels[mask] = name
    return labels


def regime_names() -> list[str]:
    return [name for name, _, _ in REGIMES]
