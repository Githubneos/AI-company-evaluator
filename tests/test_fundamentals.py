"""Point-in-time correctness for fundamentals (spec 2.2).

The failure this guards against is the classic one: joining quarterly
fundamentals on the fiscal period end rather than the filing date, which hands
the model Q3 revenue several weeks before it was public. It produces no error
and no NaN -- just a backtest that looks better than reality.
"""

import pandas as pd
import pytest

from evaluator.data.fundamentals import extract_fundamentals, join_as_of


def _facts(**overrides) -> dict:
    """Two quarters: period ends in March and June, filed weeks later."""
    facts = {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"start": "2024-01-01", "end": "2024-03-31", "val": 1000,
                         "form": "10-Q", "filed": "2024-05-01"},
                        {"start": "2024-04-01", "end": "2024-06-30", "val": 1200,
                         "form": "10-Q", "filed": "2024-08-01"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"end": "2024-03-31", "val": 5000, "form": "10-Q", "filed": "2024-05-01"},
                        {"end": "2024-06-30", "val": 5200, "form": "10-Q", "filed": "2024-08-01"},
                    ]
                }
            },
            "Liabilities": {
                "units": {
                    "USD": [
                        {"end": "2024-03-31", "val": 2000, "form": "10-Q", "filed": "2024-05-01"},
                        {"end": "2024-06-30", "val": 2100, "form": "10-Q", "filed": "2024-08-01"},
                    ]
                }
            },
            "StockholdersEquity": {
                "units": {
                    "USD": [
                        {"end": "2024-03-31", "val": 1000, "form": "10-Q", "filed": "2024-05-01"},
                        {"end": "2024-06-30", "val": 1050, "form": "10-Q", "filed": "2024-08-01"},
                    ]
                }
            },
        }
    }
    facts["us-gaap"].update(overrides)
    return facts


def test_values_are_stamped_with_filing_date_not_period_end():
    frame = extract_fundamentals(_facts())
    assert frame["filed"].tolist() == [
        pd.Timestamp("2024-05-01"),
        pd.Timestamp("2024-08-01"),
    ]


def test_fundamentals_are_invisible_before_they_are_filed():
    """The Q1 figures must not appear on any date before 2024-05-01."""
    frame = extract_fundamentals(_facts())
    index = pd.DatetimeIndex(["2024-03-31", "2024-04-30", "2024-05-01", "2024-05-02"])
    joined = join_as_of(index, frame)

    assert joined.loc["2024-03-31", "debt_to_equity"] != joined.loc["2024-03-31", "debt_to_equity"]  # NaN
    assert pd.isna(joined.loc["2024-04-30", "debt_to_equity"])
    assert joined.loc["2024-05-01", "debt_to_equity"] == pytest.approx(2.0)
    assert joined.loc["2024-05-02", "debt_to_equity"] == pytest.approx(2.0)


def test_later_quarter_appears_only_after_its_own_filing():
    frame = extract_fundamentals(_facts())
    index = pd.DatetimeIndex(["2024-07-31", "2024-08-01"])
    joined = join_as_of(index, frame)

    assert joined.loc["2024-07-31", "debt_to_equity"] == pytest.approx(2.0)
    assert joined.loc["2024-08-01", "debt_to_equity"] == pytest.approx(2100 / 1050)


def test_annual_flow_facts_are_excluded_from_quarterly_series():
    """A 10-K's full-year revenue is not comparable to a quarter's."""
    facts = _facts()
    facts["us-gaap"]["Revenues"]["units"]["USD"].append(
        {"start": "2024-01-01", "end": "2024-12-31", "val": 5000,
         "form": "10-K", "filed": "2025-02-01"}
    )
    frame = extract_fundamentals(facts)
    assert len(frame) == 2  # the annual row is filtered out


def test_restatement_does_not_overwrite_what_was_known_at_the_time():
    """A later refiling of the same period must not change history."""
    facts = _facts()
    facts["us-gaap"]["Liabilities"]["units"]["USD"].append(
        {"end": "2024-03-31", "val": 9999, "form": "10-Q", "filed": "2025-01-01"}
    )
    frame = extract_fundamentals(facts)
    index = pd.DatetimeIndex(["2024-05-02"])
    assert join_as_of(index, frame).loc["2024-05-02", "debt_to_equity"] == pytest.approx(2.0)


def test_a_gap_in_one_metric_does_not_blank_the_others():
    frame = pd.DataFrame(
        {
            "filed": pd.to_datetime(["2024-05-01", "2024-08-01"]),
            "revenue_growth_yoy": [0.1, None],
            "gross_margin": [0.4, 0.42],
            "net_margin": [0.2, 0.21],
            "debt_to_equity": [2.0, 2.1],
            "current_ratio": [1.1, 1.2],
            "earnings_surprise_streak": [0, 1],
        }
    )
    joined = join_as_of(pd.DatetimeIndex(["2024-08-02"]), frame)
    assert joined.loc["2024-08-02", "revenue_growth_yoy"] == pytest.approx(0.1)
    assert joined.loc["2024-08-02", "gross_margin"] == pytest.approx(0.42)


def test_empty_fundamentals_yield_the_full_nan_schema():
    joined = join_as_of(pd.DatetimeIndex(["2024-01-02"]), pd.DataFrame())
    assert "debt_to_equity" in joined.columns
    assert joined.isna().all().all()


def test_facts_without_a_filing_date_are_dropped():
    facts = _facts()
    facts["us-gaap"]["Assets"]["units"]["USD"].append(
        {"end": "2024-09-30", "val": 6000, "form": "10-Q", "filed": None}
    )
    frame = extract_fundamentals(facts)
    assert frame["filed"].notna().all()
