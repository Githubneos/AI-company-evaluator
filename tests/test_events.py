"""Event taxonomy tests (spec 1.3).

The load-bearing property is the after-close date shift. An 8-K accepted at
16:35 is not tradeable until the next session, and earnings are both the most
common 8-K and the largest scheduled source of big moves -- so getting this
wrong would leak hindsight exactly where it does the most damage, while every
metric still looked fine.
"""

import pandas as pd
import pytest

from evaluator.events import (
    LEGACY_ITEM_TAXONOMY,
    build_event_table,
    classify_item,
    effective_date,
    event_types,
    parse_items,
)


def _filing(**kwargs) -> dict:
    base = {
        "cik": 1,
        "ticker": "TEST",
        "form": "8-K",
        "filing_date": pd.Timestamp("2024-03-14"),
        "acceptance_datetime": pd.Timestamp("2024-03-14T12:00:00Z"),
        "report_date": pd.NaT,
        "items": "2.02,9.01",
        "accession": "0000000000-24-000001",
    }
    base.update(kwargs)
    return base


def frame(*filings) -> pd.DataFrame:
    return pd.DataFrame(list(filings))


def test_after_close_filing_moves_to_the_next_day():
    """16:35 ET is 20:35 UTC in March (EDT). The market cannot react until the 15th."""
    filings = frame(_filing(acceptance_datetime=pd.Timestamp("2024-03-14T20:35:00Z")))
    events = build_event_table(filings)
    assert events["event_date"].iloc[0] == pd.Timestamp("2024-03-15")


def test_after_close_friday_moves_to_monday_session():
    """The next *calendar* day is closed; features must see this on Monday."""
    filings = frame(
        _filing(
            filing_date=pd.Timestamp("2024-03-15"),
            acceptance_datetime=pd.Timestamp("2024-03-15T21:00:00Z"),
        )
    )
    assert build_event_table(filings)["event_date"].iloc[0] == pd.Timestamp("2024-03-18")


def test_intraday_filing_keeps_its_own_date():
    filings = frame(_filing(acceptance_datetime=pd.Timestamp("2024-03-14T14:00:00Z")))
    events = build_event_table(filings)
    assert events["event_date"].iloc[0] == pd.Timestamp("2024-03-14")


def test_shift_respects_eastern_time_not_utc():
    """19:30 UTC is 15:30 EDT -- before the close, so no shift.

    Comparing against UTC hours instead would wrongly push this to the next day.
    """
    filings = frame(
        _filing(
            filing_date=pd.Timestamp("2024-07-01"),
            acceptance_datetime=pd.Timestamp("2024-07-01T19:30:00Z"),
        )
    )
    assert build_event_table(filings)["event_date"].iloc[0] == pd.Timestamp("2024-07-01")


def test_shift_uses_eastern_offset_in_winter_too():
    """21:00 UTC is 16:00 EST in January -- at the close, so it shifts.

    A hardcoded summer offset would get this one wrong.
    """
    filings = frame(
        _filing(
            filing_date=pd.Timestamp("2024-01-10"),
            acceptance_datetime=pd.Timestamp("2024-01-10T21:00:00Z"),
        )
    )
    assert build_event_table(filings)["event_date"].iloc[0] == pd.Timestamp("2024-01-11")


def test_missing_acceptance_time_falls_back_to_filing_date():
    filings = frame(_filing(acceptance_datetime=pd.NaT))
    assert build_event_table(filings)["event_date"].iloc[0] == pd.Timestamp("2024-03-14")


def test_effective_date_handles_an_all_missing_column():
    dates = pd.Series([pd.Timestamp("2024-01-02")])
    result = effective_date(dates, pd.Series([pd.NaT]))
    assert result.iloc[0] == pd.Timestamp("2024-01-02")


def test_boilerplate_items_are_dropped():
    assert parse_items("2.02,9.01") == ["2.02"]  # modern exhibits
    assert parse_items("12,7") == ["12"]  # legacy exhibits
    assert parse_items("") == []
    assert parse_items(None) == []


def test_legacy_and_modern_item_five_mean_different_things():
    """The 2004 renumbering reused digits. Bare 5 is 'other events'; 5.02 is an
    executive change. Conflating them would mislabel a decade of history."""
    assert classify_item("5") == LEGACY_ITEM_TAXONOMY["5"]
    assert classify_item("5")[0] == "DISCLOSURE"
    assert classify_item("5.02")[0] == "EXECUTIVE_CHANGE"


def test_legacy_item_twelve_is_earnings():
    assert classify_item("12")[0] == "EARNINGS_RESULT"
    assert classify_item("2.02")[0] == "EARNINGS_RESULT"


def test_one_filing_with_two_items_yields_two_events():
    filings = frame(_filing(items="2.02,5.02"))
    events = build_event_table(filings)
    assert set(events["event_type"]) == {"EARNINGS_RESULT", "EXECUTIVE_CHANGE"}
    assert events["event_id"].nunique() == 2


def test_periodic_reports_are_captured_with_lower_confidence():
    filings = frame(_filing(form="10-K", items=None))
    events = build_event_table(filings)
    assert events["event_type"].iloc[0] == "PERIODIC_REPORT"
    assert events["confidence"].iloc[0] < 1.0


def test_unknown_item_code_is_flagged_not_silently_dropped():
    events = build_event_table(frame(_filing(items="99.99")))
    assert events["event_type"].iloc[0] == "UNCLASSIFIED_8K"
    assert events["event_subtype"].iloc[0] == "99.99"


def test_event_ids_are_stable_and_unique():
    filings = frame(_filing(), _filing(items="5.02", accession="0000000000-24-000002"))
    first = build_event_table(filings)
    second = build_event_table(filings)
    assert first["event_id"].tolist() == second["event_id"].tolist()
    assert first["event_id"].nunique() == len(first)


def test_empty_input_yields_empty_table_with_schema():
    events = build_event_table(pd.DataFrame())
    assert events.empty
    assert "event_type" in events.columns


def test_event_type_vocabulary_is_sorted_and_covers_both_eras():
    types = event_types()
    assert types == sorted(types)
    assert {"EARNINGS_RESULT", "EXECUTIVE_CHANGE", "BANKRUPTCY", "PERIODIC_REPORT"} <= set(types)


@pytest.mark.parametrize("code,expected", [("4.02", "FRAUD_ACCOUNTING"), ("1.03", "BANKRUPTCY")])
def test_high_signal_codes_map_to_their_own_types(code, expected):
    assert classify_item(code)[0] == expected
