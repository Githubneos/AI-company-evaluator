"""Event taxonomy from 8-K item codes (spec 1.3).

An 8-K item code is a structured label the filer assigns under SEC rules. Using
it directly means the taxonomy involves no text parsing, no classifier, and no
LLM -- and therefore no silent misclassification. The mapping below is the whole
labelling system.

WHAT THIS CANNOT LABEL
----------------------
Four of the spec's categories are not recoverable from item codes alone, and are
left deliberately unpopulated rather than approximated:

  GUIDANCE_RAISE / GUIDANCE_CUT   Usually inside an Item 2.02 or 8.01 narrative.
  LITIGATION_FILED / _RESOLVED    Almost always Item 8.01, undifferentiated.
  DEBT_DOWNGRADE / _UPGRADE       Rating actions come from agencies, not filings.
  DIVIDEND_CHANGE                 Populated from the payment series instead;
                                  see evaluator.data.dividends.

Guessing at these from item codes would produce labels that look complete and
are wrong, which is worse than a gap the model can see. Populating them properly
needs the LLM classification pass over filing text that was scoped out.

EARNINGS DIRECTION
------------------
Item 2.02 says results were reported; it does not say whether they beat. The
spec's EARNINGS_SURPRISE_POS/NEG split is therefore collapsed to a single
EARNINGS_RESULT type. Deriving the sign from the subsequent price move would be
circular -- that move is the label the model is trying to predict.
"""

from __future__ import annotations

import hashlib
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16

# Item 9.01 is "Financial Statements and Exhibits" -- boilerplate co-filed with
# most 8-Ks. Counting it would make event density a measure of paperwork. Its
# pre-2004 equivalent is the bare Item 7.
NOISE_ITEMS = {"9.01"}
LEGACY_NOISE_ITEMS = {"7"}

# The SEC renumbered 8-K items on 2004-08-23. Filings before that use bare
# single-digit codes with entirely different meanings -- legacy Item 5 is "other
# events", modern Item 5.02 is an executive change. Roughly a decade of history
# lands in the legacy scheme, so it needs its own map rather than being dropped
# into UNCLASSIFIED. The two schemes are told apart by the dot: modern codes are
# always dotted, legacy codes never are.
LEGACY_ITEM_TAXONOMY: dict[str, tuple[str, str]] = {
    "1": ("MA_ANNOUNCED", "CONTROL_CHANGE"),
    "2": ("MA_ANNOUNCED", "ACQUISITION_OR_DISPOSITION"),
    "3": ("BANKRUPTCY", "BANKRUPTCY_OR_RECEIVERSHIP"),
    "4": ("AUDITOR_CHANGE", "ACCOUNTANT_CHANGED"),
    "5": ("DISCLOSURE", "OTHER_EVENTS"),
    "6": ("EXECUTIVE_CHANGE", "DIRECTOR_RESIGNATION"),
    "8": ("GOVERNANCE", "FISCAL_YEAR_CHANGE"),
    "9": ("DISCLOSURE", "REG_FD"),
    "10": ("GOVERNANCE", "ETHICS_CODE_CHANGE"),
    "11": ("GOVERNANCE", "TRADING_SUSPENSION_BLACKOUT"),
    "12": ("EARNINGS_RESULT", "RESULTS_OF_OPERATIONS"),
}

ITEM_TAXONOMY: dict[str, tuple[str, str]] = {
    "1.01": ("MA_ANNOUNCED", "MATERIAL_AGREEMENT"),
    "1.02": ("MA_TERMINATED", "AGREEMENT_TERMINATED"),
    "1.03": ("BANKRUPTCY", "BANKRUPTCY_OR_RECEIVERSHIP"),
    "1.04": ("REGULATORY_ACTION", "MINE_SAFETY"),
    "2.01": ("MA_ANNOUNCED", "ACQUISITION_COMPLETED"),
    "2.02": ("EARNINGS_RESULT", "RESULTS_OF_OPERATIONS"),
    "2.03": ("DEBT_OBLIGATION", "OBLIGATION_CREATED"),
    "2.04": ("DEBT_OBLIGATION", "ACCELERATION_TRIGGERED"),
    "2.05": ("RESTRUCTURING", "EXIT_OR_DISPOSAL_COSTS"),
    "2.06": ("ASSET_IMPAIRMENT", "MATERIAL_IMPAIRMENT"),
    "3.01": ("REGULATORY_ACTION", "LISTING_RULE_NOTICE"),
    "3.02": ("EQUITY_ISSUANCE", "UNREGISTERED_SALE"),
    "3.03": ("EQUITY_ISSUANCE", "SECURITY_HOLDER_RIGHTS"),
    "4.01": ("AUDITOR_CHANGE", "ACCOUNTANT_CHANGED"),
    "4.02": ("FRAUD_ACCOUNTING", "NON_RELIANCE_RESTATEMENT"),
    "5.01": ("MA_ANNOUNCED", "CONTROL_CHANGE"),
    "5.02": ("EXECUTIVE_CHANGE", "DIRECTOR_OR_OFFICER"),
    "5.03": ("GOVERNANCE", "BYLAW_AMENDMENT"),
    "5.04": ("GOVERNANCE", "TRADING_SUSPENSION_BLACKOUT"),
    "5.05": ("GOVERNANCE", "ETHICS_CODE_CHANGE"),
    "5.06": ("GOVERNANCE", "SHELL_STATUS_CHANGE"),
    "5.07": ("GOVERNANCE", "SHAREHOLDER_VOTE"),
    "5.08": ("GOVERNANCE", "DIRECTOR_NOMINATION"),
    "6.01": ("REGULATORY_ACTION", "ABS_INFORMATIONAL"),
    "7.01": ("DISCLOSURE", "REG_FD"),
    "8.01": ("DISCLOSURE", "OTHER_EVENTS"),
}

FORM_TAXONOMY: dict[str, tuple[str, str]] = {
    "10-K": ("PERIODIC_REPORT", "ANNUAL"),
    "10-Q": ("PERIODIC_REPORT", "QUARTERLY"),
}

EVENT_COLUMNS = [
    "event_id",
    "ticker",
    "event_date",
    "event_type",
    "event_subtype",
    "source",
    "confidence",
]

#: Categories in the spec that item codes cannot populate. Surfaced so the
#: absence is visible in reports rather than mistaken for "these never happen".
UNPOPULATED_TYPES = [
    "GUIDANCE_RAISE",
    "GUIDANCE_CUT",
    "LITIGATION_FILED",
    "LITIGATION_RESOLVED",
    "DEBT_DOWNGRADE",
    "DEBT_UPGRADE",
]


def effective_date(
    filing_date: pd.Series, acceptance: pd.Series
) -> pd.Series:
    """The first date on which the market could act on a filing.

    A filing accepted after the 16:00 ET close is not tradeable information
    until the following session. Treating acceptance time as same-day knowledge
    is a lookahead leak concentrated precisely in earnings events, where it
    would do the most damage.
    """
    dates = pd.to_datetime(filing_date).dt.normalize()

    accepted = pd.to_datetime(acceptance, errors="coerce", utc=True)
    known = accepted.notna()
    if not known.any():
        return dates

    local_hour = accepted[known].dt.tz_convert(MARKET_TZ).dt.hour
    after_close = local_hour >= MARKET_CLOSE_HOUR

    shifted = dates.copy()
    shift_index = local_hour[after_close].index
    # "Next day" is not necessarily the next session: an after-close Friday
    # filing is actionable on Monday, not Saturday.  A business-day roll handles
    # the regular weekend closure (exchange holidays are still safely
    # conservative because an event dated on a holiday is only consumed by the
    # next available price row).
    shifted.loc[shift_index] = dates.loc[shift_index] + pd.offsets.BDay(1)
    return shifted


def _event_id(ticker: str, date: pd.Timestamp, event_type: str, accession: str) -> str:
    raw = f"{ticker}|{date:%Y-%m-%d}|{event_type}|{accession}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def parse_items(items: object) -> list[str]:
    """Split EDGAR's comma-joined item string, dropping boilerplate.

    Noise differs by era: modern filings bury exhibits in 9.01, legacy ones in a
    bare 7.
    """
    if not isinstance(items, str) or not items.strip():
        return []

    codes = []
    for code in items.split(","):
        code = code.strip()
        if not code:
            continue
        noise = NOISE_ITEMS if "." in code else LEGACY_NOISE_ITEMS
        if code not in noise:
            codes.append(code)
    return codes


def classify_item(code: str) -> tuple[str, str]:
    """Map one 8-K item code to (event_type, event_subtype), era-aware."""
    table = ITEM_TAXONOMY if "." in code else LEGACY_ITEM_TAXONOMY
    return table.get(code, ("UNCLASSIFIED_8K", code))


def build_event_table(filings: pd.DataFrame) -> pd.DataFrame:
    """Normalised event table per spec 1.3.

    Columns: event_id, ticker, event_date, event_type, event_subtype, source,
    confidence. One row per (filing, item code) -- an 8-K reporting both an
    earnings release and a CFO departure is genuinely two events.
    """
    if filings.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    frame = filings.copy()
    frame["event_date"] = effective_date(
        frame["filing_date"], frame.get("acceptance_datetime", pd.Series(dtype="object"))
    )

    records = []
    for row in frame.itertuples(index=False):
        codes = parse_items(getattr(row, "items", None))

        if row.form == "8-K" and codes:
            for code in codes:
                event_type, subtype = classify_item(code)
                records.append(
                    (row.ticker, row.event_date, event_type, subtype, f"8-K:{code}", 1.0, row.accession)
                )
        elif row.form in FORM_TAXONOMY:
            event_type, subtype = FORM_TAXONOMY[row.form]
            records.append(
                (row.ticker, row.event_date, event_type, subtype, row.form, 0.9, row.accession)
            )

    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    events = pd.DataFrame(
        records,
        columns=[
            "ticker",
            "event_date",
            "event_type",
            "event_subtype",
            "source",
            "confidence",
            "accession",
        ],
    )
    events["event_id"] = [
        _event_id(r.ticker, r.event_date, r.event_type, r.accession)
        for r in events.itertuples(index=False)
    ]
    return (
        events[EVENT_COLUMNS]
        .drop_duplicates(subset=["event_id"])
        .sort_values(["ticker", "event_date"])
        .reset_index(drop=True)
    )


def event_types() -> list[str]:
    """Every type the mapping can emit, for stable categorical encoding.

    Stable ordering matters: this list becomes the categorical encoding shared
    by training and serving, and a reordering between the two would silently
    relabel every event feature.
    """
    types = {t for t, _ in ITEM_TAXONOMY.values()}
    types |= {t for t, _ in LEGACY_ITEM_TAXONOMY.values()}
    types |= {t for t, _ in FORM_TAXONOMY.values()}
    types.add("UNCLASSIFIED_8K")
    return sorted(types)
