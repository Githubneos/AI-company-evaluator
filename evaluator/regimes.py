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

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

#: NBER business-cycle contractions (peak month -> trough month), inclusive.
log = logging.getLogger(__name__)

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


#: VIX bands for today's market state. The named regimes above are calendar
#: ranges, which cannot classify a live date beyond "recent"; what actually
#: matters for a model's reliability is the volatility backdrop it is being
#: asked to work in.
VOLATILITY_STATES = (
    ("calm", 0.0, 15.0),
    ("normal", 15.0, 22.0),
    ("stressed", 22.0, 30.0),
    ("crisis", 30.0, float("inf")),
)

#: Historical regimes that resembled each state, so a model's measured skill
#: there is the relevant evidence about it today. Judgement, stated openly:
#: 2022-23 was the stressed backdrop, the two crashes were the crisis ones.
ANALOGOUS_REGIMES = {
    "calm": ["recovery", "covid_recovery", "recent"],
    "normal": ["recovery", "recent"],
    "stressed": ["rate_hikes", "gfc"],
    "crisis": ["covid_crash", "gfc"],
}


def volatility_state(vix: float | None) -> str:
    """Today's backdrop from the VIX level. `unknown` when there is no quote."""
    if vix is None or not np.isfinite(vix):
        return "unknown"
    for name, low, high in VOLATILITY_STATES:
        if low <= vix < high:
            return name
    return "unknown"


@lru_cache(maxsize=4)
def _vix_on(day: str) -> float | None:
    """The latest cached VIX close. Cached per day, and never fetches.

    This runs on a request path, so it reads what the nightly jobs already
    wrote and nothing else. A stale or absent quote yields an `unknown` state,
    which the guard handles; a yfinance stall inside a web handler does not.
    """
    from evaluator.data.sources import load_macro

    try:
        macro = load_macro("2015-01-01", refresh=False)
    except Exception as exc:  # noqa: BLE001 - the guard degrades, it does not fail
        log.warning("VIX unavailable for the regime guard: %s", exc)
        return None
    if macro is None or macro.empty or "vix" not in macro:
        return None
    series = macro["vix"].dropna()
    return float(series.iloc[-1]) if len(series) else None


def current_vix() -> float | None:
    """The latest VIX close from the local cache; None if unavailable."""
    return _vix_on(str(pd.Timestamp.now().normalize().date()))


def current_regimes(vix: float | None = None, today: pd.Timestamp | None = None) -> dict:
    """The regimes whose measured skill describes a model's reliability today.

    The calendar regime says where we are in the validation split; the
    volatility state says what kind of market it is. Both are reported, and the
    guard takes the worst measured skill across them.
    """
    when = pd.Timestamp(today) if today is not None else pd.Timestamp.now().normalize()
    calendar = str(regime_for(pd.DatetimeIndex([when])).iloc[0])
    state = volatility_state(vix)
    relevant = [calendar] + [r for r in ANALOGOUS_REGIMES.get(state, []) if r != calendar]
    context = {
        "as_of": str(when.date()),
        "vix": vix,
        "state": state,
        "calendar": calendar,
        "relevant": relevant,
    }

    # The last regime has no real end date, so it absorbs every future date and
    # slowly stops describing anything. Say so rather than let "recent" imply a
    # market resembling the one the model was validated in.
    end = next((pd.Timestamp(e) for name, _, e in REGIMES if name == calendar), None)
    if end is not None and end.year >= 2100:
        started = next(pd.Timestamp(s) for name, s, _ in REGIMES if name == calendar)
        years = (when - started).days / 365.25
        context["calendar_open_ended"] = True
        context["calendar_note"] = (
            f"{calendar!r} is an open-ended bucket, running {years:.1f} years since "
            f"{started.date()}. It is where today falls in the validation split, not evidence "
            "that this market resembles that one; the volatility state is the better guide."
        )
    return context


def regime_guard(targets, *, vix: float | None = None, today=None, stage: str = "production") -> dict:
    """Is each model trustworthy in *today's* market, on its own measured record?

    A model that lost to the base rate the last time volatility looked like
    this should not be presented as a signal now. The numbers come from the
    model's own per-regime validation, so this adds no new claim: it just stops
    an average hiding the regime we are actually in.
    """
    from evaluator.model.registry import load_metadata

    context = current_regimes(current_vix() if vix is None else vix, today=today)
    relevant = context["relevant"]
    per_target, unreliable = {}, []

    for name in targets:
        by_regime = (load_metadata(name, stage) or {}).get("validation", {}).get("by_regime", {})
        measured = {
            regime: by_regime[regime]["brier_skill"]
            for regime in relevant
            if regime in by_regime and by_regime[regime].get("brier_skill") is not None
        }
        worst_regime = min(measured, key=measured.get) if measured else None
        worst = measured.get(worst_regime)
        reliable = worst is None or worst > 0
        if not reliable:
            unreliable.append(name)
        per_target[name] = {
            "measured": measured,
            "worst_regime": worst_regime,
            "worst_skill": worst,
            "reliable_now": reliable,
            "interpretation": (
                f"No per-regime record for {', '.join(relevant)}; reliability today is unmeasured."
                if worst is None
                else (
                    f"Measured skill {worst:+.4f} in {worst_regime}, the closest match to today's "
                    f"{context['state']} market. This model has lost to the base rate in conditions "
                    "like these: treat its output as no signal."
                    if not reliable
                    else f"Positive measured skill ({worst:+.4f}) in every regime resembling today's "
                    f"{context['state']} market."
                )
            ),
        }

    return {**context, "per_target": per_target, "unreliable_now": sorted(unreliable)}
