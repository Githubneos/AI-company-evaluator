"""Nightly scoring of the whole universe, so the dashboard never computes on demand.

Scoring one ticker means rebuilding its features through `build_dataset`, which
takes seconds. Doing that per request makes the dashboard slow and makes a
"highest risk today" view impossible. This runs once after the close, scores
every name from the **production** models, and writes a table the API can serve
directly.

One ticker's failure is that ticker's problem: it is recorded and the run
continues, because a nightly job that aborts on the first delisting or missing
cache produces nothing at all.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evaluator.config import ARTIFACT_DIR, MAGNITUDE, TargetSpec
from evaluator.data.universe import load_universe
from evaluator.io import atomic_write_json, atomic_write_parquet, read_parquet_or_none
from evaluator.model.predict import ModelNotTrained, available_targets, score_ticker

log = logging.getLogger(__name__)

SCORES_DIR = Path(ARTIFACT_DIR) / "scores"
LATEST_PATH = SCORES_DIR / "latest.json"

#: What the leaderboard ranks by unless asked otherwise.
DEFAULT_TARGET = TargetSpec(1, MAGNITUDE).name


def score_path(as_of: str) -> Path:
    return SCORES_DIR / f"{as_of}.parquet"


def _row(result: dict, sector: str | None) -> list[dict]:
    """One row per (ticker, target), flat enough for a table."""
    rows = []
    for name, scored in result.get("targets", {}).items():
        probabilities = scored["probabilities"]
        baseline = scored.get("baseline_probabilities") or {}
        # The interesting class is the last one: LARGE_MOVE, or SPIKE.
        outcome = list(probabilities)[-1]
        rows.append(
            {
                "ticker": result["ticker"],
                "sector": sector,
                "as_of": result["as_of"],
                "target": name,
                "kind": scored["kind"],
                "horizon_days": int(scored["horizon_days"]),
                "predicted_class": scored["predicted_class"],
                "outcome_class": outcome,
                "probability": float(probabilities[outcome]),
                "baseline": float(baseline.get(outcome, float("nan"))),
                "lift": float(scored.get("lift_over_baseline", {}).get(outcome, float("nan"))),
                "drop_probability": float(probabilities.get("DROP", float("nan"))),
                "last_close": result.get("last_close"),
                "move_threshold_pct": result.get("move_threshold_pct"),
            }
        )
    return rows


def score_universe(
    tickers: list[str] | None = None,
    *,
    lookback_start: str = "2015-01-01",
    limit: int | None = None,
) -> dict:
    """Score every name from the production models and persist the table."""
    if not available_targets():
        raise ModelNotTrained("no promoted models. Run: python -m scripts.promote --apply")

    universe = load_universe()
    sectors = dict(zip(universe["ticker"], universe["sector"]))
    names = tickers or universe["ticker"].tolist()
    if limit:
        names = names[:limit]

    started = datetime.now(UTC)
    rows: list[dict] = []
    failures: dict[str, str] = {}
    for i, ticker in enumerate(names, start=1):
        try:
            result = score_ticker(ticker, lookback_start=lookback_start)
            rows.extend(_row(result, sectors.get(ticker)))
        except Exception as exc:  # noqa: BLE001 - one name must not sink the run
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            log.warning("scoring failed for %s: %s", ticker, exc)
        if i % 25 == 0:
            log.info("scored %d/%d (%d failures)", i, len(names), len(failures))

    if not rows:
        raise RuntimeError(f"scored nothing; {len(failures)} failures")

    frame = pd.DataFrame(rows)
    as_of = str(frame["as_of"].max())
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(frame, score_path(as_of), index=False)

    summary = {
        "as_of": as_of,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round((datetime.now(UTC) - started).total_seconds(), 1),
        "tickers": int(frame["ticker"].nunique()),
        "rows": int(len(frame)),
        "targets": sorted(frame["target"].unique()),
        "failures": failures,
        "file": score_path(as_of).name,
    }
    atomic_write_json(summary, LATEST_PATH)
    log.info("scored %d tickers in %.0fs (%d failures)", summary["tickers"], summary["seconds"], len(failures))
    return summary


def latest_summary() -> dict | None:
    import json

    return json.loads(LATEST_PATH.read_text()) if LATEST_PATH.exists() else None


def load_scores(as_of: str | None = None) -> pd.DataFrame | None:
    """The most recent scored table, or a specific day's."""
    if as_of is None:
        summary = latest_summary()
        if summary is None:
            return None
        as_of = summary["as_of"]
    return read_parquet_or_none(score_path(as_of))


def leaderboard(
    target: str = DEFAULT_TARGET,
    *,
    sector: str | None = None,
    limit: int = 25,
    as_of: str | None = None,
) -> dict:
    """Highest-probability names for one target, from the stored table."""
    summary = latest_summary()
    scores = load_scores(as_of)
    if scores is None or summary is None:
        return {
            "available": False,
            "reason": "no scored universe yet. Run: python -m scripts.score_universe",
            "as_of": None,
            "rows": [],
        }

    subset = scores[scores["target"] == target]
    if sector:
        subset = subset[subset["sector"] == sector]
    subset = subset.sort_values("probability", ascending=False).head(max(1, min(limit, 500)))

    stale_by = _trading_days_since(summary["as_of"])
    return {
        "available": True,
        "as_of": summary["as_of"],
        "computed_at": summary.get("computed_at"),
        "target": target,
        "sector": sector,
        "tickers_scored": summary.get("tickers"),
        "failures": len(summary.get("failures", {})),
        "targets": summary.get("targets", []),
        "sectors": sorted(scores["sector"].dropna().unique().tolist()),
        "stale_trading_days": stale_by,
        "stale": stale_by > 1,
        "rows": subset.replace({float("nan"): None}).to_dict(orient="records"),
    }


def _trading_days_since(as_of: str) -> int:
    """How many trading days old the scored table is."""
    last = pd.Timestamp(as_of).normalize()
    today = pd.Timestamp.now().normalize()
    if today <= last:
        return 0
    return max(0, len(pd.bdate_range(last, today)) - 1)


__all__ = [
    "DEFAULT_TARGET",
    "LATEST_PATH",
    "SCORES_DIR",
    "latest_summary",
    "leaderboard",
    "load_scores",
    "score_universe",
]
