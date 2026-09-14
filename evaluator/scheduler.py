"""Job registry and runner (spec 7), the Airflow/Dagster substitute.

Every batch responsibility in the spec is a named job here with an explicit
cadence. Running them is deliberately someone else's problem -- cron, launchd,
or a real orchestrator -- because scheduling is the part of Airflow worth
outsourcing, while the job definitions are the part worth owning.

    python -m evaluator.scheduler --list
    python -m evaluator.scheduler --run resolve_outcomes
    python -m evaluator.scheduler --install-cron

Swapping in Airflow means writing a DAG whose tasks call `JOBS[name].run`; the
job bodies do not change.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

log = logging.getLogger(__name__)

#: How many names the 15-minute sentiment poll warms. FinBERT on CPU is the
#: bottleneck: the poll has to finish well inside its own interval, or runs
#: pile up behind each other.
SENTIMENT_POLL_NAMES = 50


@dataclass
class Job:
    name: str
    cadence: str
    description: str
    run: Callable[[], dict]


def _resolve_outcomes() -> dict:
    from evaluator.feedback.store import resolve_pending

    return {"resolved": resolve_pending()}


def _post_mortem() -> dict:
    from evaluator.feedback.postmortem import tag_predictions

    return tag_predictions()


def _drift_check() -> dict:
    from evaluator.features.store import load_panel
    from evaluator.retrain import decide

    panel = load_panel()
    return decide(panel.frame, panel.feature_names).as_dict()


def _retrain() -> dict:
    from evaluator.retrain import run

    return run()


def _refresh_panel() -> dict:
    from evaluator.features.store import build_panel

    panel = build_panel()
    return {"rows": len(panel.frame), "schema": panel.schema}


def _poll_sentiment() -> dict:
    """Warm the sentiment cache for the universe.

    Polling every 5-15 minutes is enough for this use case (spec 7); the Kafka
    streaming path is only worth it if true real-time becomes a requirement.
    """
    from evaluator.data.universe import cik_for, tickers
    from evaluator.sentiment.aggregate import score_ticker_sentiment

    scored, stale = 0, 0
    for ticker in tickers(limit=SENTIMENT_POLL_NAMES):
        try:
            result = score_ticker_sentiment(ticker, cik_for(ticker))
            scored += 1
            stale += int(result.stale)
        except Exception as exc:  # noqa: BLE001
            log.warning("sentiment poll failed for %s: %s", ticker, exc)
    return {"scored": scored, "stale": stale}


def _score_universe() -> dict:
    from evaluator.scoring import score_universe

    summary = score_universe()
    return {k: v for k, v in summary.items() if k != "failures"} | {"failures": len(summary["failures"])}


def _monitoring() -> dict:
    from evaluator.monitoring import system_report

    return system_report()


JOBS: dict[str, Job] = {
    job.name: job
    for job in [
        Job("score_universe", "daily", "Score every name from production models (7.3)", _score_universe),
        Job("resolve_outcomes", "daily", "Close out predictions whose window elapsed", _resolve_outcomes),
        Job("post_mortem", "weekly", "Tag why wrong predictions were wrong (6.2)", _post_mortem),
        Job("poll_sentiment", "every 15 min", "Refresh news sentiment cache (4.2)", _poll_sentiment),
        Job("drift_check", "daily", "PSI drift against training distribution (6.3)", _drift_check),
        Job("refresh_panel", "weekly", "Rebuild the feature panel (2.5)", _refresh_panel),
        Job("retrain", "monthly", "Retrain if cadence or drift requires it (6.3)", _retrain),
        Job("monitoring", "daily", "System health report (8.2)", _monitoring),
    ]
}

CRON_LINES = {
    "score_universe": "0 22 * * 1-5",
    "resolve_outcomes": "30 22 * * 1-5",
    "post_mortem": "0 3 * * 6",
    "poll_sentiment": "*/15 13-21 * * 1-5",
    "drift_check": "0 4 * * *",
    "refresh_panel": "0 2 * * 6",
    "retrain": "0 5 1 * *",
    "monitoring": "0 23 * * *",
}


def run_job(name: str) -> dict:
    if name not in JOBS:
        raise KeyError(f"unknown job {name!r}. Known: {', '.join(sorted(JOBS))}")
    job = JOBS[name]
    started = datetime.now(UTC)
    log.info("running job %s", name)
    try:
        result = job.run()
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - a failed job is a reported outcome
        log.exception("job %s failed", name)
        result, status = {"error": str(exc)}, "failed"
    return {
        "job": name,
        "status": status,
        "started_at": started.isoformat(),
        "seconds": round((datetime.now(UTC) - started).total_seconds(), 2),
        "result": result,
    }


def cron_file(python: str | None = None, cwd: str | None = None) -> str:
    python = python or sys.executable
    cwd = cwd or os.getcwd()
    lines = ["# AI Company Evaluator scheduled jobs"]
    for name, schedule in CRON_LINES.items():
        # Quoted: a checkout under "~/My Projects" would otherwise split into two
        # arguments and every job would silently fail to start.
        lines.append(f"{schedule} cd {shlex.quote(cwd)} && {shlex.quote(python)} -m evaluator.scheduler --run {name}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", default=None)
    parser.add_argument("--install-cron", action="store_true", help="print a crontab to install")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.list:
        print(f"{'job':<18} {'cadence':<14} description")
        print("-" * 76)
        for job in JOBS.values():
            print(f"{job.name:<18} {job.cadence:<14} {job.description}")
        return

    if args.install_cron:
        print(cron_file())
        print("# Install with:  python -m evaluator.scheduler --install-cron | crontab -", file=sys.stderr)
        return

    if args.run:
        if args.run not in JOBS:
            parser.error(f"unknown job {args.run!r}. Known: {', '.join(sorted(JOBS))}")
        outcome = run_job(args.run)
        print(json.dumps(outcome, indent=2, default=str))
        # A failed job has to be visible to whatever launched it: cron mails on
        # a non-zero exit, and a wrapper script can only branch on one.
        if outcome["status"] == "failed":
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
