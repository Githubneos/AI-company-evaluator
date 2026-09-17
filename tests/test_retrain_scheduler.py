"""Retraining triggers and job registry (spec 6.3, 7)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from evaluator import retrain as rt
from evaluator.scheduler import CRON_LINES, JOBS, cron_file, run_job


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    from evaluator import scheduler

    monkeypatch.setattr(scheduler, "LOCK_DIR", tmp_path / "locks")
    return tmp_path / "locks"


@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / "retrain_state.json"
    monkeypatch.setattr(rt, "STATE_PATH", path)
    return path


def _panel(n_days=800, drift=False, per_day=12) -> tuple[pd.DataFrame, list[str]]:
    """A panel whose final 60 *calendar* days optionally sit in a new regime.

    `per_day` matters: `decide` only evaluates drift once the live window holds
    more than 200 rows, and 60 calendar days is only ~42 trading days.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    cutoff = dates.max() - pd.Timedelta(days=60)

    rows = []
    for date in dates:
        centre = 5.0 if (drift and date >= cutoff) else 0.0
        for _ in range(per_day):
            rows.append({"date": date, "feat": rng.normal(centre, 1.0)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True), ["feat"]


class TestRetrainDecision:
    def test_no_history_triggers_a_full_retrain(self, state):
        decision = rt.decide()
        assert decision.should_run is True
        assert decision.mode == "full"
        assert "No recorded training" in decision.reason

    def test_recent_full_retrain_holds(self, state):
        rt.record_retrain("full")
        decision = rt.decide()
        assert decision.should_run is False
        assert decision.mode == "none"

    def test_full_retrain_subsumes_incremental(self, state):
        rt.record_retrain("full")
        saved = json.loads(state.read_text())
        assert "last_incremental" in saved

    def test_significant_drift_forces_a_full_retrain(self, state):
        """Drift must override a healthy cadence: the calendar does not know
        when the market changed."""
        rt.record_retrain("full")
        frame, features = _panel(drift=True)
        decision = rt.decide(frame, features)
        assert decision.should_run is True
        assert decision.mode == "full"
        assert "PSI" in decision.reason

    def test_stable_data_does_not_trigger_drift_retrain(self, state):
        rt.record_retrain("full")
        frame, features = _panel(drift=False)
        assert rt.decide(frame, features).should_run is False


class TestScheduler:
    def test_every_job_has_a_cron_entry(self):
        assert set(JOBS) == set(CRON_LINES)

    def test_spec_responsibilities_are_all_registered(self):
        expected = {
            "resolve_outcomes",
            "post_mortem",
            "poll_sentiment",
            "drift_check",
            "refresh_panel",
            "retrain",
            "monitoring",
        }
        assert expected <= set(JOBS)

    def test_cron_file_contains_one_line_per_job(self):
        text = cron_file(python="/usr/bin/python3", cwd="/tmp/app")
        for name in JOBS:
            assert f"--run {name}" in text

    def test_unknown_job_is_rejected(self):
        with pytest.raises(KeyError, match="unknown job"):
            run_job("does_not_exist")

    def test_failing_job_is_reported_not_raised(self, monkeypatch):
        """A failed scheduled job must produce a record, not kill the runner."""

        def boom():
            raise RuntimeError("kaboom")

        monkeypatch.setitem(JOBS, "resolve_outcomes", JOBS["resolve_outcomes"].__class__(
            "resolve_outcomes", "daily", "test", boom
        ))
        result = run_job("resolve_outcomes")
        assert result["status"] == "failed"
        assert "kaboom" in result["result"]["error"]


def test_cron_file_quotes_paths_containing_spaces():
    text = cron_file(python="/opt/my env/bin/python", cwd="/home/me/My Projects/evaluator")

    line = next(line for line in text.splitlines() if "--run score_universe" in line)
    assert "cd '/home/me/My Projects/evaluator'" in line
    assert "'/opt/my env/bin/python' -m evaluator.scheduler" in line


def test_a_failed_job_exits_nonzero(monkeypatch, capsys):
    from evaluator import scheduler

    def boom():
        raise RuntimeError("nope")

    monkeypatch.setitem(JOBS, "boom", scheduler.Job("boom", "never", "always fails", boom))
    monkeypatch.setattr(sys, "argv", ["scheduler", "--run", "boom"])

    with pytest.raises(SystemExit) as exit_info:
        scheduler.main()

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_an_unknown_job_name_is_a_usage_error_not_a_traceback(monkeypatch, capsys):
    from evaluator import scheduler

    monkeypatch.setattr(sys, "argv", ["scheduler", "--run", "no_such_job"])

    with pytest.raises(SystemExit) as exit_info:
        scheduler.main()

    assert exit_info.value.code == 2
    assert "score_universe" in capsys.readouterr().err


def test_every_job_has_a_cron_schedule_and_every_schedule_a_job():
    # A job without a schedule is never run by --install-cron; a schedule without
    # a job writes a crontab line that fails every night.
    assert set(CRON_LINES) == set(JOBS)


def test_a_job_already_running_is_skipped_not_stacked(monkeypatch):
    from evaluator import scheduler

    calls = []
    monkeypatch.setitem(JOBS, "slow", scheduler.Job("slow", "never", "slow", lambda: calls.append(1) or {}))

    with scheduler._exclusive("slow") as acquired:
        assert acquired
        outcome = run_job("slow")

    assert outcome["status"] == "skipped"
    assert calls == []
    # Once the first run finishes the job is runnable again.
    assert run_job("slow")["status"] == "ok"
    assert calls == [1]


def test_list_shows_the_cron_expression_beside_each_job(monkeypatch, capsys):
    from evaluator import scheduler

    monkeypatch.setattr(sys, "argv", ["scheduler", "--list"])
    scheduler.main()

    out = capsys.readouterr().out
    assert CRON_LINES["score_universe"] in out
    assert all(name in out for name in JOBS)
