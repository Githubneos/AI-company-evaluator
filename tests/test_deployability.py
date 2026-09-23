"""Is a model worth serving at all, whatever today's market is doing?"""

import json

import pytest

from evaluator.model import deployability
from evaluator.model.deployability import assess, is_deployable


def _reports(tmp_path, target, *, vs_vol, vs_har_range):
    """Write the two reports the verdict reads, with the given intervals."""
    directory = tmp_path / target
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "baselines.json").write_text(json.dumps({
        "target": target, "model_schema": "e8fe94da6605",
        "incremental": {
            "vs_vol": {"pooled": vs_vol[0], "ci90": list(vs_vol[1]), "fold_wins": vs_vol[2], "n_folds": 17},
            "vs_earnings": {"pooled": 0.02, "ci90": [0.01, 0.03], "fold_wins": 16, "n_folds": 17},
        },
    }))
    (directory / "vol_benchmarks.json").write_text(json.dumps({
        "target": target,
        "benchmarks": {
            "har": {"model_edge": {"pooled": 0.012, "ci90": [0.005, 0.018], "fold_wins": 12, "n_folds": 17}},
            "har_range": {"model_edge": {
                "pooled": vs_har_range[0], "ci90": list(vs_har_range[1]),
                "fold_wins": vs_har_range[2], "n_folds": 17}},
        },
    }))
    return directory


@pytest.fixture
def production(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluator.model.registry.PRODUCTION_DIR", tmp_path)
    return tmp_path


def test_a_model_beaten_by_a_simpler_one_is_not_deployable(production):
    # The interval lies entirely below zero: range-HAR won, and the folds agree.
    _reports(production, "direction_1d", vs_vol=(0.004, (0.001, 0.008), 11),
             vs_har_range=(-0.0106, (-0.0140, -0.0071), 2))

    verdict = assess("direction_1d")

    assert verdict["deployable"] is False
    assert "har_range" in verdict["reason"]
    assert "2/17 folds" in verdict["reason"]
    assert "research output, not a signal" in verdict["reason"]


def test_a_model_nothing_simpler_beats_is_deployable(production):
    _reports(production, "magnitude_5d", vs_vol=(0.0029, (0.0005, 0.0087), 13),
             vs_har_range=(0.0168, (0.0125, 0.0211), 16))

    verdict = assess("magnitude_5d")

    assert verdict["deployable"] is True
    assert all(check["measured"] and not check["dominated"] for check in verdict["checks"])


def test_a_losing_point_estimate_whose_interval_straddles_zero_is_a_tie(production):
    """A tie is not grounds to withdraw a model -- only a measured loss is."""
    _reports(production, "direction_5d", vs_vol=(-0.0046, (-0.0112, 0.0019), 7),
             vs_har_range=(-0.0005, (-0.0060, 0.0052), 6))

    verdict = assess("direction_5d")

    assert verdict["deployable"] is True
    assert verdict["reason"].startswith("No simpler alternative")


def test_a_model_with_no_reports_is_unmeasured_not_condemned(production):
    (production / "magnitude_1d").mkdir(parents=True)

    verdict = assess("magnitude_1d")

    assert verdict["deployable"] is True
    assert "unmeasured" in verdict["reason"]
    assert all(check["measured"] is False for check in verdict["checks"])


def test_the_worst_comparison_decides_not_the_average(production):
    """One clear loss is enough, however well the model does elsewhere."""
    _reports(production, "direction_1d", vs_vol=(0.02, (0.015, 0.025), 17),
             vs_har_range=(-0.011, (-0.015, -0.007), 2))

    assert assess("direction_1d")["deployable"] is False


def test_the_flag_is_read_from_the_recorded_verdict(production, monkeypatch):
    """Serving reads what promotion decided, not a fresh recomputation."""
    directory = production / "direction_1d"
    directory.mkdir(parents=True)
    (directory / "deployability.json").write_text(json.dumps({"deployable": False, "reason": "beaten"}))

    assert is_deployable("direction_1d") is False
    # Absent verdict: an absence of evidence is not a withdrawal.
    assert is_deployable("never_assessed") is True


def test_only_a_whole_interval_below_zero_counts_as_dominated():
    assert deployability.CEILING == 0.0
