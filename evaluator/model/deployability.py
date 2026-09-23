"""Whether a model is worth serving, decided from evidence already on disk.

The regime guard asks whether a model works in *today's* market. This asks the
prior question: does it beat the simple things anyone could have done instead?
Both are needed, and only one of them can be answered at request time.

Three reports already sit beside every promoted model. `baselines.json` scores
it against small models on volatility and the earnings clock; `vol_benchmarks.json`
against HAR-RV, range-HAR and GARCH(1,1). Both compare on the model's own
out-of-fold rows, with a fold-resampled interval. This module reads them and
applies one rule:

    a model is deployable unless a simpler alternative is measurably better

"Measurably" is the interval: a point estimate below zero is a maybe, while an
interval lying entirely below zero means the simpler model won and the folds
agree. That is the standard the README already applies in prose -- direction_1d
is beaten by a four-feature volatility model and by a three-coefficient
regression on the high-low range -- and the standard the system did not apply
anywhere in code. It served those models as signals regardless.

Deciding this at promotion rather than per request matters: evidence that only
surfaces when the VIX cooperates is not a guard.
"""

from __future__ import annotations

import logging

from evaluator.model.registry import load_report

log = logging.getLogger(__name__)

#: A model is only dominated when the whole interval is on the wrong side of
#: zero. A negative point estimate whose interval straddles zero is a tie, and
#: a tie is not grounds to withdraw a model.
CEILING = 0.0


def _worst(comparisons: dict[str, dict]) -> tuple[str | None, dict | None]:
    """The comparison that went worst for the model."""
    scored = {name: c for name, c in comparisons.items() if isinstance(c, dict) and "ci90" in c}
    if not scored:
        return None, None
    name = min(scored, key=lambda n: scored[n]["ci90"][1])
    return name, scored[name]


def _verdict_for(label: str, name: str | None, comparison: dict | None) -> dict:
    """Did the model survive its worst comparison in one family?"""
    if comparison is None:
        return {"family": label, "measured": False, "dominated": False,
                "note": f"No {label} report; this comparison is unmeasured, not passed."}

    lo, hi = comparison["ci90"]
    dominated = hi < CEILING
    return {
        "family": label,
        "measured": True,
        "dominated": dominated,
        "worst": name,
        "edge": comparison.get("pooled"),
        "ci90": [lo, hi],
        "fold_wins": comparison.get("fold_wins"),
        "n_folds": comparison.get("n_folds"),
    }


def assess(target: str, stage: str = "production") -> dict:
    """Read the reports beside a model and decide whether it may be served."""
    baselines = load_report(target, "baselines", stage) or {}
    benchmarks = load_report(target, "vol_benchmarks", stage) or {}

    simple_name, simple = _worst(baselines.get("incremental") or {})
    edges = {
        name: block["model_edge"]
        for name, block in (benchmarks.get("benchmarks") or {}).items()
        if isinstance(block, dict) and "model_edge" in block
    }
    vol_name, vol = _worst(edges)

    checks = [
        _verdict_for("simple baseline", simple_name, simple),
        _verdict_for("volatility benchmark", vol_name, vol),
    ]
    lost = [c for c in checks if c["dominated"]]
    deployable = not lost

    if deployable:
        measured = [c for c in checks if c["measured"]]
        reason = (
            "No simpler alternative measurably beats this model."
            if measured
            else "Deployability is unmeasured: run the baseline and volatility-benchmark reports."
        )
    else:
        worst = min(lost, key=lambda c: c["ci90"][1])
        lo, hi = worst["ci90"]
        reason = (
            f"Beaten by a simpler alternative ({worst['worst']}): edge {worst['edge']:+.4f}, "
            f"90% CI {lo:+.4f}..{hi:+.4f} lying entirely below zero, winning only "
            f"{worst['fold_wins']}/{worst['n_folds']} folds. A model that loses to "
            f"{worst['worst']} is research output, not a signal."
        )

    return {
        "target": target,
        "deployable": deployable,
        "reason": reason,
        "checks": checks,
        "model_schema": baselines.get("model_schema") or benchmarks.get("model_schema"),
    }


def is_deployable(target: str, stage: str = "production") -> bool:
    """Convenience for callers that only need the flag.

    Unmeasured counts as deployable: a missing report is an absence of
    evidence, and withdrawing a model for one would punish a fresh install.
    """
    report = load_report(target, "deployability", stage)
    return bool(report["deployable"]) if report else True


__all__ = ["assess", "is_deployable"]
