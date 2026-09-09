"""CLI: model promotion gate (spec 8.3).

    python -m scripts.promote --target magnitude_5d
    python -m scripts.promote --all --apply

A candidate is only promoted if it improves on the incumbent *and* does not
regress materially in any single regime. Spec 8.3 is explicit about why: a model
can improve on average while getting worse in high-volatility regimes, which is
exactly when accuracy matters most. Aggregate metrics hide that trade, so the
gate checks regimes individually and refuses on any one of them.

The other refusal is absolute: a candidate with no measured skill over its
baseline is never promoted, however much better it looks than the incumbent.
Beating a bad model is not evidence of being a useful one.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from evaluator.model.train import MODEL_DIR

log = logging.getLogger(__name__)

PRODUCTION_DIR = Path(MODEL_DIR).parent / "production"
REGRESSION_TOLERANCE = 0.005
MIN_SKILL = 0.0


def _load(directory: Path) -> dict | None:
    path = directory / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


def compare(candidate: dict, incumbent: dict | None) -> dict:
    """Decide whether `candidate` may replace `incumbent`."""
    cand_pooled = candidate["validation"]["pooled_out_of_sample"]
    cand_skill = cand_pooled.get("brier_skill")

    if cand_skill is None or cand_skill <= MIN_SKILL:
        return {
            "promote": False,
            "reason": (
                f"Candidate has no measured skill over baseline "
                f"(Brier skill {cand_skill}). A model that cannot beat the base "
                "rate is not promotable regardless of the incumbent."
            ),
            "candidate_skill": cand_skill,
        }

    if incumbent is None:
        return {
            "promote": True,
            "reason": f"No incumbent; candidate has positive skill ({cand_skill:+.4f}).",
            "candidate_skill": cand_skill,
        }

    inc_pooled = incumbent["validation"]["pooled_out_of_sample"]
    inc_skill = inc_pooled.get("brier_skill") or -9.0

    if cand_skill <= inc_skill:
        return {
            "promote": False,
            "reason": f"Candidate skill {cand_skill:+.4f} does not beat incumbent {inc_skill:+.4f}.",
            "candidate_skill": cand_skill,
            "incumbent_skill": inc_skill,
        }

    # Per-regime check: this is the gate that aggregate metrics cannot express.
    cand_regimes = candidate["validation"].get("by_regime", {})
    inc_regimes = incumbent["validation"].get("by_regime", {})
    regressions = []
    for regime, inc_metrics in inc_regimes.items():
        cand_metrics = cand_regimes.get(regime)
        if not cand_metrics:
            continue
        before = inc_metrics.get("brier_skill")
        after = cand_metrics.get("brier_skill")
        if before is None or after is None:
            continue
        if after < before - REGRESSION_TOLERANCE:
            regressions.append(
                {"regime": regime, "incumbent": round(before, 4), "candidate": round(after, 4)}
            )

    if regressions:
        worst = min(regressions, key=lambda r: r["candidate"] - r["incumbent"])
        return {
            "promote": False,
            "reason": (
                f"Candidate improves overall ({inc_skill:+.4f} -> {cand_skill:+.4f}) but "
                f"regresses in {len(regressions)} regime(s), worst {worst['regime']}: "
                f"{worst['incumbent']:+.4f} -> {worst['candidate']:+.4f}. "
                "Spec 8.3 refuses aggregate-only promotion."
            ),
            "candidate_skill": cand_skill,
            "incumbent_skill": inc_skill,
            "regressions": regressions,
        }

    return {
        "promote": True,
        "reason": f"Improves overall ({inc_skill:+.4f} -> {cand_skill:+.4f}) with no regime regression.",
        "candidate_skill": cand_skill,
        "incumbent_skill": inc_skill,
    }


def promote(target: str, *, apply: bool = False) -> dict:
    candidate_dir = Path(MODEL_DIR) / target
    production_dir = PRODUCTION_DIR / target

    candidate = _load(candidate_dir)
    if candidate is None:
        return {"target": target, "promote": False, "reason": f"No candidate model at {candidate_dir}."}

    verdict = compare(candidate, _load(production_dir))
    verdict["target"] = target

    if verdict["promote"] and apply:
        production_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.txt", "metadata.json"):
            shutil.copy2(candidate_dir / name, production_dir / name)
        verdict["applied"] = True

    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true", help="actually copy to production")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from evaluator.model.predict import available_targets

    targets = [args.target] if args.target else available_targets()
    if not targets:
        raise SystemExit("no trained models found")

    results = [promote(t, apply=args.apply) for t in targets]
    for verdict in results:
        mark = "PROMOTE" if verdict["promote"] else "HOLD   "
        print(f"{mark}  {verdict['target']:<16} {verdict['reason']}")

    if not args.apply:
        print("\n(dry run -- pass --apply to copy approved models to production)")


if __name__ == "__main__":
    main()
