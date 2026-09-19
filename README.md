# AI Company Evaluator

Forward price-movement risk scoring from market data, SEC filing events, and
news sentiment, with an LLM reasoning layer and a feedback loop that scores its
own predictions. Full design in [docs/SPEC.md](docs/SPEC.md).

**Research tooling, not investment advice.**

```
SEC EDGAR events ─┐
price / macro ────┼─→ features ─→ 6 GBM models ─┐
fundamentals ─────┘        (purged walk-forward) │
                                                 ├─→ fusion ─→ LLM ─→ evaluation
news ─→ FinBERT ─→ sentiment ────────────────────┤   payload
                            analog retrieval ────┤
                                                 │
        prediction log ─→ outcomes ─→ post-mortem┘
```

## What's built

All eight phases of the spec, at a scale that runs on a laptop.

| Phase | Status |
|---|---|
| 1 Data + event taxonomy | 503 S&P names, 2.56M price rows, 237k events from 8-K item codes (1993–2026) |
| 2 Features | 44 features: technical, sector-relative, event-derived, fundamentals, macro/regime |
| 3 Models | 6 LightGBM models (direction + magnitude × 1/5/20d), purged walk-forward, per-regime validation, Optuna, XGBoost challenger, TreeSHAP, analog retrieval |
| 4 Sentiment | FinBERT over yfinance + EDGAR 8-K feeds, credibility/recency weighting, staleness flags, divergence detection |
| 5 Fusion | Rules-based payload assembly; optional learned meta-model |
| 6 Feedback | Prediction log, outcome resolution, rules-then-LLM post-mortem tagging, PSI drift, retrain cadence |
| 7 Serving | FastAPI, Gemini/Anthropic providers, job scheduler with cron generation, MLflow (local) |
| 8 Monitoring | Calibration, AUC-PR, precision@k, PSI dashboards, feedback health, regime-aware promotion gate |

Lightweight equivalents replace the heavyweight infrastructure the spec names —
versioned Parquet instead of Feast, local SQLite-backed MLflow instead of a
server, a job registry + cron instead of Airflow, polling instead of Kafka. The
seams are where a real orchestrator would attach.

## Results

The first single-ticker model had **no skill** (Brier skill −0.011, AUC 0.526).
Cross-sectional training on 503 names, SEC event features, and a magnitude
target changed that.

All six models, 17 purged walk-forward folds spanning **2009–2026**:

| target | Brier skill | AUC | p@5% | base rate | negative folds |
|---|---|---|---|---|---|
| magnitude_1d | **+0.0435** | 0.627 | 0.519 | 0.260 | 1/17 |
| magnitude_5d | +0.0382 | 0.624 | 0.485 | 0.280 | **0/17** |
| magnitude_20d | +0.0292 | 0.609 | 0.442 | 0.303 | 2/17 |
| direction_20d | +0.0171 | 0.574 | 0.302 | 0.192 | 2/17 |
| direction_1d | +0.0158 | 0.590 | 0.184 | 0.142 | 1/17 |
| direction_5d | +0.0156 | 0.586 | 0.259 | 0.164 | 3/17 |

Magnitude beats direction at every horizon — volatility is autocorrelated while
short-horizon direction is close to a martingale. `magnitude_1d` reaches a 2×
lift on its highest-conviction 5% of predictions.

### Per-regime skill — the table that actually decides deployability

| target | covid_crash | gfc | covid_recovery | recovery | recent | rate_hikes |
|---|---|---|---|---|---|---|
| magnitude_1d | +0.2514 | — | +0.0674 | +0.0345 | +0.0344 | +0.0295 |
| magnitude_5d | +0.0950 | — | +0.0611 | +0.0384 | +0.0280 | +0.0179 |
| magnitude_20d | +0.0877 | +0.0898 | +0.0305 | +0.0277 | +0.0302 | +0.0119 |
| direction_1d | +0.0869 | +0.0375 | +0.0284 | +0.0156 | +0.0068 | **−0.0101** |
| direction_5d | +0.1068 | +0.0881 | +0.0124 | +0.0143 | +0.0173 | **−0.0158** |
| direction_20d | +0.1781 | +0.0085 | +0.0233 | +0.0136 | +0.0128 | **−0.0152** |

Two findings the pooled column hides completely:

**All three direction models lose money in the 2022–23 rate-hike regime**, at
every horizon, across 17 folds. That is systematic, not sampling noise, and it
is the single strongest argument for the per-regime discipline in spec 3.3 — on
aggregate all three look mildly positive. The promotion gate refuses candidates
that regress in any regime for this reason.

**Every model is strongest in crisis regimes** — the COVID crash and the GFC are
the best-scoring windows across the board. For a risk screen that is the right
way round: spec 3.3's fear is a model that only works in benign markets, and
this is the inverse.

An earlier 5-fold run (2021–2026 only) showed rate-hike negatives for the
magnitude models too. Extending to 17 folds removed them, so those were a
small-sample artifact — the direction ones were not.

**Skill, not accuracy, is the number that matters.** Accuracy near 70% is what
you get by always predicting "no large move". Every metric is reported against a
baseline predicting historical class priors, computed per fold from that fold's
own training window. The fusion payload states in plain language whether a model
clears it, and the LLM is instructed to say so rather than narrate noise.

### Against stronger baselines — most of the skill is volatility

Class priors are the weakest possible baseline. `scripts.evaluate_baselines`
fits three small models on the **same 17 folds and the identical out-of-sample
rows** (a guard refuses the comparison otherwise): volatility only (`vol_20`,
`vol_60`, `vol_ratio_20_60`, `atr_14_pct`), earnings cycle only (days since the
last earnings release and periodic report), and both plus VIX. Each 44-feature
model is then compared with whichever baseline is strongest on its target.

| target | model | vol only | earnings only | vol + earn + VIX | edge over best | 90% CI (folds) | folds won |
|---|---|---|---|---|---|---|---|
| magnitude_1d | +0.0435 | +0.0371 | +0.0070 | +0.0391 | +0.0045 | +0.0006..+0.0086 | 11/17 |
| magnitude_5d | +0.0382 | +0.0308 | +0.0132 | +0.0354 | +0.0029 | −0.0024..+0.0087 | 7/17 |
| magnitude_20d | +0.0292 | +0.0284 | +0.0145 | +0.0250 | +0.0009 | −0.0071..+0.0085 | 12/17 |
| direction_1d | +0.0158 | **+0.0253** | +0.0037 | +0.0090 | **−0.0098** | −0.0136..−0.0058 | 2/17 |
| direction_5d | +0.0156 | **+0.0201** | +0.0077 | +0.0136 | −0.0046 | −0.0112..+0.0019 | 7/17 |
| direction_20d | +0.0171 | +0.0169 | +0.0093 | +0.0136 | +0.0002 | −0.0076..+0.0099 | 6/17 |

All columns except "edge" are Brier skill against class priors. "Edge" is the
model's Brier skill *over* the best baseline, with a 90% interval from
resampling folds (rows within a fold are not independent).

What this says, plainly:

- **The headline skill is mostly volatility clustering.** A four-feature
  volatility model reaches 81–99% of the skill of the four models it doesn't
  beat outright, and it beats the other two. The remaining 40 features (events,
  fundamentals, sector-relative returns, technicals, macro) add at most +0.0045
  Brier on any target.
- **Only `magnitude_1d` has an edge whose interval excludes zero**, and it is
  small (+0.0045) and inconsistent (11/17 folds).
- **`direction_1d` is dominated.** The volatility-only model beats it in 15/17
  folds, and the interval lies entirely below zero. `direction_5d` loses too.
  The full model is worse than a subset of its own inputs: the extra features
  are fitting noise.
- **The earnings clock alone is weak** (+0.004 to +0.015). It often ranks high
  in the SHAP attributions, but that prominence does not translate into
  standalone predictive power.
- **More features made the small model worse on four targets.** Adding the
  earnings and VIX features to the volatility set lowers skill for every
  direction model and for `magnitude_20d`. VIX level is the likelier culprit
  (a non-stationary level that does not generalise across regimes), but this
  run does not isolate it.

Consequence: until a feature set beats the volatility baseline, treat these
models as volatility forecasters. The next work is feature pruning (start by
dropping VIX level and testing whether the event and fundamentals blocks earn
their place, with VIX ablated on its own), not more features.

```bash
python -m scripts.evaluate_baselines --targets magnitude_1d   # ~2 min, ~0.7 GB
```

## Known defects

- **Survivorship bias.** The universe is today's S&P 500. Firms that failed or
  were removed are absent, so downside frequencies are a floor, not an estimate.
  Not fixable in modelling — it needs CRSP delisting data (spec 1.2).
- **Four event categories are unpopulated.** Guidance, litigation, dividend, and
  rating events cannot be derived from 8-K item codes. They are left empty and
  flagged rather than approximated, because labels that look complete and are
  wrong are worse than a visible gap.
- **News relevance is weak.** yfinance returns loosely-related market news, not
  strictly company-specific coverage. A paid feed (Benzinga, Bloomberg) would
  fix this.
- **No historical sentiment.** Free feeds return only recent headlines, so
  sentiment cannot be backtested and the fusion meta-model has no sentiment
  column to learn from.
- **Effective sample size is regimes, not rows.** Twenty years contains ~8
  independent market environments. Per-regime validation exists for this reason,
  and training strides dates (keeping every Nth full cross-section) rather than
  pretending 2.5M overlapping-window rows are independent observations.
- **Direction models are not deployable as-is.** All three are negative through
  the rate-hike regime. Use the magnitude models; treat direction output as
  research only.
- **No transaction costs or liquidity modelling.** Scores are not tradeable as-is.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" -c requirements.lock
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY; LLM_PROVIDER picks
```

Keys are read from the environment only — never written to source, config,
artifacts, or the prediction database.

`requirements.lock` pins the exact versions behind the results above. Install
with it: a LightGBM or pandas upgrade can move every metric without any code
change. After deliberately upgrading, regenerate it (`pip freeze
--exclude-editable`, keeping the header), retrain, and re-run the baselines.

## Development

```bash
pytest -q          # ~20 s, no network or trained artifacts needed
ruff check .       # lint (config in pyproject.toml)
```

CI (`.github/workflows/ci.yml`) runs both on every pull request and on pushes
to `main`, installing from `requirements.lock` with CPU-only torch.

## Build the system

```bash
python -m scripts.backfill        # prices + EDGAR events  (~90 min)
python -m scripts.build_panel     # feature panel          (~45 min)
python -m scripts.train --n-splits 17 --max-train-rows 320000   # 6 models
python -m scripts.build_analogs   # historical analog index
uvicorn evaluator.serving.app:app --reload
```

## API

```
GET  /                            browser dashboard
GET  /score/{ticker}              raw model output, all targets
GET  /payload/{ticker}            fusion payload, no LLM call and no cost
GET  /evaluate/{ticker}           full pipeline through the written evaluation
GET  /sentiment/{ticker}          news sentiment with staleness flag
GET  /analogs/{ticker}            closest historical situations and outcomes
GET  /prices/{ticker}             recent daily closes, for charting
GET  /model/{target}/validation   per-fold and per-regime metrics, calibration, baselines
GET  /monitoring                  feedback health, drift, post-mortem tags
POST /feedback/resolve            close out elapsed prediction windows
```

`/evaluate` logs the prediction *before* calling the LLM, so the record survives
a reasoning-layer failure. An unlogged prediction cannot be scored later, and
that is the one loss this system cannot recover from.

## Operations

```bash
python -m evaluator.scheduler --list           # registered jobs and cadences
python -m evaluator.scheduler --install-cron   # crontab to install
python -m scripts.retrain --check              # drift + cadence decision
python -m scripts.postmortem                   # tag wrong predictions
python -m scripts.promote --all                # regime-aware promotion gate
python -m scripts.monitoring_report            # static HTML dashboard
```

The promotion gate refuses a candidate that improves on average but regresses in
any single regime (spec 8.3), and refuses any model without measured skill —
beating a bad incumbent is not evidence of being useful.

## Layout

```
evaluator/
  data/          universe, price loaders, EDGAR client, XBRL fundamentals
  events.py      8-K item code → taxonomy (modern + pre-2004 schemes)
  features/      feature definitions (shared by train and serve) + Parquet store
  labels.py      volatility-scaled direction and magnitude labels
  validation.py  purged, embargoed walk-forward — row-wise and date-wise
  metrics.py     skill-vs-baseline, calibration, AUC-PR, precision@k
  regimes.py     NBER recessions and named market regimes
  model/         training, prediction + TreeSHAP, analogs, Optuna/XGBoost
  sentiment/     news feeds, FinBERT, aggregation, divergence
  fusion.py      payload assembly;  fusion_model.py  learned meta-model
  feedback/      prediction log, outcome resolution, post-mortem tagging
  monitoring.py  PSI drift, feedback health;  retrain.py  cadence + triggers
  scheduler.py   job registry (the Airflow substitute)
  llm/           provider abstraction (Gemini, Anthropic) + system prompt
  serving/app.py FastAPI
```

## Tests

```bash
.venv/bin/python -m pytest
```

150 tests, no API key or network required.

The leakage suites are the ones that matter, because a leak fails nothing — it
just makes every metric downstream wrong:

- `test_leakage.py` — recomputing features on truncated history must give
  identical values; event features must ignore events dated after the row.
- `test_fundamentals.py` — quarterly figures join on *filing* date, not fiscal
  period end; restatements must not rewrite what was known at the time.
- `test_events.py` — 8-Ks accepted after the 16:00 ET close belong to the next
  session.
- `test_validation.py` — no date may straddle a train/test boundary in a panel
  where 500 tickers share each day.
- `test_feedback.py` — outcomes are scored against the volatility scale recorded
  when the prediction was made.
