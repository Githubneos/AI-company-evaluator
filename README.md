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
| 1 Data + event taxonomy | 503 S&P names, 2.56M price rows, 237k events from 8-K item codes (1993–2026), plus 12.3k dividend events from the payment series |
| 2 Features | 47 features: technical, sector-relative, event-derived (8-K and dividends), fundamentals, macro/regime |
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
  direction model and for `magnitude_20d`. The ablation study below traces this
  to the macro group as a whole: dropping the VIX level alone does not help.

Consequence: until a feature set beats the volatility baseline, treat these
models as volatility forecasters. The ablation study below shows where to cut.

```bash
python -m scripts.evaluate_baselines --targets magnitude_1d   # ~2 min, ~0.7 GB
```

### Which feature groups earn their place

`scripts.evaluate_ablations` adds one feature group at a time to the four
volatility features, fits small trees on the same 17 folds, and scores each set
on the model's own out-of-fold rows. A group **earns its place** only if its
edge over volatility alone is at least +0.002 Brier, its interval excludes zero
after a Bonferroni correction across the seven comparisons (98.6% intervals),
and it wins at least 70% of folds. Without those guards, a pure-noise group
cleared a plain 90% interval in synthetic tests, so they are enforced in code
and covered by a five-seed test.

Brier edge over volatility alone (✓ earns its place, ✗ hurts: interval below zero):

| group added | magnitude_1d | magnitude_5d | magnitude_20d | direction_1d | direction_5d | direction_20d |
|---|---|---|---|---|---|---|
| SEC events | **+0.0051 ✓** | **+0.0125 ✓** | **+0.0145 ✓** | −0.0005 | **+0.0071 ✓** | **+0.0096 ✓** |
| technical & returns | **+0.0090 ✓** | +0.0041 | −0.0017 | +0.0044 | +0.0016 | −0.0020 |
| sector-relative | −0.0011 | −0.0011 | −0.0016 | −0.0035 ✗ | −0.0013 | −0.0019 ✗ |
| fundamentals | −0.0024 ✗ | −0.0005 | −0.0022 | −0.0041 ✗ | −0.0011 | −0.0026 |
| macro & regime | −0.0039 | **−0.0097 ✗** | **−0.0205 ✗** | **−0.0205 ✗** | **−0.0146 ✗** | −0.0078 |
| macro without VIX level | −0.0048 | −0.0095 ✗ | −0.0202 ✗ | −0.0197 ✗ | −0.0154 ✗ | −0.0057 |
| all 44, small trees | +0.0070 | +0.0068 ✓ | −0.0003 | −0.0096 ✗ | −0.0043 | +0.0026 |
| *production model* | *+0.0066* | *+0.0077 ✓* | *+0.0009* | *−0.0098 ✗* | *−0.0046* | *+0.0002* |

What this says:

- **SEC event timing is the one group with real, consistent value.** It earns
  its place on five of six targets. It is also the only group that consistently beats
  the production model: volatility + events alone (14 features, small trees)
  scores +0.0429 vs +0.0382 on `magnitude_5d`, +0.0424 vs +0.0292 on
  `magnitude_20d`, +0.0270 vs +0.0156 on `direction_5d`, and +0.0264 vs
  +0.0171 on `direction_20d`.
- **Macro features actively hurt.** They lower skill on all six targets, significantly on
  four, by up to −0.02 Brier. Removing the VIX level does not fix it. That corrects the hypothesis in the
  section above. The likely mechanism is regime-level features that
  identify *when* a training year happened rather than anything that
  generalises forward, but that is an interpretation; the measurement is the harm.
- **Fundamentals and sector-relative features add nothing and sometimes hurt.**
- **Depth is not the problem.** All 44 features with small trees score about the
  same as the deep production model, so the harm comes from the feature
  groups, not from overfitting capacity.
- **Technical features help only `magnitude_1d`.**

The recommended feature sets (`recommended_features` in each target's
`ablations.json`) are volatility + events, plus technical for `magnitude_1d`,
and volatility alone for `direction_1d`. Retraining on these goes through the
promotion gate in the planned rebuild; nothing is swapped into production on
the strength of this table alone.

```bash
python -m scripts.evaluate_ablations --targets magnitude_20d   # ~8 min, ~1.5 GB
```

### Versus standard volatility models

If these are volatility forecasters, the benchmark is the volatility
literature, not class priors. `scripts.evaluate_vol_benchmarks` forecasts the
next h days' variance relative to the trailing variance the label is scaled
by, calibrates it to probabilities on each fold's training rows, and scores it
on the model's own out-of-sample rows (99.999% price coverage on all six targets):

- **HAR-RV** (Corsi 2009) on close-to-close realised variance
- **HAR-RV (range)**, the same regression on Parkinson range variance,
  (ln H/L)² / (4 ln 2)
- **GARCH(1,1)**, Gaussian MLE per ticker per fold

| target | model | HAR-RV | GARCH(1,1) | HAR-RV (range) | model edge over the best benchmark (bold) |
|---|---|---|---|---|---|
| magnitude_1d | +0.0435 | +0.0129 | +0.0135 | **+0.0354** | +0.0084 (12/17 folds, CI +0.0001..+0.0161) |
| magnitude_5d | +0.0382 | +0.0056 | +0.0147 | **+0.0217** | +0.0168 (16/17, CI +0.0125..+0.0211) |
| magnitude_20d | +0.0292 | +0.0028 | **+0.0212** | +0.0163 | +0.0082 (12/17, CI −0.0011..+0.0171) |
| direction_1d | +0.0158 | +0.0101 | +0.0092 | **+0.0261** | **−0.0106 (2/17, CI −0.0140..−0.0071)** |
| direction_5d | +0.0156 | +0.0050 | +0.0116 | **+0.0161** | −0.0005 (6/17, CI −0.0060..+0.0052) |
| direction_20d | +0.0171 | +0.0018 | **+0.0144** | +0.0111 | +0.0028 (9/17, CI −0.0039..+0.0102) |

**The intraday range is where the signal is.** A diagnostic shipped in the same
report (`single_feature_diagnostics`) scores one panel feature at a time
through the same folds and calibration. On `magnitude_1d`: trailing volatility
level +0.0045, the close-to-close 20d/60d volatility ratio +0.0075, and **ATR
divided by vol_60 +0.0383** — on its own, 88% of the production model's skill.
On the direction targets that single feature is *better than the model*:
+0.0285 vs +0.0158 at 1 day and +0.0230 vs +0.0156 at 5 days. Close-to-close
HAR and GARCH cannot see that range, which is why they look weak (+0.003 to
+0.021) and why the range-based HAR, with three coefficients, is the benchmark
that matters.

Against that benchmark:

- **The magnitude models earn their keep.** They beat range-HAR at all three
  horizons, clearly so at 5 and 20 days (+0.0168 in 16/17 folds, +0.0132 in
  13/17).
- **`direction_1d` is beaten by it**, winning 2 of 17 folds with an interval
  entirely below zero. A three-parameter regression on the high-low range
  forecasts 1-day direction classes better than the 44-feature model does.
- **`direction_5d` ties it**, and `direction_20d`'s edge over GARCH, its best
  benchmark, does not clear zero either.

Together with the ablations, the direction models now have two independent
reasons to be treated as research-only: a four-feature volatility model beats
`direction_1d`, and so does range-HAR.

```bash
python -m scripts.evaluate_vol_benchmarks --targets magnitude_1d   # ~3 min, ~1.1 GB
```

### Survivorship: historical membership, and what free data cannot fix

The universe was "today's S&P 500, backfilled", which quietly assumes today's
members were always members and that nothing ever left.
`data/universe/sp500_changes.csv` (407 index changes back to 1976, fetched from
Wikipedia and checked in) fixes the first half: `membership_intervals()`
reconstructs when each of 858 names was actually in the index by walking the
change history backwards from today's constituents, and the panel keeps only
rows inside those intervals.

Spot-checked against history: Lehman Brothers leaves 2008-09-16, Twitter
2018–2022, Dell leaves in 2013 when taken private and returns in 2024. 28 names
have more than one spell.

The second half is where free data runs out. Of 343 removed names inside the
panel's span:

| outcome | names | |
|---|---|---|
| recovered and used | 97 | all still-listed names the index merely dropped |
| no data at all | 200 | 129 of them genuine delistings |
| **rejected: ticker reused** | 35 | all genuine delistings |
| too little history in-window | 11 | |

**Not one genuinely delisted company was recoverable.** yfinance returns
nothing for Lehman, Twitter or Activision. Worse, for 35 dead symbols it
returns a *different* company that trades under them today: asking for AV
(Avaya, acquired 2007), WB (Wachovia, 2008) or SGP (Schering-Plough, 2009)
yields continuous data through 2026. Accepting it would have put the wrong
company's prices into the panel labelled as a failed one — a bias that looks
like better data. `evaluator/data/delisted.py` rejects any name whose removal
reason implies it stopped trading but whose series runs more than 25 days past
the removal date.

So the panel grows from 503 to 600 names and loses its "always a member"
assumption, but the companies that actually failed are still missing, and that
is the part that matters most for downside risk.

```bash
python -m scripts.fetch_universe_changes --apply   # the change history
python -m scripts.backfill --what delisted         # what can be recovered
```

### Dividend events, from the payment series

8-K item codes cannot express a dividend decision, so `DIVIDEND_CHANGE` sat in
the unpopulated list. The payment series can, and it is free:
`evaluator/data/dividends.py` derives initiations, raises, cuts, suspensions
and resumptions from yfinance's dividend history and joins them to the same
event table.

Two dating rules keep this causal:

- Events sit on the **ex-dividend date**. The announcement comes earlier, often
  by weeks, so a feature saying "a cut happened N days ago" refers to
  information the market already had.
- A **suspension has no payment to sit on**, so it is dated when a payment the
  company's own rhythm predicted failed to appear (1.5 median intervals after
  the last one), never at the last payment or at the eventual resumption.

Across the universe: **422 of 503 names pay dividends, giving 12,304 events**
(10,253 raises, 1,006 cuts, 725 initiations/resumptions, 320 suspensions),
7,545 of them inside the panel's 2005– span, and none dated in the future.
Cuts peak in 2020, matching the COVID wave.

Spot-checked against real history: Apple pays from 1987, stops (suspension
detected April 1996), resumes August 2012; Agilent initiates in 2012; AbbVie
initiates at its 2013 spin-off.

New features: `days_since_dividend_cut`, `days_since_dividend_raise` and
`dividend_cuts_365d`. They reach the models at the next panel rebuild.

## Known defects

- **Survivorship bias: reduced, not fixed.** The panel now uses historical index
  membership and adds back 97 removed names, but **no genuinely delisted company
  could be recovered** from free data. Downside frequencies remain a floor. See
  below; a real fix still needs CRSP delisting data (spec 1.2).
- **Three event categories are unpopulated.** Guidance, litigation and rating
  events cannot be derived from 8-K item codes. They are left empty and flagged
  rather than approximated, because labels that look complete and are wrong are
  worse than a visible gap. (Dividends are populated now: see below.)
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
python -m scripts.train --n-splits 17 --max-train-rows 320000   # 6 candidates
python -m scripts.promote --apply # gate, then copy to production
python -m scripts.build_analogs   # historical analog index
uvicorn evaluator.serving.app:app --reload
```

### Candidates and production

Training writes **candidates** to `artifacts/models/<target>`, overwriting the
previous run. Serving only ever reads **production**, `artifacts/production/<target>`,
which nothing but `scripts.promote` writes. Without that split the gate decides
nothing, because every retrain would already be live by the time it ran.

`scripts.promote` refuses a candidate that has no skill over its own base rate,
that does not beat the incumbent **on the rows both models scored out of fold**
(keyed by ticker and date, with a date-block bootstrap interval), or that
regresses in any single regime. Pooled skill alone is not enough: two training
runs on different universes or histories face different rows, and the easier
population wins a pooled comparison for the wrong reason. Promotion copies the
whole model directory, so the analysis reports travel with the model, and swaps
it into place so a reader never sees a half-written directory.

```bash
python -m scripts.promote                 # dry run: verdict and reasons
python -m scripts.backfill_oos_tickers --apply   # one-off, for models trained before keys were recorded
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
