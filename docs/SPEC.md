# AI Company Evaluator — Technical Project Spec

## 0. System Overview

Three subsystems feed a fusion/reasoning layer:

```
[Historical Market Data] -> [GBM Risk Model] --\
                                                  \
[Live News Feeds] -> [Sentiment Model] -----------> [Fusion Layer] -> [LLM Evaluator] -> [Output]
                                                  /
[Outcome Tracker] -> [Feedback/Mistake Log] ----/
```

- **GBM Risk Model**: predicts probability of significant price movement (drop/spike) over a forward window, trained on 50 years of historical tick + event data.
- **Sentiment Model**: scores current news sentiment for a given ticker in near real time.
- **Feedback Loop**: logs every prediction, its outcome, and a structured "what went wrong/right" tag, feeding back into retraining and into the LLM's context at inference time.
- **Fusion Layer**: a rules/ensemble layer that combines GBM score + sentiment score + feedback context into a single structured payload.
- **LLM Evaluator**: the reasoning layer that turns the structured payload into a human-readable evaluation.

---

## Phase 1 — Historical Data Infrastructure

**Goal**: build a clean, labeled, 50-year dataset of price action + associated events.

### 1.1 Data sources
- **Price/tick data**: CRSP (Center for Research in Security Prices) for daily data back to 1970s; for intraday tick data, Polygon.io, Refinitiv Tick History, or Algoseek (CRSP doesn't do intraday). Budget-dependent — CRSP is the gold standard for survivorship-bias-free daily data.
- **Corporate events**: SEC EDGAR full-text search API (8-Ks, 10-Ks, 10-Qs) for filings-driven events; Compustat for fundamentals; Wharton Research Data Services (WRDS) as an aggregator for both CRSP and Compustat.
- **News/event labels for older data**: RavenPack (has historical event tagging back decades) or manually constructed labels from Factiva/NYT archive for pre-2000 data where structured feeds don't exist.
- **Macro overlays**: FRED (Federal Reserve Economic Data) for rates, CPI, unemployment — needed as features and as regime labels (recession/expansion).

### 1.2 Survivorship bias handling
Critical for a 50-year dataset: must include delisted companies (bankruptcy, acquisition, going private). CRSP includes delisting codes and returns — use these, don't filter them out, or the model will systematically underestimate downside risk.

### 1.3 Event taxonomy (label schema)
Define a fixed taxonomy so every historical drop/spike gets tagged. Example top-level categories:

```
EARNINGS_SURPRISE_POS / EARNINGS_SURPRISE_NEG
GUIDANCE_RAISE / GUIDANCE_CUT
MA_ANNOUNCED / MA_TERMINATED
LITIGATION_FILED / LITIGATION_RESOLVED
REGULATORY_ACTION
EXECUTIVE_CHANGE (CEO/CFO departure or hire)
DIVIDEND_CHANGE
DEBT_DOWNGRADE / DEBT_UPGRADE
MACRO_SHOCK (sector-wide or market-wide, not company-specific)
PRODUCT_EVENT (recall, launch, FDA decision, etc.)
FRAUD_ACCOUNTING
UNEXPLAINED (no identifiable public catalyst — important negative-control class)
```

Store as a normalized table: `event_id, ticker, event_date, event_type, event_subtype, source, confidence`.

### 1.4 Target variable construction
Define the label the GBM will actually predict, e.g.:
```
label = 1 if abs(forward_return[t, t+N]) > threshold else 0
```
Recommend **multiple target windows** (1-day, 5-day, 20-day) trained as separate models or as a multi-output model, since a 50-year single-window model will conflate very different dynamics (earnings-day gaps vs. multi-week drift).

Also build a **direction-specific label** (drop vs. spike vs. neutral) as a 3-class target rather than only magnitude, since "big move down" and "big move up" have different feature importances.

---

## Phase 2 — Feature Engineering

### 2.1 Price/technical features
- Rolling returns (5/20/60/252-day), realized volatility, ATR
- Volume anomalies (z-score of volume vs. 60-day average)
- Momentum indicators: RSI, MACD, Bollinger Band position
- Market-relative features: beta-adjusted return, sector-relative return (critical — raw returns are dominated by market regime)

### 2.2 Fundamental features (from Compustat)
- P/E, P/B, debt/equity, current ratio, revenue growth YoY, margin trends
- Earnings surprise history (last 4-8 quarters of surprise %, to capture "serial disappointer" patterns)

### 2.3 Event-derived features
- Time-since-last-event-by-type
- Event density (number of material events in trailing 90 days — a proxy for company "instability")
- One-hot/embedding of most recent event type

### 2.4 Macro/regime features
- VIX level and trend, yield curve slope (10y-2y), sector rotation indicators
- Recession indicator flag from FRED/NBER dating

### 2.5 Feature store
Use a **feature store** (Feast, or a simpler versioned Parquet/Delta Lake setup) so that training-time and serving-time features are computed identically — this is the single most common source of train/serve skew in these systems.

---

## Phase 3 — Gradient Boosting Model

### 3.1 Model choice
**LightGBM** as primary (faster on large tabular data, native categorical support, good with the event-type categorical features). Train an **XGBoost** model in parallel as a challenger/ensemble check — GBM implementations can meaningfully diverge in feature importance rankings, useful as a sanity check.

### 3.2 Specifics
- Objective: `binary` (per direction/window) or `multiclass` for the 3-class drop/neutral/spike target
- Key hyperparameters to tune (via Optuna, Bayesian search — grid search is too slow at this data scale):
  - `num_leaves`: 31–255
  - `max_depth`: 5–12 (constrain to avoid overfitting on 50 years of what is still a limited number of true "regime" examples)
  - `learning_rate`: 0.01–0.1 with early stopping
  - `min_data_in_leaf`: tune upward (100+) given noisy financial labels
  - `feature_fraction` / `bagging_fraction`: 0.7–0.9 to reduce overfitting to any one era's feature relationships
- **Class imbalance**: large moves are rare events — use `scale_pos_weight` or focal loss, not naive resampling (SMOTE on financial time series creates lookahead artifacts).

### 3.3 Validation methodology — this is the highest-risk part of the whole project
- **Walk-forward validation only.** Never use random k-fold on time series data — it leaks future information into training.
- Use **purged, embargoed cross-validation** (as in López de Prado's *Advances in Financial Machine Learning*): purge overlapping label windows between train/test splits, and embargo a buffer period after each test fold before the next train fold starts.
- Validate performance **separately by decade/regime** (e.g., dot-com era, 2008 GFC, 2020 COVID, 2022 rate-hike cycle) — a model that only works well in low-volatility regimes is not safe to deploy as-is.

### 3.4 Explainability
- SHAP values computed per-prediction at serving time (TreeSHAP is fast enough for real-time use with LightGBM/XGBoost)
- Store top-5 SHAP features per prediction for the LLM evaluator to reference

### 3.5 Historical analog retrieval
Separately from the GBM's numeric prediction, build a **nearest-neighbor retrieval system** over the historical event-labeled dataset:
- Embed each historical event window (price pattern + fundamentals + event type) into a vector using a simple autoencoder or even just standardized feature concatenation
- Store in a vector DB (pgvector, or FAISS if self-hosted)
- At inference time, retrieve top-k most similar historical situations and their outcomes — this is what the LLM evaluator surfaces as "closest historical analogs"

---

## Phase 4 — News Sentiment Pipeline

### 4.1 Model choice
- **FinBERT** (ProsusAI/finbert or yiyanghkust/finbert-tone) as the base sentiment classifier — domain-tuned on financial text, outperforms general sentiment models on financial news.
- For nuance beyond positive/negative/neutral (e.g., detecting sarcasm, hedged language, forward-looking statements vs. backward-looking), consider fine-tuning a **DeBERTa-v3** base model on a labeled financial-news dataset (Financial PhraseBank + your own labeled set) if FinBERT's 3-class output proves too coarse.
- For event extraction from headlines (not just sentiment polarity), a lightweight NER/relation-extraction pass can tag *what kind* of event the article describes, mapping to the same event taxonomy from Phase 1 — this lets live sentiment plug into the same historical-analog retrieval system.

### 4.2 Data sources
- News APIs: Benzinga News API, NewsAPI.org, or Bloomberg/Refinitiv if budget allows (these have much better financial-news coverage and lower latency than generic news APIs)
- SEC filings feed (8-K real-time) as a distinct, high-reliability signal separate from general news
- Social/analyst sentiment (StockTwits API, or Twitter/X financial accounts) as a supplementary, lower-weight signal — noisier, weight accordingly

### 4.3 Aggregation
- Per-ticker sentiment score = weighted average of individual article scores, weighted by:
  - Source credibility (tiered list — Reuters/Bloomberg weighted higher than unverified blogs)
  - Recency (exponential decay, half-life ~6-12 hours for news-driven trading signals)
  - Volume (more independent sources agreeing = higher confidence)
- Output: `sentiment_score [-1, 1]`, `sentiment_confidence [0, 1]`, `article_count`, `top_headlines[]`

### 4.4 Divergence detection
Flag when sentiment and recent price action disagree (e.g., strongly negative sentiment but price flat/up) — this is itself a useful feature and gets surfaced explicitly to the LLM evaluator.

---

## Phase 5 — Fusion Layer

A lightweight service (not itself an ML model, or optionally a small logistic regression/shallow model) that combines:
```
fusion_input = {
  gbm_score, gbm_confidence, gbm_top_features, gbm_historical_analogs,
  sentiment_score, sentiment_confidence, sentiment_top_headlines, divergence_flag,
  feedback_context (relevant past mistakes matched by similarity)
}
```
If you want the fusion itself to be learned rather than fixed, train a **shallow meta-model** (logistic regression or small MLP) on `[gbm_score, sentiment_score, interaction terms] -> actual_outcome`, using the same walk-forward validation discipline as Phase 3. This lets the system learn, e.g., "when GBM and sentiment disagree, which one tends to be right historically."

---

## Phase 6 — Feedback Loop / Continual Learning

This is the "learn from its mistakes" piece — it needs to be explicit, not implicit, since GBMs don't online-learn like neural nets do.

### 6.1 Outcome tracking
For every prediction issued, log:
```
prediction_id, ticker, timestamp, gbm_score, sentiment_score, fusion_output,
predicted_class, actual_outcome (filled in after window closes),
error_type (false_positive / false_negative / correct),
post_mortem_tag (populated by a scheduled job, see 6.2)
```

### 6.2 Automated post-mortem tagging
A scheduled job (weekly) that, for resolved predictions with large errors, attempts to auto-tag *why* using rules + an LLM pass over the actual news that broke after the prediction:
- "Event category not in training taxonomy" (model literally couldn't have known)
- "Sentiment lagged the news" (sentiment score updated too slowly)
- "Historical analog was a poor match" (retrieved analogs had low actual similarity in hindsight)
- "Regime shift" (macro conditions changed faster than model's regime features captured)

### 6.3 Retraining cadence
- **GBM full retrain**: quarterly, on the full expanding-window dataset plus newly resolved outcomes
- **GBM incremental**: LightGBM supports `continue_training` — use this monthly for lighter-weight updates rather than full retrains
- **Sentiment model**: fine-tune only if post-mortem tags show systematic sentiment-attribution errors (this should be rarer — sentiment models degrade less than price-prediction models)
- **Drift monitoring**: track feature distribution drift (population stability index / PSI) between training data and live serving data; trigger an out-of-cycle retrain if PSI crosses a threshold (commonly 0.2)

### 6.4 Feeding mistakes back into the LLM evaluator
At inference time, the fusion layer queries the feedback log for past predictions on this ticker (or similar situations) and passes a summary into the LLM's context — this is the "learning" surfaced at the reasoning layer even between full model retrains. This is the `feedback_context` field referenced in Phase 5.

---

## Phase 7 — Serving Architecture

- **Model serving**: LightGBM models served via a lightweight FastAPI service (or BentoML/MLflow Model Serving) with TreeSHAP computed inline
- **Feature computation at serving time**: must read from the same feature store as training (Phase 2.5) to avoid skew
- **Orchestration**: Airflow or Dagster for the batch pipelines (retraining, feature backfills, post-mortem tagging)
- **Real-time sentiment**: a streaming component (Kafka + a lightweight consumer running FinBERT inference) if true real-time is needed; otherwise a polling job every 5-15 minutes is usually sufficient for this use case
- **Experiment tracking**: MLflow for all model versions, hyperparameters, and validation metrics — critical given the number of retraining cycles this system will go through
- **LLM evaluator**: called with the fusion payload as structured context (JSON in, structured markdown out)

---

## Phase 8 — Evaluation & Monitoring

### 8.1 Model-level metrics
- AUC-ROC and AUC-PR (PR curve matters more given class imbalance)
- **Brier score** and calibration curves — critical, since the LLM evaluator explicitly reports confidence, so the underlying probabilities need to actually be calibrated, not just rank-ordered correctly
- Precision/recall at top-k (since in practice, users likely act on the highest-conviction signals, not every prediction)

### 8.2 System-level monitoring
- Prediction volume and score distribution over time (catches silent model degradation)
- Feature drift (PSI) dashboards per feature group
- Sentiment source uptime/latency (if a news API goes down, sentiment silently goes stale — needs an explicit staleness flag, not silent failure)
- Feedback loop health: % of predictions with resolved outcomes, average time-to-resolution, post-mortem tagging coverage

### 8.3 Backtesting discipline
Before any model version is promoted to serving, run a full walk-forward backtest across at least 2 full market cycles it wasn't trained on, and require sign-off comparing new-version vs. current-version performance on the same held-out period — don't promote purely on aggregate metric improvement without checking regime-by-regime performance (a model can improve on average while getting meaningfully worse in high-volatility regimes, which is exactly when accuracy matters most).

---

## Suggested Build Order

1. Phase 1 (data) + Phase 2 (features) — this is the long pole, budget the most time here
2. Phase 3 (GBM) with rigorous validation — get this right before building anything downstream
3. Phase 4 (sentiment) — can be built in parallel with Phase 3 once Phase 1 news sourcing is sorted
4. Phase 5 (fusion) — thin layer, fast once 3 & 4 exist
5. Phase 6 (feedback loop) — build the *logging* infrastructure early (even in Phase 1), but the tagging/retraining automation comes last since it needs resolved outcomes to exist first
6. Phase 7 (serving) — build incrementally alongside 3-5
7. Phase 8 (monitoring) — instrument from day one, don't bolt on at the end

---

## Key Risks to Flag to Stakeholders

- **Regulatory**: this system, depending on framing, may fall under investment-adviser regulations if it's presented as personalized advice rather than research tooling — get legal sign-off on the output framing before launch.
- **Survivorship bias** in the historical dataset is the single most common way these systems quietly produce overconfident results — verify CRSP delisting data is actually being used, not just current-constituent data.
- **Overfitting to a benign regime**: a 50-year dataset sounds like a lot, but the number of *independent* major regime shifts (recessions, rate cycles) is actually small (roughly 8-10 in that period) — be honest internally about effective sample size at the regime level, not just row count.
