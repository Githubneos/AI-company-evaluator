"""FinBERT sentiment scoring (spec 4.1).

ProsusAI/finbert, CPU, batched. Domain-tuned on financial text, which matters:
general sentiment models read "shares fell on light guidance" as mildly negative
prose, while FinBERT reads it as a negative financial signal.

Scores are cached by content hash in SQLite. Headlines repeat heavily across
polls and tickers, and re-running BERT over the same string on a CPU is the
slowest avoidable thing in this pipeline.

KNOWN LIMITATION
----------------
FinBERT's 3-class output is coarse. It does not distinguish hedged from
categorical language, nor forward-looking guidance from backward-looking
results -- and those distinctions are much of what makes financial news
tradeable. Spec 4.1 suggests a fine-tuned DeBERTa-v3 for that; this is the
documented floor, not the ceiling.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from evaluator.config import ARTIFACT_DIR

log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
CACHE_DB = Path(ARTIFACT_DIR) / "sentiment_cache.db"
MAX_TOKENS = 512
BATCH_SIZE = 16

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    text_hash TEXT PRIMARY KEY,
    positive  REAL NOT NULL,
    negative  REAL NOT NULL,
    neutral   REAL NOT NULL,
    model     TEXT NOT NULL
);
"""


@dataclass
class Score:
    positive: float
    negative: float
    neutral: float

    @property
    def polarity(self) -> float:
        """Signed sentiment in [-1, 1]."""
        return self.positive - self.negative

    @property
    def confidence(self) -> float:
        """How far from neutral the model is -- its own certainty that this
        headline says anything at all."""
        return 1.0 - self.neutral


def _hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:20]


@contextmanager
def _cache(db_path: Path = CACHE_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _pipeline():
    """Load FinBERT once. First call downloads ~440 MB."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def _run_model(texts: list[str]) -> list[Score]:
    import torch

    tokenizer, model = _pipeline()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}

    scores: list[Score] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        )
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1).numpy()
        for row in probs:
            scores.append(
                Score(
                    positive=float(row[labels["positive"]]),
                    negative=float(row[labels["negative"]]),
                    neutral=float(row[labels["neutral"]]),
                )
            )
    return scores


def score_texts(texts: list[str], *, use_cache: bool = True, db_path: Path = CACHE_DB) -> list[Score]:
    """Score texts, consulting the cache first and only running BERT on misses."""
    if not texts:
        return []

    hashes = [_hash(t) for t in texts]
    cached: dict[str, Score] = {}

    if use_cache:
        with _cache(db_path) as conn:
            placeholders = ",".join("?" * len(hashes))
            rows = conn.execute(
                f"SELECT * FROM scores WHERE text_hash IN ({placeholders})", hashes
            ).fetchall()
            cached = {
                r["text_hash"]: Score(r["positive"], r["negative"], r["neutral"]) for r in rows
            }

    missing = [(h, t) for h, t in zip(hashes, texts) if h not in cached]
    if missing:
        fresh = _run_model([t for _, t in missing])
        for (h, _), score in zip(missing, fresh):
            cached[h] = score
        if use_cache:
            with _cache(db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)",
                    [
                        (h, cached[h].positive, cached[h].negative, cached[h].neutral, MODEL_NAME)
                        for h, _ in missing
                    ],
                )

    return [cached[h] for h in hashes]
