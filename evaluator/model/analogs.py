"""Historical analog retrieval (spec 3.5).

Separate from the GBM's numeric output: given today's situation, find the most
similar historical situations and report what actually happened next. This is
what the reasoning layer surfaces as "closest historical analogs", and it is
useful precisely because it is not a model prediction -- it is a set of realised
outcomes the analyst can inspect.

Implemented with standardised feature vectors and sklearn's NearestNeighbors
rather than FAISS or pgvector. At panel scale (a few million rows, ~40 dims) an
exact search is fast enough, and it avoids a vector database and its disk cost
for no accuracy gain. The seam is small if that changes.

TWO CORRECTNESS CONSTRAINTS
---------------------------
Neighbours must be drawn only from dates strictly before the query date, or the
"historical" analogs include the future. And the scaler must be fit on the index
data alone -- fitting it on the query would leak the query's distribution into
the standardisation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from evaluator.config import ARTIFACT_DIR

log = logging.getLogger(__name__)

ANALOG_DIR = Path(ARTIFACT_DIR) / "analogs"

#: A compact situation description. Using every feature would let dozens of
#: near-duplicate technical columns dominate the distance and swamp the event
#: and valuation context that makes an analog interesting.
ANALOG_FEATURES = [
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "vol_ratio_20_60",
    "volume_z_60",
    "rsi_14",
    "bb_position",
    "drawdown_from_252h",
    "excess_ret_20",
    "sector_rel_ret_20",
    "event_density_90d",
    "days_since_earnings_result",
    "vix",
    "debt_to_equity",
]


@dataclass
class AnalogIndex:
    neighbours: NearestNeighbors
    scaler: StandardScaler
    reference: pd.DataFrame
    features: list[str]

    def query(
        self,
        row: pd.DataFrame,
        as_of: pd.Timestamp,
        k: int = 5,
        *,
        exclude_ticker: str | None = None,
    ) -> list[dict]:
        """Top-k historical situations preceding `as_of`, with their outcomes."""
        vector = row.reindex(columns=self.features).to_numpy(dtype=float)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self.scaler.transform(vector)

        # Filter by availability *before* ranking.  Over-fetching nearest
        # neighbours and then removing future rows can exhaust the candidate
        # pool even when many eligible historical rows exist.  That is both an
        # incomplete answer and a subtle source of inconsistent results around
        # regime changes, when nearby future rows cluster tightly.
        eligible = self.reference["date"] < pd.Timestamp(as_of)
        if exclude_ticker:
            eligible &= self.reference["ticker"] != exclude_ticker
        candidates = self.reference.loc[eligible]
        if candidates.empty:
            return []

        matrix = candidates.reindex(columns=self.features).to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        candidate_vectors = self.scaler.transform(matrix)
        distances = np.linalg.norm(candidate_vectors - scaled[0], axis=1)
        count = min(k, len(candidates))
        chosen = np.argpartition(distances, count - 1)[:count]
        chosen = chosen[np.argsort(distances[chosen])]

        results = []
        for position in chosen:
            candidate = candidates.iloc[position]
            results.append(
                {
                    "ticker": candidate["ticker"],
                    "date": str(pd.Timestamp(candidate["date"]).date()),
                    "distance": round(float(distances[position]), 4),
                    "forward_return_5d": _round(candidate.get("forward_return_5d")),
                    "forward_return_20d": _round(candidate.get("forward_return_20d")),
                    "outcome_5d": _outcome(candidate.get("label_direction_5d")),
                }
            )
        return results


def _round(value, digits: int = 4):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _outcome(label) -> str | None:
    if label is None or pd.isna(label):
        return None
    return {0: "DROP", 1: "NEUTRAL", 2: "SPIKE"}[int(label)]


def build_index(panel_frame: pd.DataFrame, max_rows: int = 400_000, seed: int = 7) -> AnalogIndex:
    """Fit the retrieval index over the labelled history.

    Subsampled by default: an exact neighbour search over millions of rows costs
    memory that an 8 GB machine does not have, and analog quality saturates long
    before the full panel is used.
    """
    features = [f for f in ANALOG_FEATURES if f in panel_frame.columns]
    keep = ["ticker", "date", "forward_return_5d", "forward_return_20d", "label_direction_5d"]
    keep = [c for c in keep if c in panel_frame.columns]

    frame = panel_frame
    if "label_direction_5d" in panel_frame:
        frame = panel_frame[panel_frame["label_direction_5d"].notna()]
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    frame = frame.sort_values("date").reset_index(drop=True)

    matrix = np.nan_to_num(
        frame[features].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    scaler = StandardScaler().fit(matrix)
    neighbours = NearestNeighbors(n_neighbors=min(200, len(frame)), algorithm="auto").fit(
        scaler.transform(matrix)
    )

    log.info("analog index: %s rows, %d features", f"{len(frame):,}", len(features))
    # Keep the feature vectors with the reference metadata.  Date and ticker
    # eligibility must be applied before distance ranking at query time, which
    # cannot be done correctly if the persisted reference has only outcomes.
    reference_columns = list(dict.fromkeys([*keep, *features]))
    return AnalogIndex(neighbours, scaler, frame[reference_columns].copy(), features)


def save_index(index: AnalogIndex) -> None:
    import joblib

    ANALOG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"neighbours": index.neighbours, "scaler": index.scaler, "features": index.features},
        ANALOG_DIR / "index.joblib",
    )
    index.reference.to_parquet(ANALOG_DIR / "reference.parquet", index=False)
    (ANALOG_DIR / "meta.json").write_text(
        json.dumps({"rows": len(index.reference), "features": index.features}, indent=2)
    )


def load_index() -> AnalogIndex | None:
    import joblib

    path = ANALOG_DIR / "index.joblib"
    if not path.exists():
        return None
    blob = joblib.load(path)
    reference = pd.read_parquet(ANALOG_DIR / "reference.parquet")
    features = blob["features"]
    if not set(features) <= set(reference.columns):
        log.warning("analog index is from an older format; rebuild it with scripts.build_analogs")
        return None
    return AnalogIndex(blob["neighbours"], blob["scaler"], reference, features)
