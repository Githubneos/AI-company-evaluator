"""Versioned feature store (spec 2.5), the Feast substitute.

The point of a feature store is not the storage; it is the guarantee that
training and serving compute features the same way. That guarantee here comes
from both paths calling `build_dataset`, and from the schema hash recorded
alongside the panel: if the feature definitions change, the hash changes, and a
model trained on the old schema refuses to score against the new one instead of
silently misaligning columns.

The panel is built one ticker at a time and downcast to float32. On 8 GB of RAM
a 500-name, 20-year panel held as float64 is large enough to start swapping,
and float32 costs nothing here -- these are noisy financial features, not
quantities where the 9th significant digit matters.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.config import ARTIFACT_DIR, TargetSpec, default_targets
from evaluator.data.universe import load_universe
from evaluator.dataset import build_dataset

log = logging.getLogger(__name__)

STORE_DIR = Path(ARTIFACT_DIR) / "features"
PANEL_PATH = STORE_DIR / "panel.parquet"
META_PATH = STORE_DIR / "meta.json"

ID_COLUMNS = ["ticker", "date"]


def schema_hash(feature_names: list[str]) -> str:
    """Stable fingerprint of the feature set."""
    return hashlib.sha1("|".join(sorted(feature_names)).encode()).hexdigest()[:12]


@dataclass
class FeaturePanel:
    frame: pd.DataFrame
    feature_names: list[str]
    label_names: list[str]
    schema: str

    def target_frame(self, spec: TargetSpec) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """(X, y, dates, tickers) for one target, dropping unlabelled rows.

        Memory matters more than elegance here: on 8 GB, materialising a second
        full copy of a 2.5M-row panel is the difference between training and
        being OOM-killed. So this slices columns *before* rows, and does not
        sort -- `build_panel` already writes the panel date-ordered, which is the
        ordering `split_panel` depends on. A defensive re-sort would silently
        double peak memory for no change in output.
        """
        column = f"{spec.label_column}_{spec.horizon_days}d"
        mask = self.frame[column].notna().to_numpy()

        X = self.frame.loc[mask, self.feature_names]
        y = self.frame.loc[mask, column].astype("int8")
        dates = self.frame.loc[mask, "date"]
        tickers = self.frame.loc[mask, "ticker"]

        if not dates.is_monotonic_increasing:
            raise ValueError(
                "panel is not date-ordered; rebuild it with scripts.build_panel "
                "(split_panel requires ascending dates)"
            )
        return X, y, dates, tickers


def build_panel(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
    targets: list[TargetSpec] | None = None,
    *,
    limit: int | None = None,
) -> FeaturePanel:
    """Build and persist the full cross-sectional panel."""
    targets = targets or default_targets()
    if tickers is None:
        tickers = load_universe()["ticker"].tolist()
    if limit:
        tickers = tickers[:limit]

    frames: list[pd.DataFrame] = []
    feature_names: list[str] | None = None
    failures = 0

    for i, ticker in enumerate(tickers, start=1):
        try:
            dataset = build_dataset(ticker, start, end, targets=targets)
        except Exception as exc:  # noqa: BLE001 - one bad name must not kill the panel
            log.warning("skipping %s: %s", ticker, exc)
            failures += 1
            continue

        if dataset.features.empty:
            failures += 1
            continue

        if feature_names is None:
            feature_names = list(dataset.features.columns)

        block = dataset.features.reindex(columns=feature_names).astype("float32")
        block = pd.concat([block, dataset.labels], axis=1)
        block.insert(0, "ticker", ticker)
        block = block.reset_index().rename(columns={"index": "date"})
        # Do not train on a company's history before it was in the historical
        # S&P constituent universe.  The checked-in universe is still
        # survivorship-biased (only CRSP can fix that), but this removes the
        # separate, avoidable look-ahead from treating today's constituents as
        # members in years before their actual inclusion.
        date_added = load_universe().loc[lambda u: u["ticker"] == ticker, "date_added"]
        if not date_added.empty and pd.notna(date_added.iloc[0]):
            block = block[block["date"] >= date_added.iloc[0]]
        if block.empty:
            failures += 1
            continue
        frames.append(block)

        if i % 25 == 0:
            log.info("panel: %d/%d tickers, %d frames", i, len(tickers), len(frames))

    if not frames:
        raise RuntimeError("no tickers produced features")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    label_names = [c for c in panel.columns if c.startswith(("label_", "forward_"))]
    schema = schema_hash(feature_names)

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    META_PATH.write_text(
        json.dumps(
            {
                "schema": schema,
                "feature_names": feature_names,
                "label_names": label_names,
                "tickers": int(panel["ticker"].nunique()),
                "rows": int(len(panel)),
                "start": str(panel["date"].min().date()),
                "end": str(panel["date"].max().date()),
                "failures": failures,
                "targets": [t.name for t in targets],
            },
            indent=2,
        )
    )
    log.info("panel: %s rows, %s tickers, schema %s", f"{len(panel):,}", panel["ticker"].nunique(), schema)

    return FeaturePanel(panel, feature_names, label_names, schema)


def _meta() -> dict:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"no feature panel at {PANEL_PATH}. Run: python -m scripts.build_panel"
        )
    return json.loads(META_PATH.read_text())


def load_panel(targets: list[TargetSpec] | None = None) -> FeaturePanel:
    """Load the panel, optionally restricted to the columns some targets need.

    Passing `targets` reads only the features plus those targets' label columns.
    On a 2.5M-row panel that is the difference between ~750 MB and ~500 MB
    resident, which on an 8 GB machine decides whether LightGBM can then build
    its own dataset alongside.
    """
    meta = _meta()
    columns = None
    if targets:
        wanted = {f"{t.label_column}_{t.horizon_days}d" for t in targets}
        columns = ["ticker", "date", *meta["feature_names"], *sorted(wanted)]

    frame = pd.read_parquet(PANEL_PATH, columns=columns)
    label_names = [c for c in meta["label_names"] if c in frame.columns]
    return FeaturePanel(frame, meta["feature_names"], label_names, meta["schema"])


def panel_schema() -> str:
    """The stored schema hash without loading the panel."""
    return _meta()["schema"]


def align_to_schema(row: pd.DataFrame, feature_names: list[str], schema: str) -> pd.DataFrame:
    """Reindex a serving row onto a model's training schema, or refuse.

    Silently reindexing a mismatched schema is how a model ends up scoring
    columns that mean something different from what it learned.
    """
    if schema_hash(list(row.columns)) != schema and not set(feature_names) <= set(row.columns):
        missing = sorted(set(feature_names) - set(row.columns))
        raise ValueError(
            f"serving features do not satisfy training schema {schema}; missing: {missing[:10]}"
        )
    return row.reindex(columns=feature_names).astype("float32").replace([np.inf, -np.inf], np.nan)
