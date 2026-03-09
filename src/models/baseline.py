"""
Naive baseline forecasting models.

Used for:
  1. Sanity checking (LightGBM should beat both)
  2. Cold-start fallback (no training history available for a new SKU)
  3. Benchmark in backtest reports

NaiveModel
----------
Predicts using the most recent observed lag-1 value (yesterday's adjusted demand).
Expected performance: poor on seasonal/trending products.

SeasonalNaiveModel
------------------
Predicts using the same day-of-week from last week (lag-7 adjusted demand).
Expected performance: decent for weekday-pattern products; better than NaiveModel
on bakery data where Saturday sales differ significantly from Monday.

Both models do not require fitting; they read from feature columns in X.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NaiveModel(ForecastModel):
    """Predict = lag_1d (yesterday's demand)."""

    LAG_COL = "lag_1d"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveModel":
        logger.info("NaiveModel: no training required")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.LAG_COL not in X.columns:
            raise ValueError(f"NaiveModel requires column '{self.LAG_COL}' in feature matrix")
        preds = X[self.LAG_COL].fillna(0).values.astype(float)
        return np.clip(preds, 0, None)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({"model": "NaiveModel"}))
        logger.info("NaiveModel saved to %s", path)

    def load(self, path: Path) -> "NaiveModel":
        return self


class SeasonalNaiveModel(ForecastModel):
    """Predict = lag_7d (same weekday last week)."""

    LAG_COL = "lag_7d"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveModel":
        logger.info("SeasonalNaiveModel: no training required")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.LAG_COL not in X.columns:
            raise ValueError(f"SeasonalNaiveModel requires column '{self.LAG_COL}' in feature matrix")
        preds = X[self.LAG_COL].fillna(0).values.astype(float)
        return np.clip(preds, 0, None)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({"model": "SeasonalNaiveModel"}))
        logger.info("SeasonalNaiveModel saved to %s", path)

    def load(self, path: Path) -> "SeasonalNaiveModel":
        return self
