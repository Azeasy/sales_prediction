"""
Abstract base class for all forecasting models.

Defines the interface that all models must implement.
This decouples the training pipeline from any specific model choice —
swap LightGBM for XGBoost or a neural net without touching the CLI or backtest code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """
    Minimal interface for a demand forecasting model.

    All concrete models inherit from this and implement fit/predict/save/load.
    predict() always returns a non-negative numpy array (demand cannot be negative).
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel":
        """Train the model on feature matrix X and target y."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for each row in X. Returns 1-D array of floats."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""

    @abstractmethod
    def load(self, path: Path) -> "ForecastModel":
        """Deserialize model from disk. Returns self for chaining."""

    def predict_clipped(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and clip negative values to 0 (demand can't be negative)."""
        preds = self.predict(X)
        return np.clip(preds, 0, None)
