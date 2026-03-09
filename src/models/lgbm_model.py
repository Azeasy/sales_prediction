"""
LightGBM-based demand forecaster.

Why LightGBM
------------
- State-of-the-art for tabular regression on CPU
- Handles categorical features natively (product_group, day_of_week)
- Fast inference; supports feature importance for explainability
- Gradient-boosted trees are robust to outliers and scale-independent
- Trains in seconds on bakery-scale data (< 1M rows)

Architecture
------------
Single global model across all SKUs: all (sku, date) rows are training
examples with sku-specific features (lags, rolling stats, metadata) making
the model SKU-aware without requiring one model per SKU (which would break
cold-start and create maintenance headaches).

Training
--------
Rows where demand_adjusted is NaN (strategy=drop censoring) are excluded.
The validation split is the last 20% of dates (temporal hold-out).
We log validation WAPE after training for quick sanity checking.

Artifacts
---------
model: {artifacts_dir}/{model_file}        — joblib-serialized model
features: {artifacts_dir}/{features_file}  — JSON list of feature columns
importance: {artifacts_dir}/{feature_importance_file} — CSV of importances
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.base import ForecastModel
from src.utils.config import ArtifactsConfig, ModelConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LGBMForecaster(ForecastModel):
    """
    LightGBM regression model for next-day demand forecasting.

    Usage:
        model = LGBMForecaster(model_cfg, artifacts_cfg)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        model.save(artifacts_path)
    """

    def __init__(self, model_cfg: ModelConfig, artifacts_cfg: ArtifactsConfig):
        self._model_cfg = model_cfg
        self._artifacts_cfg = artifacts_cfg
        self._model = None
        self._feature_columns: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "LGBMForecaster":
        """
        Train the LightGBM model.

        Args:
            X: Feature matrix (all numeric + category dtypes).
            y: Target series (demand_adjusted). NaN rows are auto-excluded.
            eval_set: Optional (X_val, y_val) tuple for early stopping.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Run: pip install lightgbm")

        # Exclude NaN targets (strategy=drop censoring)
        valid_mask = y.notna()
        X_train = X[valid_mask].copy()
        y_train = y[valid_mask].copy()
        logger.info("Training LightGBM on %d rows (%d NaN targets excluded)",
                    len(X_train), (~valid_mask).sum())

        self._feature_columns = list(X_train.columns)

        params = dict(self._model_cfg.params)
        n_estimators = params.pop("n_estimators", 500)

        self._model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)

        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            val_mask = y_val.notna()
            fit_kwargs["eval_set"] = [(X_val[val_mask][self._feature_columns], y_val[val_mask])]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        logger.info("LightGBM training complete. Best iteration: %s",
                    getattr(self._model, "best_iteration_", "N/A"))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been trained or loaded yet")
        X_pred = X[self._feature_columns].copy()
        preds = self._model.predict(X_pred)
        return np.clip(preds, 0, None)

    def save(self, artifacts_dir: Path | None = None) -> None:
        """Save model, feature list, and feature importance to artifacts directory."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been trained")

        if artifacts_dir is None:
            artifacts_dir = Path(self._artifacts_cfg.dir)
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / self._artifacts_cfg.model_file
        joblib.dump(self._model, model_path)
        logger.info("Model saved to %s", model_path)

        features_path = artifacts_dir / self._artifacts_cfg.features_file
        features_path.write_text(json.dumps(self._feature_columns, ensure_ascii=False, indent=2))
        logger.info("Feature list saved to %s", features_path)

        importance_path = artifacts_dir / self._artifacts_cfg.feature_importance_file
        importance_df = pd.DataFrame({
            "feature": self._feature_columns,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(importance_path, index=False)
        logger.info("Feature importance saved to %s", importance_path)

    def load(self, artifacts_dir: Path | None = None) -> "LGBMForecaster":
        """Load model and feature list from artifacts directory."""
        if artifacts_dir is None:
            artifacts_dir = Path(self._artifacts_cfg.dir)
        artifacts_dir = Path(artifacts_dir)

        model_path = artifacts_dir / self._artifacts_cfg.model_file
        features_path = artifacts_dir / self._artifacts_cfg.features_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}\nRun: python -m src.cli.main train"
            )

        self._model = joblib.load(model_path)
        self._feature_columns = json.loads(features_path.read_text())
        logger.info("Model loaded from %s (%d features)", model_path, len(self._feature_columns))
        return self

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)


def build_model(model_cfg: ModelConfig, artifacts_cfg: ArtifactsConfig) -> ForecastModel:
    """
    Factory: create the right model type based on config.model.type.

    Returns an untrained model instance.
    """
    model_type = model_cfg.type.lower()
    if model_type == "lgbm":
        return LGBMForecaster(model_cfg, artifacts_cfg)
    elif model_type == "naive":
        from src.models.baseline import NaiveModel
        return NaiveModel()
    elif model_type == "seasonal_naive":
        from src.models.baseline import SeasonalNaiveModel
        return SeasonalNaiveModel()
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Choose: lgbm, naive, seasonal_naive")
