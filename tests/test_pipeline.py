"""
Integration test: happy-path pipeline.

Tests the full flow from dataset construction through feature engineering,
model training, prediction, and order recommendation on synthetic data.

This test does NOT require any API access or external files.
It creates a temporary environment and exercises the core pipeline end-to-end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    COL_DATE, COL_DEMAND_ADJUSTED, COL_EXPIRATION_DAYS, COL_IS_CENSORED,
    COL_SALES_QTY, COL_SHIPMENT_MULTIPLE, COL_SKU_CODE, COL_STOCK_BALANCE,
    COL_STORE_ID,
)
from src.demand.censoring import adjust_target, detect_censoring
from src.features.engineering import build_features, get_feature_columns
from src.models.lgbm_model import LGBMForecaster
from src.ordering.policy import apply_policy, FORECAST_COL
from src.utils.config import (
    ArtifactsConfig, CensoringConfig, FeaturesConfig, ModelConfig, PolicyConfig,
)


@pytest.fixture
def minimal_dataset():
    """12 SKUs × 45 days synthetic dataset — enough for lag features to populate."""
    np.random.seed(0)
    n_skus = 4
    n_days = 45
    skus = [f"SKU-{i:02d}" for i in range(n_skus)]
    rows = []
    for sku in skus:
        base = np.random.randint(5, 20)
        for d in pd.date_range("2026-01-01", periods=n_days, freq="D"):
            stock = 0 if d.day % 10 == 0 else np.random.randint(10, 30)
            sales = min(base + np.random.randint(-3, 4), max(0, stock))
            rows.append({
                COL_DATE: d,
                COL_STORE_ID: "default_store",
                COL_SKU_CODE: sku,
                "sku_name": f"Product {sku}",
                "product_group": "Булочки",
                COL_SALES_QTY: float(max(0, sales)),
                "sales_amount": float(max(0, sales) * 100),
                COL_STOCK_BALANCE: float(max(0, stock)),
                "loss_qty": 0.0,
                "loss_amount": 0.0,
                COL_EXPIRATION_DAYS: 3,
                COL_SHIPMENT_MULTIPLE: 1,
                "unit_of_measure": "шт",
            })
    return pd.DataFrame(rows)


class TestCensoringPipeline:
    def test_detect_and_adjust_produces_valid_columns(self, minimal_dataset):
        df = detect_censoring(minimal_dataset)
        assert COL_IS_CENSORED in df.columns
        assert df[COL_IS_CENSORED].dtype == bool or df[COL_IS_CENSORED].isin([True, False]).all()

        cfg = CensoringConfig(strategy="impute", rolling_window=7, use_dow_grouping=False)
        df = adjust_target(df, cfg)
        assert COL_DEMAND_ADJUSTED in df.columns
        # Key invariant: adjusted demand >= observed sales
        assert (df[COL_DEMAND_ADJUSTED] >= df[COL_SALES_QTY]).all()


class TestFeaturePipeline:
    def test_features_built_without_leakage(self, minimal_dataset):
        df = detect_censoring(minimal_dataset)
        cfg_cens = CensoringConfig(strategy="impute", rolling_window=7, use_dow_grouping=False)
        df = adjust_target(df, cfg_cens)

        cfg_feat = FeaturesConfig(lags=[1, 7], rolling_windows=[7])
        df = build_features(df, cfg_feat)

        feature_cols = get_feature_columns(df)
        assert len(feature_cols) > 0
        assert "lag_1d" in feature_cols
        assert "lag_7d" in feature_cols
        assert "rolling_mean_7d" in feature_cols
        # Ensure no future date columns exist
        assert "future" not in " ".join(feature_cols)


class TestModelTrainPredict:
    def test_lgbm_trains_and_predicts(self, minimal_dataset, tmp_path):
        df = detect_censoring(minimal_dataset)
        cfg_cens = CensoringConfig(strategy="impute", rolling_window=7, use_dow_grouping=False)
        df = adjust_target(df, cfg_cens)
        cfg_feat = FeaturesConfig(lags=[1, 7], rolling_windows=[7])
        df = build_features(df, cfg_feat)

        feature_cols = get_feature_columns(df)

        # Only train on rows where target is non-null and lags are populated (after warmup)
        valid = df[COL_DEMAND_ADJUSTED].notna() & df["lag_1d"].notna()
        X = df[valid][feature_cols]
        y = df[valid][COL_DEMAND_ADJUSTED]

        assert len(X) > 10, "Too few training samples"

        model_cfg = ModelConfig(type="lgbm", params={
            "n_estimators": 50, "learning_rate": 0.1, "num_leaves": 8,
            "min_child_samples": 5, "random_state": 42, "verbose": -1, "n_jobs": 1,
        })
        artifacts_cfg = ArtifactsConfig(
            dir=str(tmp_path),
            model_file="model.pkl",
            features_file="features.json",
            feature_importance_file="importance.csv",
        )

        model = LGBMForecaster(model_cfg, artifacts_cfg)
        model.fit(X, y)

        preds = model.predict_clipped(X)
        assert len(preds) == len(X)
        assert (preds >= 0).all()

        # Save and reload
        model.save(tmp_path)
        assert (tmp_path / "model.pkl").exists()
        assert (tmp_path / "features.json").exists()

        model2 = LGBMForecaster(model_cfg, artifacts_cfg)
        model2.load(tmp_path)
        preds2 = model2.predict_clipped(X)
        np.testing.assert_array_almost_equal(preds, preds2)


class TestOrderingIntegration:
    def test_order_recommendations_are_non_negative(self):
        forecast_df = pd.DataFrame({
            "date": [pd.Timestamp("2026-03-08")] * 3,
            "sku_code": ["SKU-A", "SKU-B", "SKU-C"],
            FORECAST_COL: [10.0, 0.0, 25.0],
            COL_STOCK_BALANCE: [5.0, 20.0, 0.0],
            COL_EXPIRATION_DAYS: [3, 5, 2],
            COL_SHIPMENT_MULTIPLE: [1, 2, 6],
            "rolling_std_7d": [2.0, 1.0, 5.0],
        })

        cfg = PolicyConfig(mode="balanced", safety_stock_multiplier=1.0,
                           forecast_quantile=0.55, max_cover_days=None, round_up_shipment=True)
        result = apply_policy(forecast_df, cfg)
        assert (result["order_qty"] >= 0).all()
        assert result["order_qty"].dtype in [int, np.int64, np.int32, object]

    def test_policy_ordering_service_ge_waste(self):
        """Service-first should always order >= waste-first for same inputs."""
        forecast_df = pd.DataFrame({
            "date": [pd.Timestamp("2026-03-08")],
            "sku_code": ["SKU-A"],
            FORECAST_COL: [15.0],
            COL_STOCK_BALANCE: [0.0],
            COL_EXPIRATION_DAYS: [5],
            COL_SHIPMENT_MULTIPLE: [1],
            "rolling_std_7d": [3.0],
        })

        svc_cfg = PolicyConfig(mode="service_first", safety_stock_multiplier=1.5,
                               forecast_quantile=0.85, max_cover_days=None, round_up_shipment=True)
        waste_cfg = PolicyConfig(mode="waste_first", safety_stock_multiplier=0.3,
                                 forecast_quantile=0.30, max_cover_days=1, round_up_shipment=False)

        order_svc = apply_policy(forecast_df, svc_cfg)["order_qty"].iloc[0]
        order_waste = apply_policy(forecast_df, waste_cfg)["order_qty"].iloc[0]

        assert order_svc >= order_waste
