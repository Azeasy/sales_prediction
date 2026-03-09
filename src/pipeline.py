"""
Pipeline orchestration: high-level functions used by the CLI.

Each function represents one logical pipeline step.
They are kept here (separate from the CLI) so they can also be called
programmatically from tests or notebooks without triggering CLI machinery.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.dataset_builder import DatasetBuilder
from src.data.schema import COL_DATE, COL_SKU_CODE, COL_STOCK_BALANCE, COL_DEMAND_ADJUSTED
from src.demand.censoring import detect_censoring, adjust_target
from src.evaluation.backtest import run_backtest, save_backtest_results
from src.evaluation.metrics import format_metrics
from src.features.engineering import build_features, get_feature_columns, CATEGORICAL_FEATURES
from src.models.lgbm_model import LGBMForecaster, build_model
from src.ordering.policy import apply_policy, FORECAST_COL
from src.utils.config import Config, load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def step_fetch_data(config: Config, start_date: date, end_date: date) -> None:
    """Fetch raw data from API and cache to data/raw/."""
    loader = DataLoader(config)
    loader.fetch_all_api(start_date, end_date)


def step_build_dataset(config: Config) -> pd.DataFrame:
    """Load all source data and produce the processed dataset parquet."""
    loader = DataLoader(config)
    builder = DatasetBuilder(config)

    sales_df = loader.load_sales()
    stock_df = loader.load_stock()
    products_df = loader.load_products()
    losses_df = loader.load_losses()

    # Filter to last N months if lookback_months is set
    if config.data.lookback_months is not None and config.data.lookback_months > 0:
        if not sales_df.empty and COL_DATE in sales_df.columns:
            sales_df[COL_DATE] = pd.to_datetime(sales_df[COL_DATE])
            max_date = sales_df[COL_DATE].max()
            cutoff = max_date - pd.DateOffset(months=config.data.lookback_months)
            cutoff = cutoff.normalize()

            def filter_by_date(df: pd.DataFrame) -> pd.DataFrame:
                if df.empty or COL_DATE not in df.columns:
                    return df
                df = df.copy()
                df[COL_DATE] = pd.to_datetime(df[COL_DATE])
                return df[df[COL_DATE] >= cutoff]

            n_before = len(sales_df)
            sales_df = filter_by_date(sales_df)
            stock_df = filter_by_date(stock_df)
            losses_df = filter_by_date(losses_df)
            logger.info(
                "Filtered to last %d months: %d → %d sales rows (cutoff %s)",
                config.data.lookback_months, n_before, len(sales_df), cutoff.date(),
            )

    # Apply product matching mapping if enabled
    if config.product_matching.enabled:
        mapping_path = Path(config.artifacts.dir) / "product_mapping.json"
        if mapping_path.exists():
            from src.data.product_matching import load_mapping
            mapping = load_mapping(mapping_path)

            def apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
                if df.empty or COL_SKU_CODE not in df.columns:
                    return df
                df = df.copy()
                df[COL_SKU_CODE] = df[COL_SKU_CODE].astype(str).str.strip().map(
                    lambda x: mapping.get(x, x)
                )
                return df

            sales_df = apply_mapping(sales_df)
            stock_df = apply_mapping(stock_df)
            losses_df = apply_mapping(losses_df)
            products_df = apply_mapping(products_df)
            logger.info("Applied product_mapping.json: merged duplicate SKUs")
        else:
            logger.warning(
                "product_matching.enabled=true but %s not found. Run: python -m src.cli.main match-products",
                mapping_path,
            )

    dataset = builder.build(sales_df, stock_df, losses_df, products_df)
    builder.save(dataset)
    return dataset


def step_train(config: Config) -> dict:
    """
    Full train pipeline:
      1. Load processed dataset
      2. Detect and adjust censored demand
      3. Build feature matrix
      4. Temporal train/val split (last 20% of dates = val)
      5. Train LightGBM
      6. Evaluate on val set
      7. Save artifacts
    """
    builder = DatasetBuilder(config)
    dataset = builder.load()

    # Censoring
    dataset = detect_censoring(dataset)
    dataset = adjust_target(dataset, config.censoring)

    # Features
    dataset = build_features(dataset, config.features)

    feature_cols = get_feature_columns(dataset)
    target_col = COL_DEMAND_ADJUSTED

    # Temporal split: last 20% of dates for validation
    all_dates = sorted(dataset[COL_DATE].unique())
    split_idx = int(len(all_dates) * 0.8)
    split_date = all_dates[split_idx]

    train_mask = dataset[COL_DATE] < split_date
    val_mask = dataset[COL_DATE] >= split_date

    # Drop rows with NaN target (strategy=drop censoring)
    train_valid = dataset[train_mask][target_col].notna()
    X_train = dataset[train_mask][train_valid][feature_cols]
    y_train = dataset[train_mask][train_valid][target_col]

    X_val = dataset[val_mask][feature_cols]
    y_val = dataset[val_mask][target_col]

    logger.info("Train: %d rows, Val: %d rows, Split date: %s",
                len(X_train), len(X_val), split_date)

    model = LGBMForecaster(config.model, config.artifacts)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Evaluate on validation set
    val_preds = model.predict_clipped(X_val)
    val_actuals = y_val.fillna(0).values

    from src.evaluation.metrics import compute_all_metrics
    val_metrics = compute_all_metrics(val_actuals, val_preds)
    logger.info("Validation metrics: %s", format_metrics(val_metrics))

    model.save(Path(config.artifacts.dir))
    return val_metrics


def step_predict(config: Config, target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load model and generate next-day forecasts for all SKUs.

    If target_date is None, uses the latest date in the dataset + 1.
    """
    builder = DatasetBuilder(config)
    dataset = builder.load()

    # Censoring + features (needed for lag columns)
    dataset = detect_censoring(dataset)
    dataset = adjust_target(dataset, config.censoring)
    dataset = build_features(dataset, config.features)

    feature_cols = get_feature_columns(dataset)

    if target_date is not None:
        pred_date = pd.Timestamp(target_date)
        pred_data = dataset[dataset[COL_DATE] == pred_date]
        if pred_data.empty:
            raise ValueError(f"No data found for prediction date {target_date}")
    else:
        # Predict for the last available date (i.e., forecast day+1)
        last_date = dataset[COL_DATE].max()
        pred_data = dataset[dataset[COL_DATE] == last_date]
        logger.info("No target_date specified; predicting using features from %s", last_date.date())

    model = LGBMForecaster(config.model, config.artifacts)
    model.load(Path(config.artifacts.dir))

    X_pred = pred_data[feature_cols]
    forecasts = model.predict_clipped(X_pred)

    result = pred_data[[COL_DATE, COL_SKU_CODE, COL_STOCK_BALANCE]].copy()
    result[FORECAST_COL] = forecasts

    # Also carry through rolling_std_7d if available (for policy quantile adjustment)
    if "rolling_std_7d" in pred_data.columns:
        result["rolling_std_7d"] = pred_data["rolling_std_7d"].values

    logger.info("Predictions generated for %d SKUs", len(result))
    return result


def step_recommend_order(config: Config, target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Generate forecast + apply policy to produce order_qty recommendations.

    Returns DataFrame with: date, sku_code, forecast, adjusted_forecast,
    safety_stock, stock_balance, order_qty.
    """
    # Need product metadata for the recommender
    builder = DatasetBuilder(config)
    dataset = builder.load()

    from src.data.schema import COL_EXPIRATION_DAYS, COL_SHIPMENT_MULTIPLE, COL_SKU_CODE as SKU
    meta_cols = [SKU, COL_EXPIRATION_DAYS, COL_SHIPMENT_MULTIPLE, "sku_name", "product_group"]
    available_meta = [c for c in meta_cols if c in dataset.columns]
    product_meta = dataset[available_meta].drop_duplicates(SKU)

    forecasts = step_predict(config, target_date)
    forecasts = forecasts.merge(product_meta, on=SKU, how="left")
    forecasts[COL_EXPIRATION_DAYS] = forecasts[COL_EXPIRATION_DAYS].fillna(1)
    forecasts[COL_SHIPMENT_MULTIPLE] = forecasts[COL_SHIPMENT_MULTIPLE].fillna(1)

    recommendations = apply_policy(forecasts, config.policy)
    return recommendations


def step_backtest(
    config: Config,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run backtest: generate predictions for all historical dates then simulate.
    """
    builder = DatasetBuilder(config)
    dataset = builder.load()

    dataset_with_features = detect_censoring(dataset)
    dataset_with_features = adjust_target(dataset_with_features, config.censoring)
    dataset_with_features = build_features(dataset_with_features, config.features)

    feature_cols = get_feature_columns(dataset_with_features)

    model = LGBMForecaster(config.model, config.artifacts)
    model.load(Path(config.artifacts.dir))

    X_all = dataset_with_features[feature_cols]
    all_preds = model.predict_clipped(X_all)

    predictions = dataset_with_features[[COL_DATE, COL_SKU_CODE]].copy()
    predictions[FORECAST_COL] = all_preds
    if "rolling_std_7d" in dataset_with_features.columns:
        predictions["rolling_std_7d"] = dataset_with_features["rolling_std_7d"].values

    results_df, summary = run_backtest(
        dataset=dataset,
        predictions=predictions,
        config=config,
        start_date=start_date,
        end_date=end_date,
    )

    save_backtest_results(
        results_df, summary,
        output_dir=Path(config.artifacts.dir),
        policy_name=config.policy.mode,
    )
    return results_df, summary
