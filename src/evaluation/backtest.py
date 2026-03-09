"""
Backtesting / simulation engine.

Purpose
-------
Evaluate not just forecast quality but business decision quality.
A model with mediocre WAPE but a smart policy can outperform a model
with great WAPE but a naive ordering rule.

Simulation logic (per day, per SKU)
------------------------------------
  1. Look up that day's recorded demand (sales_qty from history).
  2. Look up that day's forecast (from model predictions).
  3. Compute order_qty using the configured policy.
  4. available_stock = previous day's carry-over + today's order delivery
     (assumed to arrive at start of day; simplifying assumption).
  5. fulfilled = min(demand, available_stock)
  6. leftover = max(0, available_stock - demand)
  7. waste = leftover that would expire before next day's delivery,
             based on expiration_days logic.
  8. carry_over = leftover - waste

Aggregate outputs
-----------------
  - WAPE, bias: forecast quality
  - service_level: fraction of demand met
  - waste_rate: fraction of ordered goods wasted
  - total_stockout_events: days where demand > available_stock (per sku-day)
  - total_units_wasted: cumulative expired units

Note on simulation fidelity
----------------------------
This is a single-period simulation (today's order arrives today).
Production systems would use a lead-time model (order placed D-n, arrives D).
For an MVP, same-day delivery is the simplest correct assumption to document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import (
    COL_DATE, COL_DEMAND_ADJUSTED, COL_EXPIRATION_DAYS, COL_IS_CENSORED,
    COL_SALES_QTY, COL_SHIPMENT_MULTIPLE, COL_SKU_CODE, COL_STOCK_BALANCE,
)
from src.evaluation.metrics import compute_all_metrics, format_metrics
from src.ordering.policy import apply_policy, FORECAST_COL
from src.ordering.recommender import ORDER_COL
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_backtest(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Config,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a walk-forward simulation over the backtest date range.

    Args:
        dataset: Full processed dataset (output of DatasetBuilder).
        predictions: DataFrame with columns [date, sku_code, forecast].
                     Must cover the backtest window.
        config: Full Config (policy + censoring settings).
        start_date: Backtest start date string (YYYY-MM-DD). Defaults to min date.
        end_date: Backtest end date string (YYYY-MM-DD). Defaults to max date.

    Returns:
        (results_df, summary_dict)
        results_df: Per-day per-SKU simulation outcomes.
        summary_dict: Aggregate metrics across the full window.
    """
    dataset = dataset.copy()
    dataset[COL_DATE] = pd.to_datetime(dataset[COL_DATE])
    predictions = predictions.copy()
    predictions[COL_DATE] = pd.to_datetime(predictions[COL_DATE])

    # Date filtering
    if start_date:
        predictions = predictions[predictions[COL_DATE] >= pd.Timestamp(start_date)]
    if end_date:
        predictions = predictions[predictions[COL_DATE] <= pd.Timestamp(end_date)]

    dates = sorted(predictions[COL_DATE].unique())
    logger.info("Backtest: %d dates, %d SKUs", len(dates), predictions[COL_SKU_CODE].nunique())

    # Merge product metadata into predictions for ordering logic
    meta_cols = [COL_SKU_CODE, COL_EXPIRATION_DAYS, COL_SHIPMENT_MULTIPLE]
    product_meta = (
        dataset[meta_cols]
        .drop_duplicates(COL_SKU_CODE)
        .set_index(COL_SKU_CODE)
    )

    # Initialize carry-over stock per SKU from the last known stock balance
    # before the backtest window starts
    carry_over: dict[str, float] = {}
    min_pred_date = predictions[COL_DATE].min()
    prior_stock = dataset[dataset[COL_DATE] < min_pred_date]
    if prior_stock.empty:
        prior_stock = dataset
    for sku in predictions[COL_SKU_CODE].unique():
        sku_stock = prior_stock[prior_stock[COL_SKU_CODE] == sku]
        if sku_stock.empty:
            carry_over[sku] = 0.0
        else:
            carry_over[sku] = float(sku_stock.sort_values(COL_DATE).iloc[-1][COL_STOCK_BALANCE])

    all_rows = []

    for date in dates:
        day_preds = predictions[predictions[COL_DATE] == date].copy()

        # Merge stock balances from carry-over (simulation state)
        day_preds[COL_STOCK_BALANCE] = day_preds[COL_SKU_CODE].map(carry_over).fillna(0)

        # Merge product metadata (expiration, shipment multiple)
        day_preds = day_preds.join(product_meta, on=COL_SKU_CODE, how="left")
        day_preds[COL_EXPIRATION_DAYS] = day_preds[COL_EXPIRATION_DAYS].fillna(1)
        day_preds[COL_SHIPMENT_MULTIPLE] = day_preds[COL_SHIPMENT_MULTIPLE].fillna(1)

        # Apply policy to get order_qty
        day_ordered = apply_policy(day_preds, config.policy)

        # Look up actual demand for this date from the dataset
        actual_day = dataset[dataset[COL_DATE] == date][[COL_SKU_CODE, COL_SALES_QTY]].copy()
        actual_day = actual_day.rename(columns={COL_SALES_QTY: "actual_demand"})

        day_result = day_ordered.merge(actual_day, on=COL_SKU_CODE, how="left")
        day_result["actual_demand"] = day_result["actual_demand"].fillna(0)

        # Simulate fulfillment
        day_result["available_stock"] = day_result[COL_STOCK_BALANCE] + day_result[ORDER_COL]
        day_result["fulfilled"] = np.minimum(day_result["actual_demand"], day_result["available_stock"])
        day_result["leftover"] = np.maximum(0, day_result["available_stock"] - day_result["actual_demand"])
        day_result["stockout"] = (day_result["actual_demand"] > day_result["available_stock"]).astype(int)

        # Waste: leftover that can't be used tomorrow (expires)
        day_result["waste"] = day_result.apply(
            lambda row: row["leftover"] if int(row.get(COL_EXPIRATION_DAYS, 1)) <= 1 else 0.0,
            axis=1,
        )
        day_result["carry_over_to_next"] = np.maximum(0, day_result["leftover"] - day_result["waste"])

        # Update carry-over state for next iteration
        for _, row in day_result.iterrows():
            carry_over[row[COL_SKU_CODE]] = float(row["carry_over_to_next"])

        all_rows.append(day_result)

    results_df = pd.concat(all_rows, ignore_index=True)

    summary = _compute_summary(results_df)
    logger.info("Backtest complete. %s", format_metrics(summary))
    return results_df, summary


def _compute_summary(results_df: pd.DataFrame) -> dict:
    """Compute aggregate metrics from simulation results."""
    y_true = results_df["actual_demand"].values
    y_pred = results_df[FORECAST_COL].values
    fulfilled = results_df["fulfilled"].values
    ordered = results_df[ORDER_COL].values
    waste = results_df["waste"].values

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        fulfilled=fulfilled,
        ordered=ordered,
        expired=waste,
    )
    metrics["total_stockout_events"] = int(results_df["stockout"].sum())
    metrics["total_units_wasted"] = float(results_df["waste"].sum())
    metrics["total_units_ordered"] = float(results_df[ORDER_COL].sum())
    metrics["total_units_fulfilled"] = float(results_df["fulfilled"].sum())
    metrics["n_sku_days"] = int(len(results_df))
    return metrics


def save_backtest_results(
    results_df: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    policy_name: str = "default",
) -> None:
    """Save simulation results and summary metrics to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"backtest_results_{policy_name}.parquet"
    results_df.to_parquet(results_path, index=False)
    logger.info("Backtest results saved to %s", results_path)

    summary_path = output_dir / f"backtest_summary_{policy_name}.txt"
    lines = [f"Policy: {policy_name}", "=" * 40]
    for k, v in summary.items():
        if isinstance(v, float) and k in ("wape", "waste_rate", "service_level"):
            lines.append(f"  {k:30s}: {v:.2%}")
        elif isinstance(v, float):
            lines.append(f"  {k:30s}: {v:+.3f}")
        else:
            lines.append(f"  {k:30s}: {v}")
    summary_path.write_text("\n".join(lines))
    logger.info("Summary saved to %s", summary_path)
