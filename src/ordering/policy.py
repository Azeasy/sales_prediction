"""
Policy layer: translates a demand forecast into a policy-adjusted order recommendation.

The policy layer sits between the forecasting model and the order computation.
It does NOT modify the model or the raw forecast — it adjusts the parameters
passed to compute_order() based on the configured operating mode.

Three operating modes
---------------------

service_first
    Goal: minimize out-of-stock events. Accept higher waste as trade-off.
    - High forecast_quantile (0.85): order for a high-demand scenario, not the median.
    - High safety_stock_multiplier (1.5): generous buffer above the forecast.
    - max_cover_days: None (cover full shelf life — don't artificially cap).
    - round_up_shipment: True (always full packs; no partial packs allowed).

balanced
    Goal: balance service level and waste. Default for general use.
    - Moderate forecast_quantile (0.55): slightly above median.
    - safety_stock_multiplier: 1.0 (one day's forecast as buffer).
    - max_cover_days: expiration_days (don't order beyond shelf life).
    - round_up_shipment: True.

waste_first
    Goal: minimize write-offs and spoilage. Accept some stockout risk.
    - Low forecast_quantile (0.30): order conservatively, below median.
    - Low safety_stock_multiplier (0.3): minimal buffer.
    - max_cover_days: 1 (only cover what we expect to sell tomorrow).
    - round_up_shipment: False (round down; better to under-order slightly).

Quantile scaling
----------------
The forecast from the model is a point estimate (mean/median).
Policy adjusts it by multiplying by a quantile-based scaling factor.
This is a simple, explainable way to target different risk levels
without retraining the model for each policy.

Scaling factor: we estimate the forecast standard deviation from the
rolling_std_7d feature (if available) and compute:
    adjusted_forecast = forecast + z_score * std_estimate
where z_score maps the quantile to a normal z (e.g., 0.85 → z≈1.04).

Fallback: if no std estimate available, scale by the quantile ratio
relative to 0.5 using a fixed CV assumption (coefficient of variation=0.3).
"""

from __future__ import annotations

import math
from typing import Optional

import math

import numpy as np
import pandas as pd

from src.ordering.recommender import ORDER_COL, FORECAST_COL, compute_order, recommend_orders
from src.data.schema import COL_EXPIRATION_DAYS, COL_SHIPMENT_MULTIPLE, COL_STOCK_BALANCE, COL_SKU_CODE
from src.utils.config import PolicyConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Assumed coefficient of variation when no rolling std is available
# CV=0.3 means std = 30% of mean — typical for bakery daily demand
_FALLBACK_CV = 0.3


def _quantile_to_z(quantile: float) -> float:
    """Map a probability quantile to a standard normal z-score.

    Uses scipy.stats.norm.ppf when available; falls back to a rational
    approximation (max error < 4.5e-4) otherwise.
    """
    quantile = max(0.01, min(0.99, quantile))
    try:
        from scipy import stats as scipy_stats
        return float(scipy_stats.norm.ppf(quantile))
    except ImportError:
        pass
    # Rational approximation — accurate enough for operational quantile targeting
    if quantile < 0.5:
        sign, q = -1.0, quantile
    else:
        sign, q = 1.0, 1.0 - quantile
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t ** 2) / (1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return sign * z


def adjust_forecast_for_policy(
    forecast: float,
    std_estimate: Optional[float],
    policy_cfg: PolicyConfig,
) -> float:
    """
    Apply quantile-based scaling to a point forecast.

    At quantile=0.5, the forecast is unchanged (median).
    At quantile=0.85, the forecast is shifted up by ~1.04 * std (order more).
    At quantile=0.30, the forecast is shifted down (order less).

    Args:
        forecast: Model's point estimate for next-day demand.
        std_estimate: Estimated forecast std (from rolling_std_7d if available).
        policy_cfg: Policy configuration.

    Returns:
        Adjusted forecast as float.
    """
    q = policy_cfg.forecast_quantile
    if q == 0.5:
        return forecast

    z = _quantile_to_z(q)

    if std_estimate is not None and std_estimate > 0:
        adjusted = forecast + z * std_estimate
    else:
        # Fallback: assume CV=0.3 for std estimation
        adjusted = forecast + z * (_FALLBACK_CV * forecast)

    return max(0.0, adjusted)


def apply_policy(
    forecasts_df: pd.DataFrame,
    policy_cfg: PolicyConfig,
) -> pd.DataFrame:
    """
    Apply ordering policy to a DataFrame of raw model forecasts.

    This function:
    1. Adjusts each forecast using quantile scaling (policy's risk target).
    2. Computes safety stock from the forecast × safety_stock_multiplier.
    3. Determines max_cover_days from policy or product expiration.
    4. Calls recommend_orders() to produce final order_qty.

    Required columns:
        forecast, stock_balance, expiration_days, shipment_multiple

    Optional columns used if present:
        rolling_std_7d (for quantile adjustment)

    Args:
        forecasts_df: DataFrame with one row per (date, sku, store) combination.
        policy_cfg: Loaded policy configuration.

    Returns:
        DataFrame with adjusted_forecast, safety_stock, and order_qty columns.
    """
    df = forecasts_df.copy()
    mode = policy_cfg.mode
    logger.info("Applying policy mode='%s' (quantile=%.2f, safety_mult=%.1f)",
                mode, policy_cfg.forecast_quantile, policy_cfg.safety_stock_multiplier)

    # --- Step 1: Quantile-adjust the forecast ---
    std_col = "rolling_std_7d"
    has_std = std_col in df.columns

    adjusted_forecasts = []
    for _, row in df.iterrows():
        std_est = float(row[std_col]) if has_std and pd.notna(row[std_col]) else None
        adj = adjust_forecast_for_policy(
            forecast=float(row[FORECAST_COL]),
            std_estimate=std_est,
            policy_cfg=policy_cfg,
        )
        adjusted_forecasts.append(adj)

    df["adjusted_forecast"] = adjusted_forecasts

    # --- Step 2: Safety stock = adjusted_forecast × multiplier ---
    df["safety_stock"] = df["adjusted_forecast"] * policy_cfg.safety_stock_multiplier

    # --- Step 3: Determine max_cover_days ---
    # Policy can override; otherwise derive from expiration_days + mode
    if policy_cfg.max_cover_days is not None:
        max_cover = policy_cfg.max_cover_days
        df["_max_cover"] = max_cover
    else:
        # Derive from expiration_days based on policy mode
        if mode == "service_first":
            df["_max_cover"] = df[COL_EXPIRATION_DAYS].apply(lambda e: int(e) if pd.notna(e) else 1)
        elif mode == "balanced":
            df["_max_cover"] = df[COL_EXPIRATION_DAYS].apply(
                lambda e: max(1, int(e) - 1) if pd.notna(e) else 1
            )
        elif mode == "waste_first":
            df["_max_cover"] = df[COL_EXPIRATION_DAYS].apply(
                lambda e: max(1, int(e) - 2) if pd.notna(e) else 1
            )
        else:
            df["_max_cover"] = df[COL_EXPIRATION_DAYS].apply(lambda e: int(e) if pd.notna(e) else 1)

    # --- Step 4: Compute order quantities row by row ---
    orders = []
    for _, row in df.iterrows():
        qty = compute_order(
            forecast=float(row["adjusted_forecast"]),
            stock_balance=float(row[COL_STOCK_BALANCE]),
            expiration_days=int(row.get(COL_EXPIRATION_DAYS, 1)),
            safety_stock=float(row["safety_stock"]),
            shipment_multiple=int(row.get(COL_SHIPMENT_MULTIPLE, 1)),
            max_cover_days=int(row["_max_cover"]),
            round_up=policy_cfg.round_up_shipment,
        )
        orders.append(qty)

    df[ORDER_COL] = orders
    df.drop(columns=["_max_cover"], inplace=True)

    logger.info(
        "Policy '%s' applied: total order=%d, non-zero skus=%d",
        mode, sum(orders), sum(1 for o in orders if o > 0),
    )
    return df
