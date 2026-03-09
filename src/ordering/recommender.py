"""
Order quantity recommender.

Core function: compute_order()
-------------------------------
This is the business logic that converts a demand forecast into an order
recommendation. It is intentionally decoupled from the forecasting model —
the same ordering logic works with any forecast source (LightGBM, naive, human).

Key principle: forecast != order
---------------------------------
Even a perfect forecast doesn't produce the optimal order directly because:
  1. We already have stock on hand (usable_stock reduces what we need to order)
  2. Perishability: stock expiring before the next delivery is worthless
  3. Shipment constraints: packs come in multiples (e.g., bags of 6 buns)
  4. Policy intent: the safety stock and coverage caps differ by mode

Order formula:
  target_stock = adjusted_forecast + safety_stock
  target_stock = min(target_stock, forecast × max_cover_days)  [waste cap]
  usable_stock = stock_balance if expiration_days > 1 else 0   [discard expiring]
  raw_order = max(0, target_stock - usable_stock)
  order_qty = round(raw_order, shipment_multiple, direction)

Batch recommender: recommend_orders()
--------------------------------------
Applies compute_order() row-by-row to a DataFrame of forecasts,
returning a new DataFrame with order_qty added.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.data.schema import (
    COL_EXPIRATION_DAYS, COL_SHIPMENT_MULTIPLE, COL_SKU_CODE,
    COL_STOCK_BALANCE,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

FORECAST_COL = "forecast"
ORDER_COL = "order_qty"


def compute_order(
    forecast: float,
    stock_balance: float,
    expiration_days: int,
    safety_stock: float,
    shipment_multiple: int = 1,
    max_cover_days: Optional[int] = None,
    round_up: bool = True,
) -> int:
    """
    Compute recommended order quantity for a single (store, sku, date) combination.

    Args:
        forecast: Predicted demand for the next delivery period (units).
        stock_balance: Current stock on hand (units).
        expiration_days: Shelf life in days (from product metadata).
        safety_stock: Extra stock buffer to add on top of forecast (units).
        shipment_multiple: Orders must be multiples of this pack size.
        max_cover_days: Cap order to cover at most this many days of demand.
                        None = no cap (limited only by expiration_days).
        round_up: If True, round up to nearest shipment_multiple (service_first).
                  If False, round down (waste_first).

    Returns:
        Non-negative integer order quantity aligned to shipment_multiple.
    """
    forecast = max(0.0, float(forecast))
    stock_balance = max(0.0, float(stock_balance))
    expiration_days = max(1, int(expiration_days))
    shipment_multiple = max(1, int(shipment_multiple))

    # Stock expiring today or already expired is unusable
    usable_stock = stock_balance if expiration_days > 1 else 0.0

    # Target stock: forecast demand + safety buffer
    target_stock = forecast + safety_stock

    # Waste cap: never order more than N days of expected coverage
    if max_cover_days is not None and max_cover_days >= 1:
        cover_cap = forecast * max_cover_days
        target_stock = min(target_stock, cover_cap)

    # Also cap at what can realistically sell before product expires
    # (if a product has 3-day shelf life, ordering 10-day worth is wasteful)
    perishability_cap = forecast * expiration_days
    target_stock = min(target_stock, perishability_cap)

    raw_order = max(0.0, target_stock - usable_stock)

    # Align to shipment multiple
    if shipment_multiple > 1:
        if round_up:
            aligned = math.ceil(raw_order / shipment_multiple) * shipment_multiple
        else:
            aligned = math.floor(raw_order / shipment_multiple) * shipment_multiple
    else:
        aligned = raw_order

    return int(aligned)


def recommend_orders(
    forecasts_df: pd.DataFrame,
    safety_stock_col: str = "safety_stock",
    max_cover_days: Optional[int] = None,
    round_up: bool = True,
) -> pd.DataFrame:
    """
    Apply compute_order() to each row in a forecast DataFrame.

    Required columns in forecasts_df:
        forecast, stock_balance, expiration_days, shipment_multiple

    Optional columns:
        safety_stock (defaults to 0.0 if missing)

    Returns:
        Input DataFrame with `order_qty` column appended.
    """
    df = forecasts_df.copy()

    if safety_stock_col not in df.columns:
        df[safety_stock_col] = 0.0

    orders = []
    for _, row in df.iterrows():
        qty = compute_order(
            forecast=row[FORECAST_COL],
            stock_balance=row[COL_STOCK_BALANCE],
            expiration_days=int(row.get(COL_EXPIRATION_DAYS, 1)),
            safety_stock=float(row.get(safety_stock_col, 0.0)),
            shipment_multiple=int(row.get(COL_SHIPMENT_MULTIPLE, 1)),
            max_cover_days=max_cover_days,
            round_up=round_up,
        )
        orders.append(qty)

    df[ORDER_COL] = orders
    logger.info(
        "Order recommendations: %d SKUs, total order=%d units, "
        "avg order=%.1f, non-zero orders=%d",
        df[COL_SKU_CODE].nunique() if COL_SKU_CODE in df.columns else len(df),
        sum(orders),
        sum(orders) / max(len(orders), 1),
        sum(1 for o in orders if o > 0),
    )
    return df
