"""
Shared test fixtures.

Provides small synthetic DataFrames that mimic the real data schema.
These fixtures are used across all test modules.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    COL_DATE, COL_DEMAND_ADJUSTED, COL_EXPIRATION_DAYS, COL_IS_CENSORED,
    COL_LOSS_AMOUNT, COL_LOSS_QTY, COL_PRODUCT_GROUP, COL_SALES_AMOUNT,
    COL_SALES_QTY, COL_SHIPMENT_MULTIPLE, COL_SKU_CODE, COL_SKU_NAME,
    COL_STOCK_BALANCE, COL_STORE_ID, COL_UNIT_OF_MEASURE,
)
from src.utils.config import CensoringConfig, PolicyConfig


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """
    Synthetic dataset: 2 SKUs × 30 days.
    SKU-A: regular demand, occasional stockouts.
    SKU-B: stable demand, no stockouts.
    """
    np.random.seed(42)
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    rows = []
    for d in dates:
        # SKU-A: simulated stockouts on some days
        stock_a = 0 if d.weekday() == 6 else np.random.randint(5, 20)  # Zero on Sundays
        sales_a = min(np.random.randint(8, 15), max(0, stock_a))
        rows.append({
            COL_DATE: pd.Timestamp(d),
            COL_STORE_ID: "default_store",
            COL_SKU_CODE: "SKU-A",
            COL_SKU_NAME: "Test Product A",
            COL_PRODUCT_GROUP: "Пироги, пирожки, пончики",
            COL_SALES_QTY: float(sales_a),
            COL_SALES_AMOUNT: float(sales_a * 100),
            COL_STOCK_BALANCE: float(stock_a),
            COL_LOSS_QTY: 0.0,
            COL_LOSS_AMOUNT: 0.0,
            COL_EXPIRATION_DAYS: 3,
            COL_SHIPMENT_MULTIPLE: 1,
            COL_UNIT_OF_MEASURE: "шт",
        })
        # SKU-B: stable demand, always enough stock
        rows.append({
            COL_DATE: pd.Timestamp(d),
            COL_STORE_ID: "default_store",
            COL_SKU_CODE: "SKU-B",
            COL_SKU_NAME: "Test Product B",
            COL_PRODUCT_GROUP: "Торты",
            COL_SALES_QTY: float(np.random.randint(3, 7)),
            COL_SALES_AMOUNT: float(np.random.randint(3, 7) * 200),
            COL_STOCK_BALANCE: 20.0,
            COL_LOSS_QTY: 0.0,
            COL_LOSS_AMOUNT: 0.0,
            COL_EXPIRATION_DAYS: 5,
            COL_SHIPMENT_MULTIPLE: 2,
            COL_UNIT_OF_MEASURE: "шт",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def censoring_config() -> CensoringConfig:
    return CensoringConfig(strategy="impute", rolling_window=7, use_dow_grouping=False)


@pytest.fixture
def policy_service_first() -> PolicyConfig:
    return PolicyConfig(
        mode="service_first",
        safety_stock_multiplier=1.5,
        forecast_quantile=0.85,
        max_cover_days=None,
        round_up_shipment=True,
    )


@pytest.fixture
def policy_balanced() -> PolicyConfig:
    return PolicyConfig(
        mode="balanced",
        safety_stock_multiplier=1.0,
        forecast_quantile=0.55,
        max_cover_days=None,
        round_up_shipment=True,
    )


@pytest.fixture
def policy_waste_first() -> PolicyConfig:
    return PolicyConfig(
        mode="waste_first",
        safety_stock_multiplier=0.3,
        forecast_quantile=0.30,
        max_cover_days=1,
        round_up_shipment=False,
    )
