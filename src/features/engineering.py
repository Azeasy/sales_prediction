"""
Feature engineering for the demand forecasting model.

All features are computed per-SKU and are strictly backward-looking —
no future leakage of any kind.

Implementation note: all per-group operations use groupby().shift() and
groupby().transform() instead of groupby().apply(). This avoids a pandas 2.2
behavior change where apply() drops the groupby key column from the result,
which caused a KeyError crash on real data. transform() always preserves
the full DataFrame structure.

Feature groups
--------------
calendar        — day_of_week, is_weekend, day_of_month, month, week_of_year
lags            — demand_adjusted at lag [1, 2, 3, 7, 14] days
rolling         — rolling mean / std of demand_adjusted over [7, 14, 28] day windows
                  (window computed on observations shifted 1 day back)
censoring       — censored_rate_7d, days_since_last_stockout
metadata        — expiration_days, product_group (categorical), shipment_multiple
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import (
    COL_DATE, COL_DEMAND_ADJUSTED, COL_EXPIRATION_DAYS, COL_IS_CENSORED,
    COL_PRODUCT_GROUP, COL_SHIPMENT_MULTIPLE, COL_SKU_CODE, COL_SALES_QTY,
    COL_STOCK_BALANCE,
)
from src.utils.config import FeaturesConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Categorical columns that LightGBM handles natively
CATEGORICAL_FEATURES = ["product_group", "sub_group", "base_product_name", "day_of_week"]


def build_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """
    Add all feature columns to the dataset.

    Assumes the input is sorted by (sku_code, date) — DatasetBuilder guarantees this.
    Mutates a copy; does not modify input.

    Args:
        df: Processed dataset (after censoring adjustment).
        cfg: FeaturesConfig controlling which lags/windows to generate.

    Returns:
        DataFrame with all feature columns appended.
    """
    df = df.copy()
    df = df.sort_values([COL_SKU_CODE, COL_DATE]).reset_index(drop=True)

    df = _add_calendar_features(df)
    df = _add_lag_features(df, cfg.lags, cfg.target_col)
    df = _add_rolling_features(df, cfg.rolling_windows, cfg.target_col)
    df = _add_censoring_features(df)
    df = _add_metadata_features(df)

    # Convert categoricals to category dtype (LightGBM reads this natively)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    n_feat = len(get_feature_columns(df))
    logger.info("Feature engineering complete: %d feature columns generated", n_feat)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of model feature column names present in df.
    Excludes identifiers, targets, and raw source columns.
    """
    exclude = {
        COL_DATE, COL_SKU_CODE, "store_id", "sku_name", "article",
        COL_SALES_QTY, "sales_amount", "loss_qty",
        "loss_amount", "loss_reason", "item_type", "weight",
        COL_DEMAND_ADJUSTED, COL_IS_CENSORED,
        "unit_of_measure", "min_stock_level", "needs_fridge", "needs_freezer",
        "days_count",
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Feature group implementations
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df[COL_DATE])
    df["day_of_week"] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def _add_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    target_col: str,
) -> pd.DataFrame:
    """
    Add lag features per SKU using groupby().shift().

    Uses shift() directly on a grouped Series — this is the pandas 2.x safe
    pattern that never drops the sku_code column (unlike groupby().apply()).
    """
    source_col = target_col if target_col in df.columns else COL_SALES_QTY

    for lag in lags:
        df[f"lag_{lag}d"] = df.groupby(COL_SKU_CODE)[source_col].shift(lag)

    return df


def _add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    target_col: str,
) -> pd.DataFrame:
    """
    Add rolling mean and std per SKU using groupby().transform().

    Shift(1) is applied first (stored in a temp column) so the current
    day's value is never included in any rolling window.

    Uses transform() instead of apply() — transform() always preserves the
    full DataFrame structure and never drops the groupby key column.
    """
    source_col = target_col if target_col in df.columns else COL_SALES_QTY

    # Compute the shifted series once per group, store as a temp column
    df["_rolling_src"] = df.groupby(COL_SKU_CODE)[source_col].shift(1)

    for w in windows:
        df[f"rolling_mean_{w}d"] = (
            df.groupby(COL_SKU_CODE)["_rolling_src"]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}d"] = (
            df.groupby(COL_SKU_CODE)["_rolling_src"]
            .transform(lambda x: x.rolling(window=w, min_periods=1).std())
        )

    df.drop(columns=["_rolling_src"], inplace=True)
    return df


def _add_censoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Censoring-aware features that help the model learn about stockout context.

    censored_rate_7d: fraction of last 7 days that were censored (per SKU).
    days_since_last_stockout: count of consecutive non-zero-stock days (per SKU).

    Uses transform() to avoid the groupby.apply() key-dropping issue.
    Uses a cumsum groupby trick instead of iterrows() for performance with 200k+ rows.
    """
    if COL_IS_CENSORED not in df.columns:
        logger.debug("is_censored not found; skipping censoring features")
        return df

    # Censored rate over last 7 days (shifted to exclude current day)
    df["_cens_shifted"] = df.groupby(COL_SKU_CODE)[COL_IS_CENSORED].transform(
        lambda x: x.astype(float).shift(1)
    )
    df["censored_rate_7d"] = (
        df.groupby(COL_SKU_CODE)["_cens_shifted"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )
    df.drop(columns=["_cens_shifted"], inplace=True)

    # Days since last stockout — vectorized via cumsum block trick
    # prev_stock == 0 (or NaN on first row) marks a "reset" event.
    # cumsum() of resets creates monotonically increasing block IDs.
    # cumcount() within (sku, block) gives days elapsed since that reset.
    if COL_STOCK_BALANCE in df.columns:
        df["_prev_stock"] = df.groupby(COL_SKU_CODE)[COL_STOCK_BALANCE].shift(1)
        df["_stockout_reset"] = (df["_prev_stock"].isna() | (df["_prev_stock"] == 0)).astype(int)
        df["_block"] = df.groupby(COL_SKU_CODE)["_stockout_reset"].cumsum()
        df["days_since_last_stockout"] = df.groupby([COL_SKU_CODE, "_block"]).cumcount()
        df.drop(columns=["_prev_stock", "_stockout_reset", "_block"], inplace=True)

    return df


def _add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Product metadata as features: expiration_days, shipment_multiple.
    product_group is already a column; it becomes a categorical feature.
    """
    # expiration_days and shipment_multiple are already in the dataset.
    # product_group is converted to category dtype in build_features().
    return df
