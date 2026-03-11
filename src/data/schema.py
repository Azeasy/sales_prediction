"""
Internal data schema constants.

Defines canonical column names and expected dtypes for all DataFrames
that flow through the pipeline. Centralizing these prevents silent
column-name bugs across modules.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

# Identifiers
COL_DATE = "date"
COL_STORE_ID = "store_id"
COL_SKU_CODE = "sku_code"
COL_SKU_NAME = "sku_name"
COL_ARTICLE = "article"

# Sales
COL_SALES_QTY = "sales_qty"
COL_SALES_AMOUNT = "sales_amount"
COL_PRODUCT_GROUP = "product_group"
COL_SUB_GROUP = "sub_group"            # product line/family within product_group
COL_BASE_PRODUCT_NAME = "base_product_name"  # size-agnostic product identity
COL_ITEM_TYPE = "item_type"
COL_WEIGHT = "weight"

# Stock
COL_STOCK_BALANCE = "stock_balance"

# Losses
COL_LOSS_QTY = "loss_qty"
COL_LOSS_AMOUNT = "loss_amount"
COL_LOSS_REASON = "loss_reason"

# Product metadata
COL_EXPIRATION_DAYS = "expiration_days"
COL_SHIPMENT_MULTIPLE = "shipment_multiple"
COL_UNIT_OF_MEASURE = "unit_of_measure"
COL_MIN_STOCK_LEVEL = "min_stock_level"
COL_NEEDS_FRIDGE = "needs_fridge"
COL_NEEDS_FREEZER = "needs_freezer"

# Censoring
COL_IS_CENSORED = "is_censored"
COL_DEMAND_ADJUSTED = "demand_adjusted"

# Primary keys of the processed dataset
DATASET_PK = [COL_DATE, COL_STORE_ID, COL_SKU_CODE]

# Minimum columns required for a valid processed dataset
REQUIRED_DATASET_COLS = [
    COL_DATE, COL_STORE_ID, COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP,
    COL_SALES_QTY, COL_SALES_AMOUNT, COL_STOCK_BALANCE,
    COL_LOSS_QTY, COL_LOSS_AMOUNT,
    COL_EXPIRATION_DAYS, COL_SHIPMENT_MULTIPLE, COL_UNIT_OF_MEASURE,
]
