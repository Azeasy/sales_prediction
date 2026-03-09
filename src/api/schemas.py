"""
API field mappings: Russian API fields → internal English schema.

Each function takes raw API response JSON and returns a normalized dict
with English column names, ready for DataFrame construction.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Sales API  /sales
# ---------------------------------------------------------------------------

SALES_FIELD_MAP = {
    "Период": "date",
    "Номенклатура": "sku_name",
    "Количество": "sales_qty",
    "Сумма": "sales_amount",
    "ВидНоменклатуры": "item_type",
    "Код": "sku_code",
    "Артикул": "article",
    "Группа": "product_group",
    "Вес": "weight",
}


def parse_sales_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single sales API record to internal schema."""
    out: dict[str, Any] = {}
    for ru_key, en_key in SALES_FIELD_MAP.items():
        out[en_key] = record.get(ru_key)
    return out


# ---------------------------------------------------------------------------
# Stock API  /inventory/stock
# ---------------------------------------------------------------------------

STOCK_FIELD_MAP = {
    "Date": "date",
    "Code": "sku_code",
    "Name": "sku_name",
    "balance": "stock_balance",
}


def parse_stock_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for api_key, en_key in STOCK_FIELD_MAP.items():
        out[en_key] = record.get(api_key)
    return out


# ---------------------------------------------------------------------------
# Loss API  /loss/getall
# ---------------------------------------------------------------------------

LOSS_FIELD_MAP = {
    "Date": "date",
    "item_id": "sku_code",
    "loss": "loss_qty",
    "totalloss": "loss_amount",
    "reason": "loss_reason",
}


def parse_loss_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for api_key, en_key in LOSS_FIELD_MAP.items():
        out[en_key] = record.get(api_key)
    return out


# ---------------------------------------------------------------------------
# Product Info API  /backend/delivery_info/api/v1/GetAll
# ---------------------------------------------------------------------------
# The item_information field is a list of single-key dicts, e.g.:
#   [{"ExpirationDays": "7"}, {"UnitOfMeasure": "шт"}, {"Shipment": 1}, ...]
# We flatten that list into a plain dict first.

PRODUCT_INFO_KEYS = {
    "ExpirationDays": "expiration_days",
    "UnitOfMeasure": "unit_of_measure",
    "Shipment": "shipment_multiple",
    "ProductGroup": "product_group",
    "MinStockLevel": "min_stock_level",
    "Needfridge": "needs_fridge",
    "Needfreezer": "needs_freezer",
    "DaysCount": "days_count",
}


def _flatten_item_information(info_list: list[dict]) -> dict[str, Any]:
    """Flatten [{Key: val}, ...] list into {Key: val, ...} dict."""
    flat: dict[str, Any] = {}
    for entry in info_list:
        flat.update(entry)
    return flat


def parse_product_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a product info record, flattening the nested item_information."""
    out: dict[str, Any] = {
        "sku_code": record.get("item_code"),
        "sku_name": record.get("item_name"),
    }
    flat_info = _flatten_item_information(record.get("item_information", []))
    for info_key, en_key in PRODUCT_INFO_KEYS.items():
        raw_val = flat_info.get(info_key)
        # ExpirationDays comes as a string from the API
        if info_key == "ExpirationDays" and raw_val is not None:
            try:
                raw_val = int(raw_val)
            except (ValueError, TypeError):
                raw_val = None
        out[en_key] = raw_val
    return out
