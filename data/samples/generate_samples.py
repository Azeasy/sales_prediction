"""
Generate realistic mock sample data for offline development.

Run once:  python data/samples/generate_samples.py

Produces:
  data/samples/sales.parquet
  data/samples/stock.parquet
  data/samples/products.parquet
  data/samples/losses.parquet

All files mimic the internal schema (English column names) as returned
by the API client after parse_*_record() normalization.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

SAMPLES_DIR = Path(__file__).parent
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 3, 7)   # ~66 days of history

# 12 SKUs covering different product groups and shelf lives
PRODUCTS = [
    {"sku_code": "PIR01", "sku_name": "Пирог Русский с ягодной начинкой 0.3кг",    "product_group": "Пироги, пирожки, пончики",  "expiration_days": 3, "shipment_multiple": 1, "unit_of_measure": "шт"},
    {"sku_code": "PIR02", "sku_name": "Пирожок с мясом 0.1кг",                     "product_group": "Пироги, пирожки, пончики",  "expiration_days": 2, "shipment_multiple": 5, "unit_of_measure": "шт"},
    {"sku_code": "TORT01","sku_name": "Торт Вишенка 0.5кг",                        "product_group": "Торты",                     "expiration_days": 5, "shipment_multiple": 1, "unit_of_measure": "шт"},
    {"sku_code": "TORT02","sku_name": "Торт Наполеон 0.8кг",                       "product_group": "Торты",                     "expiration_days": 4, "shipment_multiple": 1, "unit_of_measure": "шт"},
    {"sku_code": "XL01",  "sku_name": "Хлеб белый формовой 0.5кг",                 "product_group": "Хлеб и хлебобулочные",      "expiration_days": 3, "shipment_multiple": 10,"unit_of_measure": "шт"},
    {"sku_code": "XL02",  "sku_name": "Хлеб ржаной 0.35кг",                        "product_group": "Хлеб и хлебобулочные",      "expiration_days": 5, "shipment_multiple": 10,"unit_of_measure": "шт"},
    {"sku_code": "BUL01", "sku_name": "Булочка с маком 0.08кг",                    "product_group": "Булочки",                   "expiration_days": 2, "shipment_multiple": 6, "unit_of_measure": "шт"},
    {"sku_code": "BUL02", "sku_name": "Ватрушка творожная 0.1кг",                  "product_group": "Булочки",                   "expiration_days": 2, "shipment_multiple": 6, "unit_of_measure": "шт"},
    {"sku_code": "BEL01", "sku_name": "Беляш Омский 0.4кг",                        "product_group": "ЗАМОРОЖЕННЫЕ ПОЛУФАБРИКАТЫ","expiration_days": 7, "shipment_multiple": 1, "unit_of_measure": "шт"},
    {"sku_code": "KRE01", "sku_name": "Круассан с шоколадом 0.08кг",               "product_group": "Слоёные и дрожжевые",       "expiration_days": 3, "shipment_multiple": 4, "unit_of_measure": "шт"},
    {"sku_code": "PEC01", "sku_name": "Печенье овсяное 0.4кг",                     "product_group": "Печенье и вафли",           "expiration_days": 30,"shipment_multiple": 1, "unit_of_measure": "кг"},
    {"sku_code": "PONCH", "sku_name": "Пончик с сахаром 0.07кг",                   "product_group": "Пироги, пирожки, пончики",  "expiration_days": 1, "shipment_multiple": 6, "unit_of_measure": "шт"},
]

# Base daily demand per SKU (units) — roughly calibrated for a mid-size bakery
BASE_DEMAND = {
    "PIR01":  12, "PIR02":  25, "TORT01": 4,  "TORT02": 3,
    "XL01":   40, "XL02":   30, "BUL01": 35,  "BUL02": 28,
    "BEL01":  10, "KRE01":  20, "PEC01": 8,   "PONCH": 45,
}

DOW_MULTIPLIER = {0: 0.85, 1: 0.90, 2: 0.95, 3: 1.00, 4: 1.10, 5: 1.40, 6: 1.30}  # Mon–Sun


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _simulate_demand(sku: str, d: date) -> float:
    base = BASE_DEMAND[sku]
    dow_mult = DOW_MULTIPLIER[d.weekday()]
    # Add seasonal trend: slight uptick in March
    seasonal = 1 + 0.1 * (d.month - 1) / 12
    noise = np.random.normal(1.0, 0.2)
    return max(0.0, base * dow_mult * seasonal * noise)


def generate():
    dates = list(_date_range(START_DATE, END_DATE))

    sales_rows = []
    stock_rows = []
    loss_rows = []

    # Simulate carry-over stock state
    stock_state: dict[str, float] = {p["sku_code"]: BASE_DEMAND[p["sku_code"]] * 1.5 for p in PRODUCTS}

    for d in dates:
        for prod in PRODUCTS:
            sku = prod["sku_code"]
            exp_days = prod["expiration_days"]
            true_demand = _simulate_demand(sku, d)

            avail = stock_state.get(sku, 0.0)
            observed_sales = min(true_demand, avail)
            is_stockout = observed_sales < true_demand and avail < true_demand * 0.9

            # Update stock
            stock_after = max(0.0, avail - observed_sales)

            # Simulate waste: if product is very short-lived and weekend dip didn't sell
            waste = 0.0
            if exp_days <= 2 and d.weekday() == 0:  # Monday: yesterday's unsold perishables
                waste = min(stock_after, max(0.0, np.random.exponential(2)))

            stock_after = max(0.0, stock_after - waste)

            # Simulate next-day order (simplified reorder logic for sample data)
            reorder = BASE_DEMAND[sku] * DOW_MULTIPLIER.get((d.weekday() + 1) % 7, 1.0)
            reorder = max(0.0, reorder + np.random.normal(0, 2))
            stock_state[sku] = stock_after + reorder

            # Sales row
            if observed_sales > 0 or (not is_stockout):
                sales_rows.append({
                    "date": d,
                    "sku_code": sku,
                    "sku_name": prod["sku_name"],
                    "sales_qty": round(observed_sales, 2),
                    "sales_amount": round(observed_sales * np.random.uniform(80, 200), 2),
                    "product_group": prod["product_group"],
                    "article": sku,
                    "item_type": "Товар",
                    "weight": None,
                })

            # Stock row (end of day balance)
            stock_rows.append({
                "date": d,
                "sku_code": sku,
                "sku_name": prod["sku_name"],
                "stock_balance": round(max(0.0, stock_after), 2),
            })

            # Loss row
            if waste > 0.01:
                loss_rows.append({
                    "date": d,
                    "sku_code": sku,
                    "loss_qty": round(waste, 2),
                    "loss_amount": round(waste * np.random.uniform(80, 200), 2),
                    "loss_reason": "loss",
                })

    # Products table
    products_df = pd.DataFrame(PRODUCTS)

    sales_df = pd.DataFrame(sales_rows)
    stock_df = pd.DataFrame(stock_rows)
    loss_df = pd.DataFrame(loss_rows) if loss_rows else pd.DataFrame(
        columns=["date", "sku_code", "loss_qty", "loss_amount", "loss_reason"]
    )

    # Convert dates to datetime
    for df in [sales_df, stock_df, loss_df]:
        if "date" in df.columns and not df.empty:
            df["date"] = pd.to_datetime(df["date"])

    sales_df.to_parquet(SAMPLES_DIR / "sales.parquet", index=False)
    stock_df.to_parquet(SAMPLES_DIR / "stock.parquet", index=False)
    loss_df.to_parquet(SAMPLES_DIR / "losses.parquet", index=False)
    products_df.to_parquet(SAMPLES_DIR / "products.parquet", index=False)

    print(f"Generated {len(sales_df)} sales rows")
    print(f"Generated {len(stock_df)} stock rows")
    print(f"Generated {len(loss_df)} loss rows")
    print(f"Generated {len(products_df)} product rows")
    print(f"Files saved to {SAMPLES_DIR.resolve()}")


if __name__ == "__main__":
    generate()
