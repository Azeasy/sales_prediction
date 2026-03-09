#!/usr/bin/env python3
"""
Print fields that may need cleansing — product names, codes, and other
text/numeric fields that might hide dirty data.

Run from project root:
    python scripts/inspect_cleansing_candidates.py

Output is plain text to stdout. Pipe to a file or less for review:
    python scripts/inspect_cleansing_candidates.py > cleansing_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

import pandas as pd


def load_raw():
    """Load raw parquet files."""
    sales_files = sorted(RAW_DIR.glob("sales_*.parquet"))
    stock_files = sorted(RAW_DIR.glob("stock_*.parquet"))
    losses_files = sorted(RAW_DIR.glob("losses_*.parquet"))
    products_path = RAW_DIR / "products.parquet"

    sales = pd.concat([pd.read_parquet(f) for f in sales_files], ignore_index=True) if sales_files else pd.DataFrame()
    stock = pd.concat([pd.read_parquet(f) for f in stock_files], ignore_index=True) if stock_files else pd.DataFrame()
    losses = pd.concat([pd.read_parquet(f) for f in losses_files], ignore_index=True) if losses_files else pd.DataFrame()
    products = pd.read_parquet(products_path) if products_path.exists() else pd.DataFrame()

    return sales, stock, losses, products


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def sub(title: str):
    print(f"\n--- {title} ---")


def show_sample(series, n: int = 15, label: str = ""):
    """Print sample of unique values."""
    uniq = series.dropna().astype(str).unique()
    print(f"  Unique count: {len(uniq)}")
    for i, v in enumerate(uniq[:n]):
        display = repr(v)[:80] + ("..." if len(repr(v)) > 80 else "")
        print(f"    {i+1:2}. {display}")
    if len(uniq) > n:
        print(f"    ... and {len(uniq) - n} more")


def inspect_text_field(df: pd.DataFrame, col: str, source: str):
    """Inspect a text column for cleansing candidates."""
    if col not in df.columns:
        return
    s = df[col].astype(str)

    sub(f"{source}.{col}")

    # Empty / null / whitespace
    empty = s.str.strip() == ""
    null_count = df[col].isna().sum()
    if empty.any() or null_count > 0:
        print(f"  ⚠ Empty or null: {empty.sum() + null_count} rows")
        if empty.any():
            print(f"    (whitespace-only: {empty.sum()})")

    # Leading/trailing whitespace
    has_leading = s != s.str.lstrip()
    has_trailing = s != s.str.rstrip()
    if has_leading.any() or has_trailing.any():
        print(f"  ⚠ Leading/trailing whitespace: {has_leading.sum()} leading, {has_trailing.sum()} trailing")
        for v in s[has_leading | has_trailing].unique()[:5]:
            print(f"    Example: {repr(v)}")

    # Very long values
    long = s.str.len() > 100
    if long.any():
        print(f"  ⚠ Very long (>100 chars): {long.sum()} rows")
        for v in s[long].unique()[:3]:
            print(f"    {repr(v[:120])}...")

    # Sample of unique values
    show_sample(df[col], n=20, label=col)


def inspect_sku_name_consistency(sales: pd.DataFrame, stock: pd.DataFrame, products: pd.DataFrame):
    """Same sku_code, different sku_name across sources."""
    section("SKU_CODE ↔ SKU_NAME INCONSISTENCY (same code, different names)")

    def collect(df, source):
        if df.empty or "sku_code" not in df.columns or "sku_name" not in df.columns:
            return {}
        return df.groupby("sku_code")["sku_name"].apply(lambda x: x.dropna().astype(str).unique().tolist()).to_dict()

    sales_map = collect(sales, "sales")
    stock_map = collect(stock, "stock")
    prod_map = collect(products, "products")

    all_codes = set(sales_map) | set(stock_map) | set(prod_map)
    issues = []

    for code in sorted(all_codes):
        names = set()
        for m in [sales_map, stock_map, prod_map]:
            if code in m:
                for n in m[code]:
                    names.add(n.strip())
        if len(names) > 1:
            issues.append((code, names))

    if not issues:
        print("  None found.")
        return

    print(f"  Found {len(issues)} SKU codes with conflicting names:\n")
    for code, names in issues[:30]:
        print(f"  sku_code: {code}")
        for n in sorted(names):
            print(f"    → {repr(n)[:90]}")
        print()


def inspect_article_issues(sales: pd.DataFrame):
    """Article field: empty, duplicates, same article different sku."""
    section("ARTICLE FIELD (product identifier)")

    if sales.empty or "article" not in sales.columns:
        return

    art = sales["article"].astype(str)
    empty = (art == "") | (art == "nan") | (art.str.strip() == "")
    print(f"  Empty or NaN: {empty.sum()} / {len(sales)} rows ({100*empty.mean():.1f}%)")

    # Same article, different sku_code
    grp = sales.groupby("article")["sku_code"].nunique()
    multi = grp[grp > 1]
    if len(multi) > 0:
        sub("Same article, different sku_codes (possible duplicates or renames)")
        print(f"  {len(multi)} articles map to multiple sku_codes:\n")
        for art_val in multi.index[:15]:
            if str(art_val).strip() in ("", "nan"):
                continue
            skus = sales[sales["article"] == art_val]["sku_code"].unique().tolist()
            names = sales[sales["article"] == art_val]["sku_name"].iloc[0]
            print(f"  article={repr(art_val)[:40]}")
            print(f"    sku_codes: {skus}")
            print(f"    name: {repr(names)[:70]}")
            print()

    # Same sku_code, different articles
    grp2 = sales.groupby("sku_code")["article"].nunique()
    multi2 = grp2[grp2 > 1]
    if len(multi2) > 0:
        sub("Same sku_code, different articles")
        print(f"  {len(multi2)} sku_codes have multiple articles:\n")
        for sku in multi2.index[:10]:
            arts = sales[sales["sku_code"] == sku]["article"].unique().tolist()
            print(f"  sku_code={sku}  →  articles: {arts}")
        print()


def inspect_product_names(sales: pd.DataFrame, n_sample: int = 50):
    """Print product names for manual review — look for typos, extra chars."""
    section("PRODUCT NAMES (sku_name) — manual review")

    if sales.empty or "sku_name" not in sales.columns:
        return

    # One row per (sku_code, sku_name) for readability
    subset = sales[["sku_code", "sku_name", "product_group"]].drop_duplicates()
    subset = subset.sort_values(["product_group", "sku_name"])

    print(f"  Total unique (sku_code, sku_name) pairs: {len(subset)}\n")
    print("  Sample (sorted by product_group, sku_name):\n")

    for _, row in subset.head(n_sample).iterrows():
        print(f"  [{row['sku_code']:12}] {row['product_group'][:35]:35} | {row['sku_name'][:60]}")


def inspect_product_group(sales: pd.DataFrame):
    """Product group consistency and odd values."""
    section("PRODUCT_GROUP")

    if sales.empty or "product_group" not in sales.columns:
        return

    grp = sales["product_group"].astype(str)
    empty = (grp == "") | (grp == "nan")
    print(f"  Empty: {empty.sum()} rows")
    show_sample(sales["product_group"], n=40)


def inspect_numeric_anomalies(sales: pd.DataFrame, stock: pd.DataFrame, losses: pd.DataFrame):
    """Negative values, zeros, extreme values."""
    section("NUMERIC ANOMALIES")

    if not sales.empty and "sales_qty" in sales.columns:
        neg = (sales["sales_qty"] < 0).sum()
        if neg > 0:
            print(f"  ⚠ sales_qty < 0: {neg} rows (returns/corrections)")
            print(sales[sales["sales_qty"] < 0][["date", "sku_code", "sku_name", "sales_qty"]].head(10).to_string())
        else:
            print("  sales_qty: no negatives")

    if not stock.empty and "stock_balance" in stock.columns:
        neg = (stock["stock_balance"] < 0).sum()
        if neg > 0:
            print(f"  ⚠ stock_balance < 0: {neg} rows")
        else:
            print("  stock_balance: no negatives")

    if not losses.empty:
        for col in ["loss_qty", "loss_amount"]:
            if col in losses.columns:
                s = pd.to_numeric(losses[col], errors="coerce")
                nan_count = s.isna().sum()
                if nan_count > 0:
                    print(f"  ⚠ {col}: {nan_count} non-numeric (coerce to NaN)")
                neg = (s < 0).sum()
                if neg > 0:
                    print(f"  ⚠ {col} < 0: {neg} rows")


def inspect_loss_reason(losses: pd.DataFrame):
    """Loss reason values."""
    section("LOSS_REASON")

    if losses.empty or "loss_reason" not in losses.columns:
        return

    show_sample(losses["loss_reason"], n=20)


def inspect_item_type(sales: pd.DataFrame):
    """Item type values."""
    section("ITEM_TYPE")

    if sales.empty or "item_type" not in sales.columns:
        return

    show_sample(sales["item_type"], n=20)


def main():
    print("Data cleansing candidates report")
    print("Run: python scripts/inspect_cleansing_candidates.py")
    print("Data source: data/raw/*.parquet")

    sales, stock, losses, products = load_raw()

    if sales.empty and stock.empty:
        print("\nNo data found in data/raw/. Run fetch-data first.")
        return

    inspect_text_field(sales, "sku_name", "sales")
    inspect_text_field(sales, "sku_code", "sales")
    inspect_text_field(sales, "article", "sales")
    inspect_product_group(sales)
    inspect_item_type(sales)

    inspect_sku_name_consistency(sales, stock, products)
    inspect_article_issues(sales)
    inspect_product_names(sales, n_sample=80)

    inspect_numeric_anomalies(sales, stock, losses)
    inspect_loss_reason(losses)

    if not stock.empty:
        inspect_text_field(stock, "sku_name", "stock")
        inspect_text_field(stock, "sku_code", "stock")

    section("DONE")
    print("\nReview the output above. Focus on:")
    print("  - sku_name: typos, extra spaces, inconsistent naming")
    print("  - article: empty values, same article → different sku_codes")
    print("  - product_group: odd categories")
    print("  - sku_code ↔ sku_name: same code, different names across sales/stock/products")


if __name__ == "__main__":
    main()
