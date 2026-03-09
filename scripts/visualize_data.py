#!/usr/bin/env python3
"""
Visualize raw and processed data for the auto-order MVP.

Run from project root:
    python scripts/visualize_data.py

Outputs figures to scripts/figures/ (created if missing).

Requires: matplotlib (add to requirements.txt or install separately)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "scripts" / "figures"


def load_raw_data():
    """Load all raw parquet files from data/raw."""
    sales_files = sorted(RAW_DIR.glob("sales_*.parquet"))
    stock_files = sorted(RAW_DIR.glob("stock_*.parquet"))
    losses_files = sorted(RAW_DIR.glob("losses_*.parquet"))
    products_path = RAW_DIR / "products.parquet"

    sales = pd.concat([pd.read_parquet(f) for f in sales_files], ignore_index=True) if sales_files else pd.DataFrame()
    stock = pd.concat([pd.read_parquet(f) for f in stock_files], ignore_index=True) if stock_files else pd.DataFrame()
    losses = pd.concat([pd.read_parquet(f) for f in losses_files], ignore_index=True) if losses_files else pd.DataFrame()
    products = pd.read_parquet(products_path) if products_path.exists() else pd.DataFrame()

    for df, name in [(sales, "sales"), (stock, "stock"), (losses, "losses")]:
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    return sales, stock, losses, products


def load_processed():
    """Load the built dataset if it exists."""
    path = PROCESSED_DIR / "dataset.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_sales_over_time(sales: pd.DataFrame, out_dir: Path):
    """Daily and weekly sales volume over time."""
    if sales.empty:
        return
    daily = sales.groupby("date")["sales_qty"].sum()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].fill_between(daily.index, daily.values, alpha=0.6)
    axes[0].set_ylabel("Units sold")
    axes[0].set_title("Daily total sales volume")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    weekly = daily.resample("W").sum()
    axes[1].bar(weekly.index, weekly.values, width=5, alpha=0.7)
    axes[1].set_ylabel("Units sold")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Weekly total sales volume")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "01_sales_over_time.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_sales_over_time.png")


def plot_stock_availability(stock: pd.DataFrame, out_dir: Path):
    """Stock data availability by month (zeros vs non-zeros)."""
    if stock.empty:
        return
    stock = stock.copy()
    stock["month"] = pd.to_datetime(stock["date"]).dt.to_period("M")
    by_month = stock.groupby("month").agg(
        total=("stock_balance", "count"),
        nonzero=("stock_balance", lambda x: (x > 0).sum()),
    )
    by_month["pct_nonzero"] = 100 * by_month["nonzero"] / by_month["total"]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(by_month))
    ax.bar(x, by_month["pct_nonzero"], color="steelblue", alpha=0.8, label="% rows with stock > 0")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in by_month.index], rotation=45)
    ax.set_ylabel("% of stock rows with balance > 0")
    ax.set_xlabel("Month")
    ax.set_title("Stock API data availability by month\n(0% = API returned all zeros for that period)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_stock_availability_by_month.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_stock_availability_by_month.png")


def plot_sku_overlap(sales: pd.DataFrame, stock: pd.DataFrame, out_dir: Path):
    """Venn-style view of SKU overlap between sales and stock."""
    if sales.empty or stock.empty:
        return
    sales_skus = set(sales["sku_code"].unique())
    stock_skus = set(stock["sku_code"].unique())
    both = sales_skus & stock_skus
    sales_only = sales_skus - stock_skus
    stock_only = stock_skus - sales_skus

    fig, ax = plt.subplots(figsize=(8, 5))
    # Simple bar chart
    labels = ["Sales only\n(no stock)", "Both\n(sales + stock)", "Stock only\n(ghost SKUs)"]
    counts = [len(sales_only), len(both), len(stock_only)]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(labels, counts, color=colors, alpha=0.8)
    ax.set_ylabel("Number of SKUs")
    ax.set_title("SKU overlap: Sales vs Stock APIs\n(ghost SKUs = in stock but never sold)")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, str(c), ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "03_sku_overlap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_sku_overlap.png")


def plot_top_skus(sales: pd.DataFrame, out_dir: Path, n: int = 20):
    """Top N SKUs by total sales volume."""
    if sales.empty:
        return
    top = sales.groupby("sku_code")["sales_qty"].sum().nlargest(n)
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(top))
    ax.barh(y_pos, top.values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total units sold")
    ax.set_title(f"Top {n} SKUs by sales volume")
    plt.tight_layout()
    plt.savefig(out_dir / "04_top_skus_by_sales.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_top_skus_by_sales.png")


def plot_losses(losses: pd.DataFrame, out_dir: Path):
    """Losses over time (if any)."""
    if losses.empty or "loss_qty" not in losses.columns:
        return
    df = losses.copy()
    df["loss_qty"] = pd.to_numeric(df["loss_qty"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["loss_qty"].sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(daily.index, daily.values, width=0.8, color="coral", alpha=0.8)
    ax.set_ylabel("Units written off")
    ax.set_xlabel("Date")
    ax.set_title("Daily losses (write-offs / expired)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "05_losses_over_time.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_losses_over_time.png")


def plot_dataset_summary(dataset: pd.DataFrame, out_dir: Path):
    """Processed dataset: rows per day, stock_data_available."""
    if dataset is None or dataset.empty:
        return
    daily_rows = dataset.groupby("date").size()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(daily_rows.index, daily_rows.values, alpha=0.8)
    axes[0].set_ylabel("Rows per day")
    axes[0].set_title("Processed dataset: (date × SKU) grid size per day")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    if "stock_data_available" in dataset.columns:
        pct = dataset.groupby("date")["stock_data_available"].mean() * 100
        axes[1].fill_between(pct.index, pct.values, alpha=0.6, color="green")
        axes[1].set_ylabel("% of rows with stock data")
        axes[1].set_xlabel("Date")
        axes[1].set_title("Stock data availability (1 = at least one SKU had stock > 0 that day)")
        axes[1].set_ylim(0, 105)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "06_dataset_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_dataset_summary.png")


def plot_sales_distribution(sales: pd.DataFrame, out_dir: Path):
    """Distribution of sales_qty per (date, SKU) — shows zero-inflation."""
    if sales.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    qty = sales["sales_qty"]
    axes[0].hist(qty[qty > 0], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Sales qty (positive only)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of non-zero sales per row")

    zero_pct = 100 * (qty == 0).mean()
    axes[1].pie(
        [(qty > 0).sum(), (qty == 0).sum()],
        labels=[f"Sales > 0\n({(qty > 0).sum()} rows)", f"Sales = 0\n({(qty == 0).sum()} rows)"],
        autopct="%1.1f%%",
        colors=["steelblue", "lightgray"],
    )
    axes[1].set_title("Zero vs non-zero sales rows")
    plt.tight_layout()
    plt.savefig(out_dir / "07_sales_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_sales_distribution.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Visualizing data → {FIGURES_DIR}\n")

    sales, stock, losses, products = load_raw_data()
    dataset = load_processed()

    print("Raw data loaded:")
    print(f"  Sales:  {len(sales):,} rows, {sales['sku_code'].nunique() if not sales.empty else 0} SKUs")
    print(f"  Stock:  {len(stock):,} rows, {stock['sku_code'].nunique() if not stock.empty else 0} SKUs")
    print(f"  Losses: {len(losses):,} rows")
    if dataset is not None:
        print(f"  Dataset: {len(dataset):,} rows, {dataset['sku_code'].nunique()} SKUs")
    print()

    plot_sales_over_time(sales, FIGURES_DIR)
    plot_stock_availability(stock, FIGURES_DIR)
    plot_sku_overlap(sales, stock, FIGURES_DIR)
    plot_top_skus(sales, FIGURES_DIR)
    plot_losses(losses, FIGURES_DIR)
    plot_dataset_summary(dataset, FIGURES_DIR)
    plot_sales_distribution(sales, FIGURES_DIR)

    print(f"\nDone. Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
