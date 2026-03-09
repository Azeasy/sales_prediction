"""
CLI entry point for the Auto-Order MVP.

All commands load config from configs/base.yaml and optionally merge
a policy override file. Use --config to specify a different base config.

Examples
--------
# Build dataset from local samples (default)
python -m src.cli.main build-dataset

# Train the LightGBM model
python -m src.cli.main train

# Predict next-day demand
python -m src.cli.main predict

# Generate order recommendations with service_first policy
python -m src.cli.main recommend-order --policy service_first

# Run a backtest simulation
python -m src.cli.main backtest --start-date 2026-02-01 --end-date 2026-03-07

# Fetch data from the live API
python -m src.cli.main fetch-data --start-date 2026-01-01 --end-date 2026-03-07
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

import click
import pandas as pd

from src.data.schema import COL_PRODUCT_GROUP, COL_SKU_CODE, COL_SKU_NAME
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = get_logger(__name__)

VALID_POLICIES = ["service_first", "balanced", "waste_first"]


def _parse_date(date_str: str) -> date:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise click.BadParameter(f"Cannot parse date '{date_str}'. Use YYYY-MM-DD format.")


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", "config_path", default=None, help="Path to override YAML config.")
@click.option("--log-level", default="INFO", help="Log level: DEBUG, INFO, WARNING, ERROR")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, log_level: str):
    """Auto-Order MVP: demand forecasting + order recommendations for perishable products."""
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


# ---------------------------------------------------------------------------
# fetch-data
# ---------------------------------------------------------------------------

def _default_start_date() -> str:
    """Default: 4 months ago."""
    from datetime import timedelta
    d = date.today() - timedelta(days=4 * 30)
    return d.strftime("%Y-%m-%d")


@cli.command("fetch-data")
@click.option("--start-date", default=None, help="Start date YYYY-MM-DD. Default: 4 months ago.")
@click.option("--end-date", default=None, help="End date YYYY-MM-DD. Default: today.")
@click.pass_context
def fetch_data(ctx: click.Context, start_date: str | None, end_date: str | None):
    """Fetch raw data from Forecasto APIs and cache to data/raw/.

    Requires FORECASTO_TOKEN environment variable to be set.
    For offline development, use build-dataset directly (uses local samples).
    """
    from src.pipeline import step_fetch_data

    config = load_config(base_path=ctx.obj.get("config_path"), policy=None)
    if config.data.source != "api":
        click.echo("Warning: config.data.source is not 'api'. Set it to 'api' in your config to fetch live data.")
        click.echo("Continuing with API fetch regardless.")

    sd = _parse_date(start_date or _default_start_date())
    ed = _parse_date(end_date or date.today().strftime("%Y-%m-%d"))

    click.echo(f"Fetching data from {sd} to {ed}...")
    step_fetch_data(config, sd, ed)
    click.echo("Done. Raw data cached to data/raw/")


# ---------------------------------------------------------------------------
# match-products
# ---------------------------------------------------------------------------

@cli.command("match-products")
@click.option("--fuzzy-threshold", default=92, help="Min token_set_ratio for fuzzy match (0-100)")
@click.option("--embed-threshold", default=0.92, help="Min cosine similarity for embedding match")
@click.option("--no-embeddings", is_flag=True, help="Skip embedding pass (fuzzy only)")
@click.option("--output", default=None, help="Save mapping to this JSON path. Default: artifacts/product_mapping.json")
@click.pass_context
def match_products(ctx: click.Context, fuzzy_threshold: float, embed_threshold: float, no_embeddings: bool, output: str | None):
    """Run hybrid product matching (fuzzy + embeddings) and save sku_code -> canonical_sku mapping.

    Use the mapping in build-dataset by setting product_matching.enabled: true in config.
    """
    from src.data.loader import DataLoader
    from src.data.product_matching import run_product_matching, save_mapping

    config = load_config(base_path=ctx.obj.get("config_path"))
    loader = DataLoader(config)

    click.echo("Loading products for matching...")
    sales = loader.load_sales()
    products = loader.load_products()

    # Build product list: sales first (richer), products for any missing
    if not sales.empty:
        df = sales[[COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP]].drop_duplicates(COL_SKU_CODE, keep="first")
    else:
        df = pd.DataFrame(columns=[COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP])

    if not products.empty:
        prod = products[[COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP]].drop_duplicates(COL_SKU_CODE, keep="first")
        existing = set(df[COL_SKU_CODE].astype(str).str.strip()) if not df.empty else set()
        add = prod[~prod[COL_SKU_CODE].astype(str).str.strip().isin(existing)]
        if len(add) > 0:
            df = pd.concat([df, add], ignore_index=True)

    if df.empty:
        click.echo("No sales or products data. Run fetch-data or use local samples.")
        return

    df[COL_PRODUCT_GROUP] = df[COL_PRODUCT_GROUP].fillna("Unknown")

    click.echo(f"Matching {len(df)} products (fuzzy threshold={fuzzy_threshold}, embed={embed_threshold})...")
    mapping = run_product_matching(
        df,
        fuzzy_threshold=fuzzy_threshold,
        embed_threshold=embed_threshold,
        use_embeddings=not no_embeddings,
    )

    out_path = Path(output) if output else Path(config.artifacts.dir) / "product_mapping.json"
    save_mapping(mapping, out_path)
    click.echo(f"Mapping saved to {out_path}")
    n_merged = sum(1 for k, v in mapping.items() if k != v)
    click.echo(f"Merged {n_merged} SKUs into canonical forms. Set product_matching.enabled: true to apply.")


# ---------------------------------------------------------------------------
# build-dataset
# ---------------------------------------------------------------------------

@cli.command("build-dataset")
@click.pass_context
def build_dataset(ctx: click.Context):
    """Join sales, stock, losses, and product info into processed dataset.

    In local mode (default), reads from data/samples/.
    Output: data/processed/dataset.parquet
    """
    from src.pipeline import step_build_dataset

    config = load_config(base_path=ctx.obj.get("config_path"))
    click.echo(f"Building dataset (source={config.data.source})...")
    dataset = step_build_dataset(config)
    click.echo(f"Dataset built: {len(dataset)} rows, {dataset['sku_code'].nunique()} SKUs")
    click.echo(f"  Date range: {dataset['date'].min().date()} → {dataset['date'].max().date()}")
    click.echo("  Saved to: data/processed/dataset.parquet")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command("train")
@click.pass_context
def train(ctx: click.Context):
    """Train the LightGBM demand forecasting model.

    Reads data/processed/dataset.parquet, applies censoring adjustment,
    builds features, trains model, and saves artifacts to artifacts/.
    """
    from src.pipeline import step_train

    config = load_config(base_path=ctx.obj.get("config_path"))
    click.echo(f"Training model (type={config.model.type}, censoring={config.censoring.strategy})...")
    val_metrics = step_train(config)
    click.echo("Training complete.")
    click.echo("Validation metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            fmt = f"{v:.2%}" if k in ("wape", "waste_rate", "service_level") else f"{v:+.4f}"
            click.echo(f"  {k}: {fmt}")
    click.echo(f"Artifacts saved to: {config.artifacts.dir}/")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@cli.command("predict")
@click.option("--date", "target_date", default=None, help="Date to predict for (YYYY-MM-DD). Defaults to latest.")
@click.option("--output", default=None, help="Save predictions to this CSV path.")
@click.pass_context
def predict(ctx: click.Context, target_date: str | None, output: str | None):
    """Generate next-day demand forecasts for all SKUs.

    Uses the trained model from artifacts/. Run `train` first.
    """
    from src.pipeline import step_predict

    config = load_config(base_path=ctx.obj.get("config_path"))
    click.echo("Generating forecasts...")
    forecasts = step_predict(config, target_date)

    click.echo(f"\n{'SKU Code':<12} {'Forecast':>10} {'Stock':>10}")
    click.echo("-" * 36)
    for _, row in forecasts.head(20).iterrows():
        click.echo(f"{row['sku_code']:<12} {row['forecast']:>10.1f} {row['stock_balance']:>10.1f}")
    if len(forecasts) > 20:
        click.echo(f"  ... and {len(forecasts) - 20} more SKUs")

    if output:
        forecasts.to_csv(output, index=False)
        click.echo(f"\nForecasts saved to {output}")


# ---------------------------------------------------------------------------
# recommend-order
# ---------------------------------------------------------------------------

@cli.command("recommend-order")
@click.option("--policy", default="balanced", type=click.Choice(VALID_POLICIES),
              help="Ordering policy mode.")
@click.option("--date", "target_date", default=None, help="Date to recommend for. Defaults to latest.")
@click.option("--output", default=None, help="Save recommendations to this CSV path.")
@click.pass_context
def recommend_order(ctx: click.Context, policy: str, target_date: str | None, output: str | None):
    """Generate order quantity recommendations using the selected policy.

    Combines demand forecast + current stock + product shelf life + policy mode.
    Run `train` first.
    """
    from src.pipeline import step_recommend_order

    config = load_config(base_path=ctx.obj.get("config_path"), policy=policy)
    click.echo(f"Generating order recommendations (policy={policy})...")
    recommendations = step_recommend_order(config, target_date)

    click.echo(f"\n{'SKU Code':<12} {'Forecast':>10} {'Stock':>8} {'Order Qty':>10}")
    click.echo("-" * 44)
    for _, row in recommendations.head(20).iterrows():
        click.echo(
            f"{row['sku_code']:<12} {row['forecast']:>10.1f} "
            f"{row['stock_balance']:>8.1f} {row['order_qty']:>10d}"
        )
    if len(recommendations) > 20:
        click.echo(f"  ... and {len(recommendations) - 20} more SKUs")

    total_order = recommendations["order_qty"].sum()
    click.echo(f"\nTotal units to order: {total_order}")
    click.echo(f"SKUs with non-zero orders: {(recommendations['order_qty'] > 0).sum()}")

    if output:
        recommendations.to_csv(output, index=False)
        click.echo(f"\nRecommendations saved to {output}")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@cli.command("backtest")
@click.option("--policy", default="balanced", type=click.Choice(VALID_POLICIES),
              help="Ordering policy to evaluate.")
@click.option("--start-date", default=None, help="Backtest start date YYYY-MM-DD.")
@click.option("--end-date", default=None, help="Backtest end date YYYY-MM-DD.")
@click.pass_context
def backtest(ctx: click.Context, policy: str, start_date: str | None, end_date: str | None):
    """Run backtest simulation: forecast + order + simulate fulfillment.

    Produces a per-day per-SKU results parquet and a summary text file in artifacts/.
    Run `train` first.
    """
    from src.pipeline import step_backtest

    config = load_config(base_path=ctx.obj.get("config_path"), policy=policy)
    click.echo(f"Running backtest (policy={policy})...")
    if start_date or end_date:
        click.echo(f"  Date range: {start_date or 'earliest'} → {end_date or 'latest'}")

    results_df, summary = step_backtest(config, start_date, end_date)

    click.echo("\nBacktest Results:")
    click.echo("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            if k in ("wape", "waste_rate", "service_level"):
                click.echo(f"  {k:<30}: {v:.2%}")
            else:
                click.echo(f"  {k:<30}: {v:+.3f}")
        else:
            click.echo(f"  {k:<30}: {v}")
    click.echo(f"\nResults saved to artifacts/")


if __name__ == "__main__":
    cli()
