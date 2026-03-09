"""
Dataset builder: join sales, stock, losses, and product metadata into a single
daily fact table keyed by (date, store_id, sku_code).

Output schema matches src/data/schema.py REQUIRED_DATASET_COLS.

Design decisions:
- Left-join from sales as the primary table (sales are the events we model).
- Stock and losses are enrichment: missing = 0.
- Product metadata is a dimension: missing = sensible defaults (expiration=1, shipment=1).
- store_id is injected as a constant since the Forecasto API is single-store.
  The column is present so the codebase is drop-in ready for multi-store.
- Final output is sorted by (sku_code, date) for predictable lag computation.

Data quality handling:
- Ghost SKUs: The stock endpoint tracks a different (smaller) set of SKUs than sales.
  Including stock-only SKUs in the grid creates tens of thousands of zero-demand rows
  that teach the model "this SKU never sells." The grid is built from SALES SKUs only.
- Negative sales: Return/correction entries (sales_qty < 0) are clipped to 0.
- Losses dtype: The losses API returns qty/amount as strings; coerced to float here.
- Stock data availability: A `stock_data_available` flag marks dates where the stock
  API returned at least one non-zero balance. The API returns all-zeros for historical
  dates (Apr 2025 – Jan 2026), so this flag tells the model when stock is trustworthy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import (
    COL_ARTICLE, COL_DATE, COL_EXPIRATION_DAYS, COL_IS_CENSORED, COL_LOSS_AMOUNT,
    COL_LOSS_QTY, COL_LOSS_REASON, COL_PRODUCT_GROUP, COL_SHIPMENT_MULTIPLE,
    COL_SKU_CODE, COL_SKU_NAME, COL_STOCK_BALANCE, COL_STORE_ID, COL_UNIT_OF_MEASURE,
    COL_SALES_AMOUNT, COL_SALES_QTY,
)
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Joins raw source DataFrames into the processed dataset.

    Usage:
        builder = DatasetBuilder(config)
        dataset = builder.build(sales_df, stock_df, losses_df, products_df)
        builder.save(dataset)
    """

    def __init__(self, config: Config):
        self._cfg = config
        self._processed_dir = Path(config.data.processed_dir)
        self._store_id = config.data.default_store_id

    def build(
        self,
        sales_df: pd.DataFrame,
        stock_df: pd.DataFrame,
        losses_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join all source DataFrames into a single processed dataset.

        Args:
            sales_df: from DataLoader.load_sales()
            stock_df: from DataLoader.load_stock()
            losses_df: from DataLoader.load_losses()
            products_df: from DataLoader.load_products()

        Returns:
            DataFrame keyed by (date, store_id, sku_code), sorted by (sku_code, date).
        """
        logger.info("Building dataset from %d sales, %d stock, %d loss, %d product rows",
                    len(sales_df), len(stock_df), len(losses_df), len(products_df))

        sales = self._prepare_sales(sales_df)
        stock = self._prepare_stock(stock_df)
        losses = self._prepare_losses(losses_df)
        products = self._prepare_products(products_df)

        # Build base: expand to full (date x sku) grid so every combination
        # has a row even on zero-sale days (important for lag features)
        full_grid = self._build_full_grid(sales, stock)

        # Merge stock on (date, sku_code)
        df = full_grid.merge(stock, on=[COL_DATE, COL_SKU_CODE], how="left", suffixes=("", "_stock"))

        # Merge losses on (date, sku_code)
        df = df.merge(losses, on=[COL_DATE, COL_SKU_CODE], how="left")

        # Merge product metadata on sku_code
        df = df.merge(products, on=COL_SKU_CODE, how="left", suffixes=("", "_prod"))

        # Fill missing numerics with sensible defaults
        df[COL_SALES_QTY] = df[COL_SALES_QTY].fillna(0)
        df[COL_SALES_AMOUNT] = df[COL_SALES_AMOUNT].fillna(0)
        df[COL_STOCK_BALANCE] = df[COL_STOCK_BALANCE].fillna(0)
        df[COL_LOSS_QTY] = df[COL_LOSS_QTY].fillna(0)
        df[COL_LOSS_AMOUNT] = df[COL_LOSS_AMOUNT].fillna(0)

        # Stock data availability flag.
        # The stock API returns all-zeros for months where it has no historical data
        # (confirmed: Apr 2025 – Jan 2026 are 100% zero). Mark dates where at least
        # one SKU had non-zero stock so the model knows when to trust stock_balance.
        dates_with_stock = df.loc[df[COL_STOCK_BALANCE] > 0, COL_DATE].unique()
        df["stock_data_available"] = df[COL_DATE].isin(dates_with_stock).astype("int8")
        n_stock_dates = len(dates_with_stock)
        n_total_dates = df[COL_DATE].nunique()
        logger.info(
            "Stock data available on %d / %d dates (%.0f%% of date range)",
            n_stock_dates, n_total_dates, 100 * n_stock_dates / max(n_total_dates, 1),
        )
        # Perishable default: 1-day shelf life (safest assumption)
        df[COL_EXPIRATION_DAYS] = df[COL_EXPIRATION_DAYS].fillna(1)
        df[COL_SHIPMENT_MULTIPLE] = df[COL_SHIPMENT_MULTIPLE].fillna(1)
        df[COL_UNIT_OF_MEASURE] = df[COL_UNIT_OF_MEASURE].fillna("шт")

        # Ensure product_group is populated (fall back to sales-side group if available)
        if f"{COL_PRODUCT_GROUP}_prod" in df.columns:
            df[COL_PRODUCT_GROUP] = df[COL_PRODUCT_GROUP].fillna(df[f"{COL_PRODUCT_GROUP}_prod"])
            df.drop(columns=[f"{COL_PRODUCT_GROUP}_prod"], inplace=True)
        df[COL_PRODUCT_GROUP] = df[COL_PRODUCT_GROUP].fillna("Unknown")

        # Similarly resolve sku_name
        if f"{COL_SKU_NAME}_prod" in df.columns:
            df[COL_SKU_NAME] = df[COL_SKU_NAME].fillna(df[f"{COL_SKU_NAME}_prod"])
            df.drop(columns=[f"{COL_SKU_NAME}_prod"], inplace=True)

        # Inject store_id (single-store: constant across all rows)
        df[COL_STORE_ID] = self._store_id

        # Sort for deterministic lag computation
        df.sort_values([COL_SKU_CODE, COL_DATE], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self._validate(df)
        logger.info("Dataset built: %d rows, %d SKUs, date range %s → %s",
                    len(df),
                    df[COL_SKU_CODE].nunique(),
                    df[COL_DATE].min().date(),
                    df[COL_DATE].max().date())
        return df

    def save(self, df: pd.DataFrame, filename: str = "dataset.parquet") -> Path:
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        out = self._processed_dir / filename
        df.to_parquet(out, index=False)
        logger.info("Dataset saved to %s", out)
        return out

    def load(self, filename: str = "dataset.parquet") -> pd.DataFrame:
        path = self._processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {path}. "
                "Run: python -m src.cli.main build-dataset"
            )
        df = pd.read_parquet(path)
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_string_columns(self, df: pd.DataFrame, cols: list[str]) -> None:
        """Strip leading/trailing whitespace from string columns (in-place). Preserves NaN."""
        for c in cols:
            if c in df.columns and df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    def _prepare_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate duplicate (date, sku_code) rows by summing quantities."""
        df = df.copy()
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])

        self._strip_string_columns(df, [COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_ARTICLE])

        # Clip negative sales_qty to 0. Negative values are return/correction
        # entries (e.g., a cancelled sale recorded as -1). They are valid accounting
        # entries but not useful as demand observations.
        n_negative = (df[COL_SALES_QTY] < 0).sum()
        if n_negative > 0:
            logger.warning(
                "Clipping %d negative sales_qty rows to 0 (return/correction entries)",
                n_negative,
            )
            df[COL_SALES_QTY] = df[COL_SALES_QTY].clip(lower=0)
            df[COL_SALES_AMOUNT] = df[COL_SALES_AMOUNT].clip(lower=0)

        agg = {
            COL_SALES_QTY: "sum",
            COL_SALES_AMOUNT: "sum",
        }
        # Carry forward string cols from first occurrence
        str_cols = [c for c in [COL_SKU_NAME, COL_PRODUCT_GROUP] if c in df.columns]
        for c in str_cols:
            agg[c] = "first"

        result = df.groupby([COL_DATE, COL_SKU_CODE], as_index=False).agg(agg)
        return result

    def _prepare_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep latest stock balance per (date, sku_code) if duplicates exist."""
        df = df.copy()
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        self._strip_string_columns(df, [COL_SKU_CODE, COL_SKU_NAME])
        return df.groupby([COL_DATE, COL_SKU_CODE], as_index=False)[COL_STOCK_BALANCE].last()

    def _prepare_losses(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[COL_DATE, COL_SKU_CODE, COL_LOSS_QTY, COL_LOSS_AMOUNT])
        df = df.copy()
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        self._strip_string_columns(df, [COL_SKU_CODE, COL_LOSS_REASON])

        # The losses API returns qty and amount as strings (object dtype).
        # Coerce to float; invalid entries become NaN and are filled with 0.
        for col in [COL_LOSS_QTY, COL_LOSS_AMOUNT]:
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.groupby([COL_DATE, COL_SKU_CODE], as_index=False).agg(
            {COL_LOSS_QTY: "sum", COL_LOSS_AMOUNT: "sum"}
        )

    def _prepare_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate product metadata; one row per sku_code."""
        df = df.copy()
        self._strip_string_columns(df, [COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_UNIT_OF_MEASURE])
        return df.drop_duplicates(subset=[COL_SKU_CODE], keep="last")

    def _build_full_grid(self, sales: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
        """
        Create a complete (date × sku) grid from SALES SKUs only, crossed with
        all dates in the sales date range.

        This ensures zero-sale days are represented as rows with sales_qty=0,
        which is critical for proper lag and rolling-window computation.

        Why sales-only (not union with stock):
        The stock API tracks a much smaller set of SKUs than the sales API
        (88 overlap out of 555 sales SKUs and 218 stock SKUs). Including
        stock-only SKUs adds ~130 × 341 = ~44,000 rows that have zero sales
        on every single day — these "ghost SKUs" teach the model that those
        codes never sell, polluting lag features and pushing forecasts toward 0.
        """
        sales_skus = sales[COL_SKU_CODE].unique()
        stock_only_skus = set(stock[COL_SKU_CODE].unique()) - set(sales_skus)

        if stock_only_skus:
            logger.info(
                "Excluding %d stock-only SKUs from grid (no sales history): %s%s",
                len(stock_only_skus),
                ", ".join(sorted(stock_only_skus)[:5]),
                " ..." if len(stock_only_skus) > 5 else "",
            )

        all_dates = pd.date_range(
            sales[COL_DATE].min(),
            sales[COL_DATE].max(),
            freq="D",
        )

        grid = pd.MultiIndex.from_product(
            [all_dates, sales_skus], names=[COL_DATE, COL_SKU_CODE]
        ).to_frame(index=False)

        # Left join sales onto grid so we get NaN where no sale occurred
        merged = grid.merge(sales, on=[COL_DATE, COL_SKU_CODE], how="left")
        return merged

    def _validate(self, df: pd.DataFrame) -> None:
        from src.data.schema import REQUIRED_DATASET_COLS
        missing = [c for c in REQUIRED_DATASET_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if df[[COL_DATE, COL_STORE_ID, COL_SKU_CODE]].duplicated().any():
            n_dups = df[[COL_DATE, COL_STORE_ID, COL_SKU_CODE]].duplicated().sum()
            logger.warning("%d duplicate (date, store_id, sku_code) rows found — check data", n_dups)
