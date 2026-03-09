"""
Censored demand detection and adjustment.

Background
----------
In perishable retail, a stockout caps what we can observe.  When a product
runs out mid-day, recorded sales reflect inventory constraints rather than
true customer demand.  Training a forecaster on censored sales causes it to
systematically under-predict — the model learns "on low-stock days, sales
are low" when the reality is "on low-stock days, we stopped selling".

Detection heuristic
-------------------
A row is flagged `is_censored = True` when:
  1. stock_balance == 0  (classic stockout: nothing left)
  2. OR stock_balance > 0 AND stock_balance <= sales_qty
     (sales consumed all available stock — implicitly capped)

Note: zero-sale days with non-zero stock are NOT censored (genuine low demand).

Three adjustment strategies (config-driven)
-------------------------------------------
none   → demand_adjusted = sales_qty (no correction; simplest baseline)
drop   → demand_adjusted = NaN on censored rows → excluded from model training
impute → demand_adjusted = max(sales_qty, estimate) where estimate is:
          rolling_median(sales over last N days, same SKU, censored rows masked)
          optionally further refined by same-day-of-week rolling median

Imputation guarantee: demand_adjusted is NEVER lower than observed sales_qty.
This is the key correctness constraint — we never say "true demand was less than
what we actually sold."

Trade-offs documented in GUIDE.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import (
    COL_DATE, COL_DEMAND_ADJUSTED, COL_IS_CENSORED, COL_SALES_QTY,
    COL_SKU_CODE, COL_STOCK_BALANCE,
)
from src.utils.config import CensoringConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_STRATEGIES = {"none", "drop", "impute"}


_HIGH_CENSORING_WARNING_THRESHOLD = 0.70  # warn when >70% rows are flagged


def detect_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `is_censored` boolean column to the dataset.

    A record is censored when observable sales were capped by stock constraints.
    Operates in-place on a copy; does not modify the input DataFrame.

    Heuristic (two conditions, either triggers censoring):
    1. stock_constrained: stock_balance > 0 AND stock_balance <= sales_qty AND sales > 0
       → Stock existed but was entirely consumed during the period.
         This is the strongest daily signal: supply was the bottleneck.
    2. stockout_with_sales: stock_balance == 0 AND sales_qty > 0
       → End-of-period stock is zero AND something was sold.
         For perishable bakery items, daily sell-through to zero is *normal* and
         is NOT necessarily a censoring event (the bakery may simply have ordered
         the right amount).  This condition is therefore weaker.

    ⚠️  If >70% of rows are flagged, the heuristic is almost certainly picking up
    normal sell-through rather than genuine stockouts.  In that case consider:
      • censoring.strategy: none  (use raw sales as target — safest default)
      • Investigating whether stock_balance reflects opening or closing inventory.

    Args:
        df: Processed dataset with at least stock_balance and sales_qty columns.

    Returns:
        DataFrame with is_censored column added.
    """
    df = df.copy()

    # Condition 1: stock existed but was entirely consumed by sales
    stock_constrained = (
        (df[COL_STOCK_BALANCE] > 0)
        & (df[COL_STOCK_BALANCE] <= df[COL_SALES_QTY])
        & (df[COL_SALES_QTY] > 0)
    )

    # Condition 2: end-of-period stock is zero AND some sales occurred
    # (sell-through to zero — weaker signal for bakery data)
    stockout_with_sales = (df[COL_STOCK_BALANCE] == 0) & (df[COL_SALES_QTY] > 0)

    df[COL_IS_CENSORED] = stock_constrained | stockout_with_sales

    n_censored = int(df[COL_IS_CENSORED].sum())
    n_total = max(len(df), 1)
    pct = 100.0 * n_censored / n_total
    logger.info("Censored demand: %d rows flagged (%.1f%%)", n_censored, pct)

    if pct / 100 > _HIGH_CENSORING_WARNING_THRESHOLD:
        logger.warning(
            "⚠️  Very high censoring rate (%.0f%%). "
            "For perishable bakery items, daily sell-through to zero stock is "
            "normal behaviour — NOT a stockout.  The censoring heuristic is likely "
            "flagging legitimate sales as censored, which causes imputation to "
            "over-estimate demand.\n"
            "  → Recommended fix: set  censoring.strategy: none  in base.yaml\n"
            "    This uses raw observed sales as the training target (no adjustment).\n"
            "    Re-run:  python -m src.cli.main train",
            pct,
        )

    return df


def adjust_target(df: pd.DataFrame, cfg: CensoringConfig) -> pd.DataFrame:
    """
    Create `demand_adjusted` column using the configured censoring strategy.

    Must be called after detect_censoring() so that `is_censored` exists.

    Args:
        df: Dataset with is_censored column present.
        cfg: CensoringConfig from the loaded YAML config.

    Returns:
        DataFrame with demand_adjusted column added.
    """
    if COL_IS_CENSORED not in df.columns:
        raise ValueError("detect_censoring() must be called before adjust_target()")

    strategy = cfg.strategy
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown censoring strategy '{strategy}'. Must be one of {VALID_STRATEGIES}")

    df = df.copy()

    if strategy == "none":
        df[COL_DEMAND_ADJUSTED] = df[COL_SALES_QTY]
        logger.info("Censoring strategy=none: using raw sales_qty as target")

    elif strategy == "drop":
        df[COL_DEMAND_ADJUSTED] = df[COL_SALES_QTY].where(~df[COL_IS_CENSORED], other=np.nan)
        n_dropped = df[COL_IS_CENSORED].sum()
        logger.info("Censoring strategy=drop: %d censored rows set to NaN (excluded from training)", n_dropped)

    elif strategy == "impute":
        df = _impute_censored(df, cfg)

    return df


# ---------------------------------------------------------------------------
# Imputation implementation
# ---------------------------------------------------------------------------

def _impute_censored(df: pd.DataFrame, cfg: CensoringConfig) -> pd.DataFrame:
    """
    Replace censored sales with a rolling estimate of true demand.

    Algorithm per SKU:
      1. Mask censored rows (treat as NaN) to avoid circular bias.
      2. Compute rolling median of the last `rolling_window` uncensored days.
      3. If use_dow_grouping=True, also compute a same-weekday rolling median
         over the same window (uses last N same-weekday obs that are uncensored).
      4. Final estimate = max(rolling_median, dow_median).
      5. demand_adjusted = max(sales_qty, estimate) — never below observed.

    Note on rolling direction: all windows look backward only (shift(1) before
    rolling) so there is zero future leakage even at inference time.
    """
    df = df.copy()
    df = df.sort_values([COL_SKU_CODE, COL_DATE]).reset_index(drop=True)

    window = cfg.rolling_window
    use_dow = cfg.use_dow_grouping

    def _estimate_sku(grp: pd.DataFrame) -> pd.Series:
        """Compute imputed demand estimate for a single SKU group."""
        sales = grp[COL_SALES_QTY].copy()
        is_censored = grp[COL_IS_CENSORED]

        # Mask censored values so they don't feed into the rolling estimate
        uncensored_sales = sales.where(~is_censored, other=np.nan)

        # Rolling median on uncensored obs (shift 1 to exclude current day)
        rolling_est = (
            uncensored_sales
            .shift(1)
            .rolling(window=window, min_periods=1)
            .median()
        )

        estimate = rolling_est

        if use_dow:
            dow = pd.to_datetime(grp[COL_DATE]).dt.dayofweek
            dow_est = pd.Series(index=grp.index, dtype=float)
            for day in range(7):
                mask = dow == day
                dow_sales = uncensored_sales.where(mask, other=np.nan)
                # For each dow, rolling median on that weekday's obs
                dow_rolling = (
                    dow_sales
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .median()
                )
                dow_est = dow_est.combine(dow_rolling, lambda a, b: b if pd.notna(b) else a)

            # Combine: take element-wise max of both estimates
            estimate = estimate.combine(dow_est, lambda a, b: max(
                a if pd.notna(a) else 0,
                b if pd.notna(b) else 0,
            ))

        return estimate

    # Apply per SKU — use transform-style to guarantee index alignment
    # We build estimates column-by-column to avoid groupby.apply index issues
    df["_est"] = np.nan

    for sku, grp_idx in df.groupby(COL_SKU_CODE).groups.items():
        grp = df.loc[grp_idx]
        est = _estimate_sku(grp)
        # est may come back with the group's original positional index; realign
        df.loc[grp_idx, "_est"] = est.values

    # demand_adjusted: censored rows → max(observed, estimate); others → observed
    sales = df[COL_SALES_QTY].values
    est_vals = df["_est"].values
    is_cens = df[COL_IS_CENSORED].values

    demand_adj = sales.copy().astype(float)
    for i in range(len(demand_adj)):
        if is_cens[i]:
            est = est_vals[i] if not np.isnan(est_vals[i]) else sales[i]
            demand_adj[i] = max(sales[i], est)

    df[COL_DEMAND_ADJUSTED] = demand_adj
    df.drop(columns=["_est"], inplace=True)

    n_imputed = int(is_cens.sum())
    cens_idx = np.where(is_cens)[0]
    avg_lift = float(np.mean(demand_adj[cens_idx] - sales[cens_idx])) if n_imputed > 0 else 0.0
    logger.info(
        "Censoring strategy=impute: %d rows imputed, avg demand lift=+%.2f units",
        n_imputed, avg_lift if not np.isnan(avg_lift) else 0,
    )
    return df
