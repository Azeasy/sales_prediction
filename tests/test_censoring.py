"""Tests for censored demand detection and adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    COL_DEMAND_ADJUSTED, COL_IS_CENSORED, COL_SALES_QTY, COL_STOCK_BALANCE,
)
from src.demand.censoring import adjust_target, detect_censoring
from src.utils.config import CensoringConfig


def _make_df(sales, stocks):
    """Helper to create a minimal DataFrame for censoring tests."""
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(sales), freq="D"),
        "sku_code": ["SKU-A"] * len(sales),
        COL_SALES_QTY: sales,
        COL_STOCK_BALANCE: stocks,
    })


class TestDetectCensoring:
    def test_zero_stock_is_censored(self):
        df = _make_df([5.0], [0.0])
        result = detect_censoring(df)
        assert result.loc[0, COL_IS_CENSORED] is True or result.loc[0, COL_IS_CENSORED] == True

    def test_positive_stock_normal_sales_not_censored(self):
        df = _make_df([5.0], [20.0])
        result = detect_censoring(df)
        assert result.loc[0, COL_IS_CENSORED] is False or result.loc[0, COL_IS_CENSORED] == False

    def test_stock_equals_sales_is_censored(self):
        """Sold exactly as much as stock → stock constrained."""
        df = _make_df([10.0], [10.0])
        result = detect_censoring(df)
        assert result.loc[0, COL_IS_CENSORED] == True

    def test_stock_less_than_sales_is_censored(self):
        """Sold more than remaining stock (data inconsistency) → censored."""
        df = _make_df([12.0], [8.0])
        result = detect_censoring(df)
        assert result.loc[0, COL_IS_CENSORED] == True

    def test_zero_sales_positive_stock_not_censored(self):
        """Zero sales with stock available = genuine zero demand."""
        df = _make_df([0.0], [15.0])
        result = detect_censoring(df)
        assert result.loc[0, COL_IS_CENSORED] == False

    def test_original_not_modified(self):
        df = _make_df([5.0, 0.0], [0.0, 10.0])
        original_len = len(df)
        _ = detect_censoring(df)
        assert COL_IS_CENSORED not in df.columns  # Original untouched
        assert len(df) == original_len

    def test_multiple_rows_correct_flags(self):
        sales = [5.0, 5.0, 0.0, 5.0]
        stocks = [0.0, 20.0, 10.0, 5.0]
        df = _make_df(sales, stocks)
        result = detect_censoring(df)
        expected = [True, False, False, True]
        assert list(result[COL_IS_CENSORED]) == expected


class TestAdjustTargetNone:
    def test_none_strategy_passes_through(self):
        df = _make_df([5.0, 10.0], [0.0, 20.0])
        df = detect_censoring(df)
        cfg = CensoringConfig(strategy="none")
        result = adjust_target(df, cfg)
        assert list(result[COL_DEMAND_ADJUSTED]) == [5.0, 10.0]


class TestAdjustTargetDrop:
    def test_censored_rows_become_nan(self):
        df = _make_df([5.0, 10.0, 3.0], [0.0, 20.0, 5.0])
        df = detect_censoring(df)
        cfg = CensoringConfig(strategy="drop")
        result = adjust_target(df, cfg)
        # Row 0: zero stock → censored → NaN
        assert pd.isna(result.loc[0, COL_DEMAND_ADJUSTED])
        # Row 1: normal → not NaN
        assert result.loc[1, COL_DEMAND_ADJUSTED] == 10.0

    def test_non_censored_rows_unchanged(self):
        df = _make_df([5.0, 10.0], [20.0, 20.0])
        df = detect_censoring(df)
        cfg = CensoringConfig(strategy="drop")
        result = adjust_target(df, cfg)
        assert list(result[COL_DEMAND_ADJUSTED]) == [5.0, 10.0]


class TestAdjustTargetImpute:
    def test_imputed_never_below_observed(self):
        """Key correctness invariant: demand_adjusted >= sales_qty always."""
        sales = [10.0, 8.0, 12.0, 5.0, 9.0, 7.0, 10.0, 6.0]
        # Last row has zero stock → censored
        stocks = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 0.0]
        df = _make_df(sales, stocks)
        df = detect_censoring(df)
        cfg = CensoringConfig(strategy="impute", rolling_window=7, use_dow_grouping=False)
        result = adjust_target(df, cfg)
        # demand_adjusted must always be >= sales_qty
        assert (result[COL_DEMAND_ADJUSTED] >= result[COL_SALES_QTY]).all()

    def test_non_censored_rows_unchanged(self):
        """Non-censored rows: demand_adjusted == sales_qty."""
        sales = [10.0, 8.0, 12.0]
        stocks = [20.0, 20.0, 20.0]
        df = _make_df(sales, stocks)
        df = detect_censoring(df)
        cfg = CensoringConfig(strategy="impute", rolling_window=7, use_dow_grouping=False)
        result = adjust_target(df, cfg)
        assert list(result[COL_DEMAND_ADJUSTED]) == [10.0, 8.0, 12.0]

    def test_requires_is_censored_column(self):
        df = _make_df([5.0], [0.0])
        cfg = CensoringConfig(strategy="impute")
        with pytest.raises(ValueError, match="detect_censoring"):
            adjust_target(df, cfg)

    def test_invalid_strategy_raises(self):
        df = _make_df([5.0], [0.0])
        df = detect_censoring(df)
        cfg = CensoringConfig(strategy="invalid_strategy")
        with pytest.raises(ValueError, match="Unknown censoring strategy"):
            adjust_target(df, cfg)
