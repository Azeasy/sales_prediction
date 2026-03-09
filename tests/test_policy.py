"""
Tests for the policy layer.

Key assertions:
  - Different policies produce meaningfully different order quantities.
  - service_first produces >= balanced >= waste_first for the same forecast.
  - Policy parameters are applied correctly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ordering.policy import apply_policy, adjust_forecast_for_policy
from src.utils.config import PolicyConfig


def _make_forecast_df(forecast=10.0, stock=0.0, exp_days=3, shipment=1, n=1):
    """Create a minimal forecast DataFrame for policy tests."""
    rows = [{
        "date": pd.Timestamp("2026-03-01"),
        "sku_code": f"SKU-{i}",
        "forecast": forecast,
        "stock_balance": stock,
        "expiration_days": exp_days,
        "shipment_multiple": shipment,
        "rolling_std_7d": 2.0,
    } for i in range(n)]
    return pd.DataFrame(rows)


class TestAdjustForecastForPolicy:
    def test_quantile_50_no_change(self):
        cfg = PolicyConfig(mode="balanced", forecast_quantile=0.50)
        adj = adjust_forecast_for_policy(10.0, std_estimate=2.0, policy_cfg=cfg)
        assert adj == pytest.approx(10.0, abs=0.01)

    def test_high_quantile_increases_forecast(self):
        cfg = PolicyConfig(mode="service_first", forecast_quantile=0.85)
        adj = adjust_forecast_for_policy(10.0, std_estimate=2.0, policy_cfg=cfg)
        assert adj > 10.0

    def test_low_quantile_decreases_forecast(self):
        cfg = PolicyConfig(mode="waste_first", forecast_quantile=0.30)
        adj = adjust_forecast_for_policy(10.0, std_estimate=2.0, policy_cfg=cfg)
        assert adj < 10.0

    def test_never_below_zero(self):
        cfg = PolicyConfig(mode="waste_first", forecast_quantile=0.01)
        adj = adjust_forecast_for_policy(0.0, std_estimate=5.0, policy_cfg=cfg)
        assert adj >= 0.0

    def test_fallback_cv_when_no_std(self):
        """Should still work without std estimate."""
        cfg = PolicyConfig(mode="service_first", forecast_quantile=0.85)
        adj = adjust_forecast_for_policy(10.0, std_estimate=None, policy_cfg=cfg)
        assert adj > 10.0


class TestApplyPolicy:
    def test_service_first_orders_more_than_balanced(self):
        df = _make_forecast_df(forecast=10.0, stock=0.0, exp_days=5)

        svc = PolicyConfig(mode="service_first", safety_stock_multiplier=1.5,
                           forecast_quantile=0.85, max_cover_days=None, round_up_shipment=True)
        bal = PolicyConfig(mode="balanced", safety_stock_multiplier=1.0,
                           forecast_quantile=0.55, max_cover_days=None, round_up_shipment=True)

        order_svc = apply_policy(df, svc)["order_qty"].iloc[0]
        order_bal = apply_policy(df, bal)["order_qty"].iloc[0]

        assert order_svc >= order_bal, f"service_first={order_svc} should >= balanced={order_bal}"

    def test_balanced_orders_more_than_waste_first(self):
        df = _make_forecast_df(forecast=10.0, stock=0.0, exp_days=5)

        bal = PolicyConfig(mode="balanced", safety_stock_multiplier=1.0,
                           forecast_quantile=0.55, max_cover_days=None, round_up_shipment=True)
        waste = PolicyConfig(mode="waste_first", safety_stock_multiplier=0.3,
                             forecast_quantile=0.30, max_cover_days=1, round_up_shipment=False)

        order_bal = apply_policy(df, bal)["order_qty"].iloc[0]
        order_waste = apply_policy(df, waste)["order_qty"].iloc[0]

        assert order_bal >= order_waste, f"balanced={order_bal} should >= waste_first={order_waste}"

    def test_output_has_required_columns(self):
        df = _make_forecast_df(forecast=10.0)
        cfg = PolicyConfig(mode="balanced")
        result = apply_policy(df, cfg)
        assert "order_qty" in result.columns
        assert "adjusted_forecast" in result.columns
        assert "safety_stock" in result.columns

    def test_order_qty_is_non_negative(self):
        df = _make_forecast_df(forecast=0.0, stock=100.0, exp_days=3, n=5)
        cfg = PolicyConfig(mode="service_first", safety_stock_multiplier=1.5, forecast_quantile=0.85)
        result = apply_policy(df, cfg)
        assert (result["order_qty"] >= 0).all()

    def test_shipment_multiple_respected(self):
        df = _make_forecast_df(forecast=10.0, stock=0.0, exp_days=5, shipment=6)
        cfg = PolicyConfig(mode="balanced", round_up_shipment=True)
        result = apply_policy(df, cfg)
        assert result["order_qty"].iloc[0] % 6 == 0

    def test_waste_first_round_down_shipment(self):
        df = _make_forecast_df(forecast=10.0, stock=0.0, exp_days=5, shipment=6)
        cfg = PolicyConfig(mode="waste_first", safety_stock_multiplier=0.3,
                           forecast_quantile=0.30, max_cover_days=1, round_up_shipment=False)
        result = apply_policy(df, cfg)
        assert result["order_qty"].iloc[0] % 6 == 0  # Still a multiple


class TestPoliciesProduceDifferentBehavior:
    """Ensures the three policies are not just renamed aliases."""

    def test_all_three_policies_differ(self):
        df = _make_forecast_df(forecast=15.0, stock=0.0, exp_days=5, shipment=1)

        orders = {}
        for mode, mult, q, mcd, ru in [
            ("service_first", 1.5, 0.85, None, True),
            ("balanced", 1.0, 0.55, None, True),
            ("waste_first", 0.3, 0.30, 1, False),
        ]:
            cfg = PolicyConfig(mode=mode, safety_stock_multiplier=mult,
                               forecast_quantile=q, max_cover_days=mcd, round_up_shipment=ru)
            orders[mode] = apply_policy(df, cfg)["order_qty"].iloc[0]

        # All three should differ
        vals = list(orders.values())
        assert not all(v == vals[0] for v in vals), f"All policies produced same order: {orders}"
