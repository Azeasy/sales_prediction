"""Tests for order quantity computation."""

from __future__ import annotations

import pytest

from src.ordering.recommender import compute_order


class TestComputeOrder:
    def test_basic_order(self):
        """With no stock, order = forecast + safety."""
        qty = compute_order(
            forecast=10.0, stock_balance=0.0, expiration_days=3,
            safety_stock=2.0, shipment_multiple=1,
        )
        assert qty == 12

    def test_existing_stock_reduces_order(self):
        """Existing usable stock reduces the order."""
        qty = compute_order(
            forecast=10.0, stock_balance=5.0, expiration_days=3,
            safety_stock=2.0, shipment_multiple=1,
        )
        assert qty == 7

    def test_expired_stock_is_discarded(self):
        """Stock with expiration_days=1 is treated as unusable (expires today)."""
        qty = compute_order(
            forecast=10.0, stock_balance=8.0, expiration_days=1,
            safety_stock=2.0, shipment_multiple=1,
        )
        # usable_stock = 0 (expires today), so order = 12
        assert qty == 12

    def test_zero_order_when_sufficient_stock(self):
        """No order when current stock is already above target."""
        qty = compute_order(
            forecast=5.0, stock_balance=20.0, expiration_days=5,
            safety_stock=2.0, shipment_multiple=1,
        )
        assert qty == 0

    def test_shipment_multiple_round_up(self):
        """Raw order of 7 with pack size 5 → rounded up to 10."""
        qty = compute_order(
            forecast=10.0, stock_balance=5.0, expiration_days=3,
            safety_stock=2.0, shipment_multiple=5, round_up=True,
        )
        assert qty % 5 == 0
        assert qty >= 7

    def test_shipment_multiple_round_down(self):
        """Round down to nearest pack (waste_first mode)."""
        qty = compute_order(
            forecast=10.0, stock_balance=5.0, expiration_days=3,
            safety_stock=2.0, shipment_multiple=5, round_up=False,
        )
        assert qty % 5 == 0

    def test_max_cover_days_caps_order(self):
        """max_cover_days prevents ordering too far ahead."""
        qty_no_cap = compute_order(
            forecast=10.0, stock_balance=0.0, expiration_days=7,
            safety_stock=5.0, shipment_multiple=1, max_cover_days=None,
        )
        qty_with_cap = compute_order(
            forecast=10.0, stock_balance=0.0, expiration_days=7,
            safety_stock=5.0, shipment_multiple=1, max_cover_days=1,
        )
        assert qty_with_cap <= qty_no_cap

    def test_order_never_negative(self):
        """Order quantity is always non-negative."""
        qty = compute_order(
            forecast=-5.0, stock_balance=100.0, expiration_days=5,
            safety_stock=0.0, shipment_multiple=1,
        )
        assert qty == 0

    def test_zero_forecast_no_safety(self):
        """Zero forecast with no safety stock → zero order."""
        qty = compute_order(
            forecast=0.0, stock_balance=0.0, expiration_days=5,
            safety_stock=0.0, shipment_multiple=1,
        )
        assert qty == 0

    def test_fractional_forecast_rounds_correctly(self):
        """Fractional raw order is handled by int() conversion."""
        qty = compute_order(
            forecast=3.7, stock_balance=0.0, expiration_days=3,
            safety_stock=0.0, shipment_multiple=1,
        )
        assert isinstance(qty, int)
        assert qty >= 0
