"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import bias, compute_all_metrics, service_level, wape, waste_rate


class TestWAPE:
    def test_perfect_forecast(self):
        y = np.array([10.0, 20.0, 30.0])
        assert wape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([12.0, 8.0])
        # |10-12| + |10-8| = 2+2 = 4; sum(y_true) = 20 → WAPE = 0.2
        assert wape(y_true, y_pred) == pytest.approx(0.2)

    def test_all_zeros_returns_nan(self):
        assert math.isnan(wape(np.zeros(5), np.zeros(5)))

    def test_zero_actual_non_zero_pred(self):
        assert math.isnan(wape(np.zeros(3), np.array([1.0, 2.0, 3.0])))

    def test_with_series(self):
        import pandas as pd
        y_true = pd.Series([5.0, 10.0, 15.0])
        y_pred = pd.Series([5.0, 10.0, 15.0])
        assert wape(y_true, y_pred) == pytest.approx(0.0)

    def test_complete_miss(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([0.0, 0.0])
        # |10| + |10| = 20; sum = 20 → WAPE = 1.0
        assert wape(y_true, y_pred) == pytest.approx(1.0)


class TestBias:
    def test_unbiased(self):
        y = np.array([10.0, 20.0])
        assert bias(y, y) == pytest.approx(0.0)

    def test_over_forecast(self):
        # Predicting 20% more than actual → positive bias
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([12.0, 12.0])
        assert bias(y_true, y_pred) == pytest.approx(0.2)

    def test_under_forecast(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([8.0, 8.0])
        assert bias(y_true, y_pred) == pytest.approx(-0.2)

    def test_zero_denominator(self):
        assert math.isnan(bias(np.zeros(3), np.array([1.0, 2.0, 3.0])))


class TestServiceLevel:
    def test_full_service(self):
        demand = np.array([10.0, 20.0])
        fulfilled = np.array([10.0, 20.0])
        assert service_level(demand, fulfilled) == pytest.approx(1.0)

    def test_partial_service(self):
        demand = np.array([10.0, 10.0])
        fulfilled = np.array([5.0, 10.0])
        # (5+10) / (10+10) = 0.75
        assert service_level(demand, fulfilled) == pytest.approx(0.75)

    def test_no_service(self):
        demand = np.array([10.0, 10.0])
        fulfilled = np.array([0.0, 0.0])
        assert service_level(demand, fulfilled) == pytest.approx(0.0)

    def test_zero_demand_returns_nan(self):
        assert math.isnan(service_level(np.zeros(3), np.zeros(3)))


class TestWasteRate:
    def test_no_waste(self):
        ordered = np.array([10.0, 20.0])
        sold = np.array([10.0, 20.0])
        assert waste_rate(ordered, sold) == pytest.approx(0.0)

    def test_known_waste(self):
        ordered = np.array([10.0, 10.0])
        sold = np.array([8.0, 6.0])
        # waste = (10-8) + (10-6) = 2+4 = 6; ordered = 20 → rate = 0.3
        assert waste_rate(ordered, sold) == pytest.approx(0.3)

    def test_with_expired(self):
        ordered = np.array([10.0])
        sold = np.array([7.0])
        expired = np.array([3.0])
        assert waste_rate(ordered, sold, expired) == pytest.approx(0.3)

    def test_zero_ordered_returns_nan(self):
        assert math.isnan(waste_rate(np.zeros(3), np.zeros(3)))


class TestComputeAllMetrics:
    def test_returns_dict_with_expected_keys(self):
        y = np.array([10.0, 20.0])
        result = compute_all_metrics(y, y)
        assert "wape" in result
        assert "bias" in result

    def test_with_operational_metrics(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([10.0, 10.0])
        fulfilled = np.array([10.0, 10.0])
        ordered = np.array([12.0, 12.0])
        result = compute_all_metrics(y_true, y_pred, fulfilled=fulfilled, ordered=ordered)
        assert "service_level" in result
        assert "waste_rate" in result
        assert result["service_level"] == pytest.approx(1.0)
