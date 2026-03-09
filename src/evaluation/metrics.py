"""
Evaluation metrics for the auto-order system.

Forecast quality metrics
------------------------
WAPE (Weighted Absolute Percentage Error)
    = sum(|y - yhat|) / sum(y)
    Range: [0, ∞).  0 = perfect.  <0.2 = good for bakery demand.
    Weight by volume so high-selling SKUs matter more.
    Preferred over MAPE because MAPE is undefined when y=0 and inflates
    for low-volume SKUs.

Bias
    = sum(yhat - y) / sum(y)
    Range: (-∞, ∞).  0 = unbiased.  Positive = over-forecast (waste risk).
    Negative = under-forecast (stockout risk).
    Critical signal: a model with good WAPE but high bias still causes
    systematic operational problems.

Operational / decision quality metrics
---------------------------------------
These measure business outcomes, not forecast accuracy.

service_level
    = sum(fulfilled_demand) / sum(true_demand)
    Fraction of demand that was actually met. 1.0 = no stockouts.

waste_rate
    = total_waste / total_ordered
    Fraction of ordered goods that expired / were written off.

Note: why best forecast ≠ best order
--------------------------------------
Even a perfect forecast (WAPE=0) doesn't guarantee optimal ordering:
- Perishability: ordering 1 day early may mean stock expires
- Shipment multiples: you can't order 1.5 units of a pack-of-6 product
- Policy mode: service_first intentionally inflates orders for safety
- Uncertainty: ordering at the median forecast will produce ~50% stockout rate

The evaluation shows both forecast quality AND order quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|y - yhat|) / sum(y)

    Returns NaN if sum(y) == 0 (undefined on all-zero actuals).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    total = y_true.sum()
    if total == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / total)


def bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Forecast bias (relative over/under-prediction).

    Bias = sum(yhat - y) / sum(y)

    Positive → systematic over-forecast (waste risk).
    Negative → systematic under-forecast (stockout risk).
    Returns NaN if sum(y) == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    total = y_true.sum()
    if total == 0:
        return float("nan")
    return float((y_pred - y_true).sum() / total)


def service_level(
    true_demand: np.ndarray | pd.Series,
    fulfilled: np.ndarray | pd.Series,
) -> float:
    """
    Fraction of demand that was fulfilled.

    service_level = sum(min(demand, fulfilled)) / sum(demand)

    fulfilled is the quantity actually sold given available stock.
    Returns NaN if sum(demand) == 0.
    """
    d = np.asarray(true_demand, dtype=float)
    f = np.asarray(fulfilled, dtype=float)

    total_demand = d.sum()
    if total_demand == 0:
        return float("nan")
    return float(np.minimum(d, f).sum() / total_demand)


def waste_rate(
    ordered: np.ndarray | pd.Series,
    sold: np.ndarray | pd.Series,
    expired: np.ndarray | pd.Series | None = None,
) -> float:
    """
    Fraction of ordered goods that were wasted (expired or unsold).

    If expired is provided: waste_rate = sum(expired) / sum(ordered)
    Otherwise: waste_rate = sum(max(0, ordered - sold)) / sum(ordered)

    Returns NaN if sum(ordered) == 0.
    """
    o = np.asarray(ordered, dtype=float)
    total_ordered = o.sum()
    if total_ordered == 0:
        return float("nan")

    if expired is not None:
        w = np.asarray(expired, dtype=float).sum()
    else:
        s = np.asarray(sold, dtype=float)
        w = np.maximum(0, o - s).sum()
    return float(w / total_ordered)


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    fulfilled: np.ndarray | pd.Series | None = None,
    ordered: np.ndarray | pd.Series | None = None,
    expired: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """
    Compute all available metrics in one call.

    Returns a dict of metric name → value (NaN where undefined).
    """
    result: dict[str, float] = {
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if fulfilled is not None:
        result["service_level"] = service_level(y_true, fulfilled)
    if ordered is not None:
        result["waste_rate"] = waste_rate(ordered, y_true if fulfilled is None else fulfilled, expired)
    return result


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a human-readable string of metric values."""
    parts = []
    for name, val in metrics.items():
        if np.isnan(val):
            parts.append(f"{name}=N/A")
        elif name in ("wape", "waste_rate", "service_level"):
            parts.append(f"{name}={val:.2%}")
        else:
            parts.append(f"{name}={val:+.3f}")
    return "  ".join(parts)
