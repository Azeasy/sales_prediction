# Auto-Order MVP — Demand Forecasting & Order Recommendation

A CPU-only system for perishable retail auto-ordering.
Forecasts next-day demand per SKU, recommends order quantities accounting for
stock on hand and shelf life, and supports three configurable operating policies.

---

## Assignment Mapping

| Task Part | This Repo |
|-----------|-----------|
| Part A — Demand forecast | `src/models/`, `src/features/`, `src/demand/censoring.py` |
| Part B — Order quantity  | `src/ordering/recommender.py` |
| Part C — Dynamic policy (OOS vs Waste) | `src/ordering/policy.py`, `configs/policy_*.yaml` |
| Censored demand | `src/demand/censoring.py` (detect + 3 strategies) |
| Metrics (WAPE, Bias) | `src/evaluation/metrics.py` |
| Backtest / simulation | `src/evaluation/backtest.py` |
| CLI | `src/cli/main.py` |
| API access | `src/api/client.py`, `src/api/schemas.py` |

---

## Architecture Overview

```
APIs / Local Samples
      │
      ▼
  DataLoader           ← data/samples/ (local) or Forecasto API (live)
      │
      ▼
 DatasetBuilder        ← joins sales + stock + losses + products
      │                   keyed by (date, store_id, sku_code)
      ▼
data/processed/dataset.parquet
      │
      ├──► CensoringModule   ← detect stockout days; adjust training target
      │
      ├──► FeatureEngineering ← calendar, lags, rolling stats, metadata
      │
      ├──► LGBMForecaster     ← train / predict; artifacts/ saved artifacts
      │
      └──► PolicyLayer        ← quantile-adjust forecast + safety stock
                │
                └──► compute_order()   ← perishability-aware order formula
                          │
                          └──► order_qty (per SKU)
```

**Storage decision: Parquet files, no database.**
The dataset is small (single client, months of daily SKU-level data). Parquet provides columnar efficiency, schema preservation, and zero infrastructure overhead. Joins are simple pandas merges on `(date, sku_code)`. DuckDB can read Parquet natively for ad-hoc SQL queries without any schema migration.

---

## Data Sources

Four endpoints from the Forecasto platform (all behind `api.forecasto.ru`):

| Endpoint | What it provides | Internal table |
|----------|-----------------|----------------|
| `/sales` | Daily sales per SKU | `data/raw/sales.parquet` |
| `/inventory/stock` | Stock balance per SKU per date | `data/raw/stock.parquet` |
| `/backend/delivery_info/api/v1/GetAll` | Product metadata (shelf life, pack size) | `data/raw/products.parquet` |
| `/loss/getall` | Write-offs / expired goods | `data/raw/losses.parquet` |

All Russian field names are normalized to English in `src/api/schemas.py` before any downstream code sees them.

---

## Censored Demand Approach

**The problem:** When a product sells out, observed sales are capped by available stock — not by actual demand. Training a forecaster on raw sales in these conditions teaches the model "low stock = low demand," causing systematic under-forecasting and perpetuating the stockout cycle.

**Detection heuristic:**
```
is_censored = (stock_balance == 0)
           OR (stock_balance > 0 AND stock_balance <= sales_qty AND sales_qty > 0)
```

**Three config-driven strategies** (`censoring.strategy` in `configs/base.yaml`):

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `none` | Use raw `sales_qty` as target | **Default for bakery data** — daily sell-through to zero is normal, not a stockout |
| `drop` | Exclude censored rows from training | Clean but loses data |
| `impute` | Replace with rolling estimate ≥ observed | Use only if mid-day stockouts are confirmed from intra-day data |

**Imputation algorithm (`impute` strategy):**
1. For each SKU, mask censored rows (treat as NaN to avoid feedback loop).
2. Compute 7-day rolling median of uncensored sales (shifted 1 day to prevent leakage).
3. Optionally compute same-day-of-week rolling median for seasonal alignment.
4. `demand_adjusted = max(sales_qty, estimate)` — **never imputes below observed**.

**Why best forecast ≠ best order:**
Even a perfect point forecast doesn't yield the optimal order because:
- You already hold stock (reduces what you need to order)
- Products expire (ordering 3 days worth of a 2-day shelf-life item is wasteful)
- Packs come in multiples (rounding to nearest pack)
- Policy intent: `service_first` intentionally over-orders for safety margin

---

## Modeling Approach

**Model: LightGBM (`src/models/lgbm_model.py`)**
- Single global model across all SKUs (SKU-specific context via features)
- Trained as a regression task on `demand_adjusted` target
- CPU-only; trains in seconds on bakery-scale data
- Native categorical support for `product_group`, `day_of_week`
- Saves artifacts to `artifacts/`: model pkl, feature list JSON, importance CSV

**Baselines for comparison:**
- `NaiveModel`: predict yesterday's demand (`lag_1d`)
- `SeasonalNaiveModel`: predict same weekday last week (`lag_7d`)

**Feature groups (all leak-free):**

| Group | Features |
|-------|----------|
| Calendar | `day_of_week`, `is_weekend`, `day_of_month`, `month`, `week_of_year` |
| Lags | `lag_1d`, `lag_2d`, `lag_3d`, `lag_7d`, `lag_14d` |
| Rolling | `rolling_mean_7d/14d/28d`, `rolling_std_7d/14d/28d` |
| Censoring | `censored_rate_7d`, `days_since_last_stockout` |
| Metadata | `expiration_days`, `product_group` (categorical), `shipment_multiple` |

**Temporal train/val split:** Last 20% of dates are held out for validation. No shuffle — respects time ordering.

---

## Ordering Logic

Core function: `src/ordering/recommender.py::compute_order()`

```
usable_stock = stock_balance  if expiration_days > 1  else 0
target_stock = adjusted_forecast + safety_stock
target_stock = min(target_stock, forecast × max_cover_days)   # waste cap
target_stock = min(target_stock, forecast × expiration_days)   # perishability cap
raw_order    = max(0, target_stock - usable_stock)
order_qty    = round(raw_order, shipment_multiple, direction)
```

This function is **completely decoupled from the forecasting model.** It can accept any demand estimate.

---

## Policy Modes

Three meaningfully different operating modes controlled by YAML config:

| Parameter | `service_first` | `balanced` | `waste_first` |
|-----------|----------------|------------|---------------|
| `safety_stock_multiplier` | 1.5 | 1.0 | 0.3 |
| `forecast_quantile` | 0.85 | 0.55 | 0.30 |
| `max_cover_days` | expiration_days | expiration_days − 1 | 1 |
| `round_up_shipment` | True | True | False (round down) |

**`forecast_quantile` scaling:** The policy adjusts the point forecast by adding `z_score × std_estimate` before computing the order. At quantile 0.85, we order for a high-demand scenario. At 0.30, we order conservatively. This is transparent and doesn't require retraining.

---

## How to Run Locally

### Prerequisites

**macOS only** — LightGBM requires the OpenMP runtime:

```bash
brew install libomp
```

**All platforms** — install Python dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `FORECASTO_TOKEN` if using live API.

### Quick demo (sample data, no API needed)

```bash
# Step 1: Build the processed dataset from local sample files
python3 -m src.cli.main build-dataset

# Step 2: Train the LightGBM model
python3 -m src.cli.main train

# Step 3: Forecast next-day demand
python3 -m src.cli.main predict

# Step 4: Generate order recommendations (balanced policy)
python3 -m src.cli.main recommend-order --policy balanced

# Step 5: Run backtest simulation
python3 -m src.cli.main backtest --policy balanced
```

### Using the live API

```bash
# Set your token
export FORECASTO_TOKEN=your_token_here

# Edit configs/base.yaml: set data.source to "api"

# Fetch data from the API
python -m src.cli.main fetch-data --start-date 2026-01-01 --end-date 2026-03-07

# Then proceed as above
python -m src.cli.main build-dataset
python -m src.cli.main train
```

### Compare all three policies

```bash
for policy in service_first balanced waste_first; do
  echo "=== $policy ==="
  python -m src.cli.main backtest --policy $policy
done
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## How to Run in Docker

```bash
# Build the image
docker build -t auto-order-mvp .

# Build dataset (mounts local data dir)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  auto-order-mvp build-dataset

# Train
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  auto-order-mvp train

# Order recommendations with service_first policy
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  auto-order-mvp recommend-order --policy service_first

# With live API token
docker run --rm \
  -e FORECASTO_TOKEN=your_token \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  auto-order-mvp fetch-data --start-date 2026-01-01 --end-date 2026-03-07
```

---

## Assumptions

| # | Assumption | Rationale |
|---|-----------|-----------|
| 1 | **Single store** — no store field in API responses | API response has no store identifier; `store_id="default_store"` used. Architecture is multi-store ready. |
| 2 | **Daily granularity** | APIs return per-day records; no sub-day data available. |
| 3 | **Product metadata is static** | `GetAll` returns current state; assumed infrequent changes. |
| 4 | **Quantity forecasting, not revenue** | `Количество` (units sold) is the target, not `Сумма` (revenue). |
| 5 | **Shelf life starts at delivery** | `ExpirationDays` is total product shelf life from receipt. |
| 6 | **Same-day order delivery** | Backtest assumes orders arrive same day. In production, add lead-time offset. |
| 7 | **Loss records = expiry write-offs** | The `loss` endpoint captures expired/wasted goods, not theft or damage. |

---

## Limitations

- **Lead time not modeled**: The backtest assumes orders arrive the same day. Real deployments need a lead-time parameter (`order placed today, arrives in N days`).
- **No demand elasticity**: Price changes are not modeled. The system assumes demand is independent of pricing.
- **Single-step forecast only**: This is a next-day (h=1) system. Multi-step forecasting for order planning further into the future would require a different architecture.
- **No SKU-level model tuning**: One global LightGBM model handles all SKUs. SKUs with very unusual patterns may benefit from per-SKU models or ensemble.
- **Static product metadata**: If `ExpirationDays` or `Shipment` changes, the dataset must be rebuilt.
- **Simulation fidelity**: The backtest is a simplified single-period simulation. It does not model supplier constraints, delivery windows, or minimum order quantities.

---

## Future Improvements

- **Lead-time modeling**: Parameterize delivery lag; adjust ordering formula accordingly.
- **Probabilistic forecasting**: Replace point estimate with quantile regression (LightGBM supports `quantile` objective natively) to eliminate the heuristic CV assumption in policy quantile scaling.
- **Multi-store support**: The schema is already store-ready. Add store_id to API fetch calls when the platform supports multi-store.
- **Hyperparameter optimization**: Add Optuna sweep for LightGBM params, guided by validation WAPE.
- **Online retraining**: Schedule daily re-training on a rolling window. Add model versioning to artifacts.
- **Alerting**: Flag SKUs with sudden demand shifts (z-score anomaly on lag residuals) for manual review.
- **MLflow or similar tracking**: Log training runs, metrics, and artifacts for experiment comparison.
